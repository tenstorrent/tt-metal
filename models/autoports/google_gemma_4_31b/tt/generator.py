# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Readiness and serving generator for the optimized TP4 Gemma 4 31B model."""

from __future__ import annotations

import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_31b.tt.functional_decoder import HF_MODEL_ID
from models.autoports.google_gemma_4_31b.tt.model import (
    Gemma4FullModel,
    Gemma4FullModelConfig,
    _kv_cache_identity,
    _resolve_checkpoint,
    resolve_eos_token_ids,
)
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.readiness_check.contract import Generator, NextInputFn
from models.common.sampling import SamplingParams, format_sampling_params

_ARGMAX_TILE_KERNEL = "models/autoports/google_gemma_4_31b/tt/kernels/gemma4_argmax_tile_local_winner.cpp"
_ARGMAX_PAIR_REDUCE_KERNEL = "models/autoports/google_gemma_4_31b/tt/kernels/gemma4_argmax_pair_reduce.cpp"
_SAMPLING_SEED_SKIP = 2**32 - 1


def _read_first_mesh_token(token_tensor: ttnn.Tensor) -> int:
    return int(ttnn.to_torch(ttnn.get_device_tensors(token_tensor)[0]).reshape(-1)[0].item())


def _core_grid_for_first_n(device, num_cores: int) -> tuple[ttnn.CoreRangeSet, list[ttnn.CoreCoord]]:
    grid_size = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)][:num_cores]
    return ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in cores}), cores


class Gemma4GreedyTP4Sampler:
    """Traceable greedy TP4 maxloc with deterministic lower-token tie breaking."""

    def __init__(self, *, mesh_device, vocab_per_device: int, max_batch_size: int, num_cores: int = 8) -> None:
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        if tuple(mesh_device.shape) != (1, 4) or self.num_devices != 4:
            raise ValueError(f"Gemma4GreedyTP4Sampler requires 1x4 TP4 mesh, got {tuple(mesh_device.shape)}")
        if vocab_per_device % 32 != 0:
            raise ValueError(f"vocab_per_device must be tile-aligned, got {vocab_per_device}")
        if max_batch_size < 1 or max_batch_size > 32:
            raise ValueError(f"max_batch_size must be in [1, 32], got {max_batch_size}")
        self.vocab_per_device = int(vocab_per_device)
        self.max_batch_size = int(max_batch_size)
        self.total_tiles = self.vocab_per_device // 32
        sample_tensor = ttnn.from_torch(
            torch.zeros((self.num_devices, 1), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        sample_device = ttnn.get_device_tensors(sample_tensor)[0].device()
        self.core_grid, self.cores = _core_grid_for_first_n(sample_device, num_cores)
        sample_tensor.deallocate(True)
        self.final_core = self.cores[-1]
        self.output_grid = ttnn.CoreRangeSet({ttnn.CoreRange(self.final_core, self.final_core)})
        self.pair_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.output_grid, (self.max_batch_size, 2), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.gather_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.output_grid, (self.num_devices * self.max_batch_size, 2), ttnn.ShardOrientation.ROW_MAJOR
            ),
        )
        self.local_pairs = ttnn.from_torch(
            torch.zeros((self.num_devices, self.max_batch_size, 2), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=self.pair_mem,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        self.gathered_pairs = ttnn.from_torch(
            torch.zeros((self.num_devices, self.max_batch_size, 2), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=self.gather_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        full_grid = mesh_device.compute_with_storage_grid_size()
        sem_core_range = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))]
        )
        self.receiver_sem = ttnn.create_global_semaphore(mesh_device, sem_core_range, 0)
        self.receiver_addr = int(ttnn.get_global_semaphore_address(self.receiver_sem))

    def _tile_winner_program(
        self, *, device, scores, scores_addr: int, output_pair_addr: int, vocab_offset: int, active_batch_size: int
    ):
        tile_cb = 0
        gather_cb = 1
        tile_scratch_bytes = 2048
        winner_page_bytes = 64 + 128
        # A two-u32 row is physically 16-byte aligned in this height-sharded
        # L1 tensor. Using the logical 8-byte width overwrites row-0 padding
        # instead of producing the next fixed-slot pair.
        output_pair_page_bytes = 16
        tiles_per_sender = math.ceil(self.total_tiles / len(self.cores))
        final_worker = device.worker_core_from_logical_core(self.final_core)
        tensor_accessor_args = ttnn.TensorAccessorArgs(scores)
        kernels = []
        for sender_idx, core in enumerate(self.cores):
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=_ARGMAX_TILE_KERNEL,
                    core_ranges=ttnn.CoreRangeSet({ttnn.CoreRange(core, core)}),
                    compile_time_args=[
                        self.total_tiles,
                        tiles_per_sender,
                        tile_scratch_bytes,
                        winner_page_bytes,
                        len(self.cores),
                        len(self.cores) - 1,
                        self.receiver_addr,
                        tile_cb,
                        gather_cb,
                        sender_idx,
                        1 if core == self.final_core else 0,
                        int(active_batch_size),
                        output_pair_page_bytes,
                        *tensor_accessor_args.get_compile_time_args(),
                    ],
                    common_runtime_args=[
                        int(scores_addr),
                        int(output_pair_addr),
                        int(final_worker.x),
                        int(final_worker.y),
                        int(vocab_offset),
                    ],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
        return ttnn.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                ttnn.CBDescriptor(
                    total_size=tile_scratch_bytes,
                    core_ranges=self.core_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=tile_cb, data_format=ttnn.bfloat16, page_size=tile_scratch_bytes
                        )
                    ],
                ),
                ttnn.CBDescriptor(
                    total_size=winner_page_bytes * len(self.cores) * active_batch_size,
                    core_ranges=self.core_grid,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(
                            buffer_index=gather_cb, data_format=ttnn.uint32, page_size=winner_page_bytes
                        )
                    ],
                ),
            ],
            semaphores=[],
        )

    def _pair_reduce_program(self, *, gathered_pairs, output_token, active_batch_size: int):
        scratch_cb = 0
        pair_payload_bytes = 8
        scratch_page_bytes = 64
        gathered_accessor_args = ttnn.TensorAccessorArgs(gathered_pairs)
        output_accessor_args = ttnn.TensorAccessorArgs(output_token)
        kernel = ttnn.KernelDescriptor(
            kernel_source=_ARGMAX_PAIR_REDUCE_KERNEL,
            core_ranges=ttnn.CoreRangeSet({ttnn.CoreRange(self.final_core, self.final_core)}),
            compile_time_args=[
                self.num_devices,
                pair_payload_bytes,
                scratch_cb,
                scratch_page_bytes,
                int(active_batch_size),
                self.max_batch_size,
                *gathered_accessor_args.get_compile_time_args(),
                *output_accessor_args.get_compile_time_args(),
            ],
            common_runtime_args=[int(gathered_pairs.buffer_address()), int(output_token.buffer_address())],
            config=ttnn.ReaderConfigDescriptor(),
        )
        scratch = ttnn.CBDescriptor(
            total_size=scratch_page_bytes * self.num_devices,
            core_ranges=ttnn.CoreRangeSet({ttnn.CoreRange(self.final_core, self.final_core)}),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=scratch_cb, data_format=ttnn.uint32, page_size=scratch_page_bytes)
            ],
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch], semaphores=[])

    def decode_forward(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor):
        active_batch_size = int(tt_out_tok.shape[-1])
        if active_batch_size < 1 or active_batch_size > self.max_batch_size:
            raise ValueError(f"active batch size must be in [1, {self.max_batch_size}], got {active_batch_size}")
        scores_per_device = ttnn.get_device_tensors(logits)
        pairs_per_device = ttnn.get_device_tensors(self.local_pairs)
        mesh_program = ttnn.MeshProgramDescriptor()
        for col in range(self.num_devices):
            coord = ttnn.MeshCoordinate(0, col)
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = self._tile_winner_program(
                device=scores_per_device[col].device(),
                scores=scores_per_device[col],
                scores_addr=int(scores_per_device[col].buffer_address()),
                output_pair_addr=int(pairs_per_device[col].buffer_address()),
                vocab_offset=col * self.vocab_per_device,
                active_batch_size=active_batch_size,
            )
        ttnn.generic_op([logits, self.local_pairs], mesh_program)
        gathered = ttnn.all_gather(
            self.local_pairs,
            dim=0,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=self.gather_mem,
            output_tensor=self.gathered_pairs,
        )
        gathered_per_device = ttnn.get_device_tensors(gathered)
        tokens_per_device = ttnn.get_device_tensors(tt_out_tok)
        reduce_program = ttnn.MeshProgramDescriptor()
        for col in range(self.num_devices):
            coord = ttnn.MeshCoordinate(0, col)
            reduce_program[ttnn.MeshCoordinateRange(coord, coord)] = self._pair_reduce_program(
                gathered_pairs=gathered_per_device[col],
                output_token=tokens_per_device[col],
                active_batch_size=active_batch_size,
            )
        ttnn.generic_op([gathered, tt_out_tok], reduce_program)
        return tt_out_tok, None

    def teardown(self) -> None:
        for tensor in (self.gathered_pairs, self.local_pairs):
            if tensor.is_allocated():
                tensor.deallocate(True)


_GREEDY_SAMPLING = (1, 0.0, 1.0)
_SAMPLING_SLOTS = 32  # Sampling1D slot tensors are tile-shaped


class Gemma4Generator(Generator):
    """Two-level generator with explicit external state and traced token feedback."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_mixed_prompt_lengths": True,
        "supports_inactive_rows": True,
        "supports_on_device_sampling": True,
    }

    def __init__(
        self,
        *,
        model_dir: str | Path,
        mesh_device,
        model_id_or_path: str | Path = HF_MODEL_ID,
        model: Gemma4FullModel | None = None,
        model_config: Gemma4FullModelConfig | None = None,
        tokenizer=None,
        host_sampling_compat: bool = False,
        max_batch_size: int | None = None,
        cache_context: int | None = None,
        tensor_cache_path: str | Path | None = None,
        allocate_standalone_cache: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        if model_config is None and model is not None:
            model_config = model.config
        requested_max_batch = 1 if max_batch_size is None else int(max_batch_size)
        self.model_config = model_config or Gemma4FullModelConfig(max_batch_size=requested_max_batch)
        self.max_batch_size = int(self.model_config.max_batch_size if max_batch_size is None else max_batch_size)
        if self.max_batch_size != self.model_config.max_batch_size:
            raise ValueError("generator and model max_batch_size must match")
        checkpoint = _resolve_checkpoint(model_id_or_path)
        if tensor_cache_path is None:
            tensor_cache_path = os.environ.get("GEMMA4_31B_TENSOR_CACHE")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=True
        )
        # The base checkpoint stops on one id, the instruction-tuned one on three.
        self._eos_token_ids = resolve_eos_token_ids(self.tokenizer, checkpoint)
        self.model = model or Gemma4FullModel.from_pretrained(
            mesh_device=mesh_device,
            model_id_or_path=checkpoint,
            config=self.model_config,
            tensor_cache_path=tensor_cache_path,
        )
        if self.max_batch_size != self.model.config.max_batch_size:
            raise ValueError("generator and model max_batch_size must match")
        self.host_sampling_compat = bool(host_sampling_compat)
        self.cache_context = int(cache_context or self.model.config.max_seq_len)
        self._owns_standalone_cache = bool(allocate_standalone_cache)
        if self._owns_standalone_cache:
            self.kv_cache, self.page_tables = self.model.allocate_paged_kv_cache(
                max_context=self.cache_context, batch_size=self.max_batch_size
            )
        else:
            # vLLM owns serving-cache sizing and allocation.  Keeping these
            # handles unset makes an accidental fallback to the standalone
            # cache fail at the ownership boundary instead of silently using a
            # second cache with incompatible block tables.
            self.kv_cache = None
            self.page_tables = None
        self.page_table_generation = 0
        self._cache_dirty = False
        self.timings: dict[str, float] = {}

        sampling_config = Sampling1DConfig(
            vocab_size=self.model.vocab_size,
            mesh_device=mesh_device,
            max_batch_size=self.max_batch_size,
            max_top_k=32,
            allow_force_argmax=False,
            pad_to_power_of_2=False,
            ag_topology=ttnn.Topology.Linear,
            num_gather_links=2,
            sampling_cluster_axis=1,
            use_broadcast_all_gather=True,
            gather_values_dtype=self.model.config.sampling_dtype,
        )
        self.sampler = Sampling1D.from_config(sampling_config)
        self.force_argmax_sampler = Sampling1D.from_config(
            Sampling1DConfig(
                **{
                    **sampling_config.__dict__,
                    "allow_force_argmax": True,
                }
            )
        )
        self.sampler.load_device_buffers()
        self.force_argmax_sampler.load_device_buffers()
        self.greedy_tp4_sampler = Gemma4GreedyTP4Sampler(
            mesh_device=mesh_device,
            vocab_per_device=self.model.vocab_per_device,
            max_batch_size=self.max_batch_size,
        )
        # Prefill returns one sampler-ready row per active prompt. Standalone
        # traced sampling defaults to max_batch_size; serving may explicitly
        # trace a smaller physical batch. Lazily materialize the common sampler
        # for smaller eager batches before trace capture.
        self._eager_samplers: dict[tuple[int, bool], Sampling1D] = {}
        self._sampling_trace_id: int | None = None
        self._sampling_trace_output: tuple[ttnn.Tensor, Any] | None = None
        self._sampling_trace_logits: ttnn.Tensor | None = None
        self._sampling_trace_key: tuple[int, int, float, float, bool] | None = None
        self._sampling_params: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None
        self.trace_lifecycle_counters = {
            "release_calls": 0,
            "release_synchronizations": 0,
            "sampling_seed_initializations": 0,
            "sampling_seed_host_copies": 0,
        }

    def _new_token_buffer(self, batch_size: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.zeros((1, 1, 1, batch_size), dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _resolve_cache_pair(self, page_table, kv_cache):
        """Use caller-owned cache state only when both ownership handles are explicit."""
        if (page_table is None) != (kv_cache is None):
            raise ValueError("page_table and kv_cache must either both be provided or both be omitted")
        if page_table is None:
            if not getattr(self, "_owns_standalone_cache", True):
                raise ValueError("serving mode requires caller-owned page_table and kv_cache handles")
            return self.page_tables, self.kv_cache, False
        return page_table, kv_cache, True

    def _normalize_decode_slots(
        self,
        positions: torch.Tensor,
        *,
        prompt_lengths: Sequence[int] | None,
        active_batch_size: int | None,
    ) -> tuple[tuple[int, ...], int]:
        values = tuple(int(value) for value in positions.reshape(-1).tolist())
        if any(value < -1 for value in values):
            raise ValueError("inactive decode slots must use position -1")
        active = sum(value >= 0 for value in values)
        if active_batch_size is None:
            active_batch_size = active
        if int(active_batch_size) != active:
            raise ValueError("active_batch_size must equal the number of non-negative decode positions")
        if active < 1:
            raise ValueError("at least one decode slot must be active")
        if any(value >= self.cache_context for value in values if value >= 0):
            raise ValueError("active decode position exceeds the configured cache context")
        if prompt_lengths is None:
            normalized_lengths = tuple(max(value, 0) for value in values)
        else:
            normalized_lengths = tuple(int(value) for value in prompt_lengths)
            if len(normalized_lengths) != len(values):
                raise ValueError("prompt_lengths must contain one value per fixed decode slot")
            for position, prompt_length in zip(values, normalized_lengths):
                if prompt_length < 0 or (position < 0 and prompt_length != 0):
                    raise ValueError(
                        "inactive slots require prompt length 0 and all prompt lengths must be non-negative"
                    )
                if position >= 0 and prompt_length > position:
                    raise ValueError("an active prompt length cannot exceed its current decode position")
        return normalized_lengths, active

    def _validate_next_trace_position(self) -> None:
        state = self.model.trace_state
        if state.initial_positions is None:
            raise RuntimeError("decode trace positions are not initialized")
        replay = int(state.counters["model_trace_replays"]) - int(state.initial_replay_count)
        positions = state.initial_positions.reshape(-1)
        active = positions >= 0
        if active.any() and int((positions[active] + replay).max()) >= self.cache_context:
            raise ValueError("decode would exceed the configured cache context")

    def _validate_generation_window(self, prompt_length: int, max_new_tokens: int) -> None:
        if prompt_length < 1:
            raise ValueError("prompt must be non-empty")
        if prompt_length + max(max_new_tokens - 1, 0) > self.cache_context:
            raise ValueError("prompt and requested generation exceed the configured cache context")

    def _make_sampling_params(
        self, *, batch_size: int, top_k: int, top_p: float, temperature: float
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        if not 1 <= top_k <= 32:
            raise ValueError("top_k must be in [1, 32]")

        # top_k=1, top_p=0, temperature=1 is semantically greedy while the
        # physical local candidate tensor remains tile-shaped at max_top_k=32.
        def device(source, dtype):
            return ttnn.from_torch(
                source,
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return (
            device(torch.full((batch_size,), top_k, dtype=torch.int32), ttnn.uint32),
            device(torch.full((batch_size,), top_p, dtype=torch.bfloat16), ttnn.bfloat16),
            device(torch.full((batch_size,), temperature, dtype=torch.bfloat16), ttnn.bfloat16),
        )

    def _normalize_sampling_seeds(
        self,
        seed: int | Sequence[int] | torch.Tensor | None,
        *,
        active_slots: Sequence[bool],
        physical_batch_size: int,
    ) -> list[int]:
        """Resolve request seeds while keeping inactive fixed slots at the skip sentinel."""
        if physical_batch_size > self.max_batch_size or len(active_slots) != physical_batch_size:
            raise ValueError("sampling seed slots must match the physical decode batch")
        if seed is None:
            requested = [secrets.randbelow(_SAMPLING_SEED_SKIP) for _ in range(sum(active_slots))]
        elif isinstance(seed, (int, torch.Tensor)) and (not isinstance(seed, torch.Tensor) or seed.numel() == 1):
            base = int(seed if isinstance(seed, int) else seed.reshape(-1)[0].item())
            if not 0 <= base < _SAMPLING_SEED_SKIP:
                raise ValueError("sampling seeds must be uint32 values other than UINT32_MAX")
            requested = [(base + index) % _SAMPLING_SEED_SKIP for index in range(sum(active_slots))]
        else:
            requested = [int(value) for value in torch.as_tensor(seed, dtype=torch.int64).reshape(-1).tolist()]
            if len(requested) not in (sum(active_slots), physical_batch_size):
                raise ValueError("sampling seeds must contain one value per active row or fixed slot")
            if len(requested) == physical_batch_size:
                requested = [value for value, active in zip(requested, active_slots) if active]
            if any(value < 0 or value >= _SAMPLING_SEED_SKIP for value in requested):
                raise ValueError("sampling seeds must be uint32 values other than UINT32_MAX")

        values = []
        active_index = 0
        for active in active_slots:
            if active:
                values.append(requested[active_index])
                active_index += 1
            else:
                values.append(_SAMPLING_SEED_SKIP)
        values.extend([_SAMPLING_SEED_SKIP] * (self.max_batch_size - physical_batch_size))
        return values

    def _copy_sampler_seeds(self, sampler: Sampling1D, values: Sequence[int]) -> None:
        if len(values) != int(sampler.config.max_batch_size):
            raise ValueError("sampler seed values must match the sampler's fixed slot count")
        host = ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, sampler._seeds)
        self.trace_lifecycle_counters["sampling_seed_host_copies"] += 1

    def _initialize_non_greedy_rng(self, values: Sequence[int], *, sampler: Sampling1D | None = None) -> None:
        """Seed once at a request boundary, then make traced manual_seed a device-side no-op."""
        sampler = self.sampler if sampler is None else sampler
        self._copy_sampler_seeds(sampler, values)
        ttnn.manual_seed(
            seeds=sampler._seeds,
            user_ids=sampler._user_ids,
            sub_core_grids=sampler.config.sub_core_grids,
        )
        # The real seed values must be consumed before their persistent buffer
        # is replaced with the skip sentinel. This synchronization is once per
        # request, never in the steady token loop.
        ttnn.synchronize_device(self.mesh_device)
        self.trace_lifecycle_counters["sampling_seed_initializations"] += 1
        self._copy_sampler_seeds(sampler, [_SAMPLING_SEED_SKIP] * int(sampler.config.max_batch_size))

    def _release_all_decode_traces(self) -> None:
        """Release both trace programs before freeing either trace's buffers."""
        self.trace_lifecycle_counters["release_calls"] += 1
        if self._sampling_trace_id is not None or self.model.trace_state.trace_id is not None:
            # Model and sampler replay are intentionally nonblocking.  Drain
            # CQ0 before releasing their programs or deallocating any bound
            # buffers; the following prefill/layout transition is allowed to
            # allocate only after both traces are gone.
            ttnn.synchronize_device(self.mesh_device)
            self.trace_lifecycle_counters["release_synchronizations"] += 1
        if self._sampling_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._sampling_trace_id)
        self._sampling_trace_id = None
        self.model.release_decode_trace()
        self._sampling_trace_output = None
        self._sampling_trace_logits = None
        self._sampling_trace_key = None
        if self._sampling_params is not None:
            for parameter in self._sampling_params:
                if parameter.is_allocated():
                    parameter.deallocate(True)
        self._sampling_params = None

    def _get_eager_sampler(self, batch_size: int, *, force_argmax: bool) -> Sampling1D:
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("eager sampling batch must be within max_batch_size")
        if batch_size == self.max_batch_size:
            return self.force_argmax_sampler if force_argmax else self.sampler
        key = (batch_size, bool(force_argmax))
        if key not in self._eager_samplers:
            if self.model.trace_state.trace_id is not None or self._sampling_trace_id is not None:
                raise RuntimeError("eager sampler buffers must be allocated before decode trace capture")
            sampler = Sampling1D.from_config(
                Sampling1DConfig(
                    vocab_size=self.model.vocab_size,
                    mesh_device=self.mesh_device,
                    max_batch_size=batch_size,
                    max_top_k=32,
                    allow_force_argmax=force_argmax,
                    pad_to_power_of_2=False,
                    ag_topology=ttnn.Topology.Linear,
                    num_gather_links=2,
                    sampling_cluster_axis=1,
                    use_broadcast_all_gather=True,
                    gather_values_dtype=self.model.config.sampling_dtype,
                )
            )
            sampler.load_device_buffers()
            self._eager_samplers[key] = sampler
        return self._eager_samplers[key]

    def _sample_eager(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        force_argmax: bool = False,
        seed: int | Sequence[int] | torch.Tensor | None = None,
        allow_live_traces: bool = False,
    ):
        # Eager sampling touches only pre-existing persistent sampler objects
        # and transient parameter tensors, both safe alongside live decode
        # traces; the RNG stream it advances is re-seeded at the next request
        # boundary by prepare/reuse. A caller may opt in with
        # allow_live_traces only when this batch size and sampling mode have
        # already run eagerly once (programs compiled, buffers allocated);
        # _get_eager_sampler still hard-fails on any first-time allocation.
        if not allow_live_traces and (
            self.model.trace_state.trace_id is not None or self._sampling_trace_id is not None
        ):
            raise RuntimeError("eager sampling is not allowed while decode traces are live")
        batch_size = int(tt_out_tok.shape[-1])
        output_prefix = tuple(int(tt_out_tok.shape[index]) for index in range(3))
        if output_prefix != (1, 1, 1) or int(logits.shape[-2]) != batch_size:
            raise ValueError("eager logits and output token buffers must have the same batch size")
        sampler = self._get_eager_sampler(batch_size, force_argmax=force_argmax)
        if force_argmax:
            return sampler.decode_forward(logits, tt_out_tok=tt_out_tok)
        if self._is_semantic_greedy(top_k=top_k, top_p=top_p, temperature=temperature):
            return self.greedy_tp4_sampler.decode_forward(logits, tt_out_tok=tt_out_tok)
        seeds = self._normalize_sampling_seeds(
            seed,
            active_slots=[True] * batch_size,
            physical_batch_size=batch_size,
        )
        self._initialize_non_greedy_rng(seeds[:batch_size], sampler=sampler)
        params = self._make_sampling_params(batch_size=batch_size, top_k=top_k, top_p=top_p, temperature=temperature)
        try:
            return sampler.decode_forward(logits, k=params[0], p=params[1], temp=params[2], tt_out_tok=tt_out_tok)
        finally:
            for parameter in params:
                parameter.deallocate(True)

    @staticmethod
    def _is_semantic_greedy(*, top_k: int, top_p: float, temperature: float) -> bool:
        return int(top_k) == 1 and float(top_p) == 0.0 and float(temperature) == 1.0

    @staticmethod
    def resolve_sampling(sampling_params) -> tuple[int, float, float]:
        """Normalise request sampling params into this generator's (top_k, top_p, temperature).

        Delegates to the platform formatter tt_transformers uses, which clamps
        top_k into the sampler's [1, 32] (k < 1 meaning unrestricted -> 32) and
        inverts temperature, the form ttnn.sampling wants: its kernel multiplies
        the top-k values by temp (sampling.cpp:465).
        """
        if sampling_params is None:
            return _GREEDY_SAMPLING

        def values(name: str, default: Any) -> list[Any]:
            value = getattr(sampling_params, name, default)
            if isinstance(value, torch.Tensor):
                return value.reshape(-1).tolist()
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
            return [value]

        temperatures, top_ks, top_ps = values("temperature", 0.0), values("top_k", 1), values("top_p", 1.0)
        if len(temperatures) != len(top_ks):
            raise ValueError("temperature and top_k must contain one value per fixed slot")
        if len(top_ps) == 1:
            top_ps = top_ps * len(temperatures)

        formatted = format_sampling_params(
            SamplingParams(temperature=list(temperatures), top_k=list(top_ks), top_p=list(top_ps)),
            _SAMPLING_SLOTS,
        )
        slots = len(temperatures)
        requested = {
            (int(k), float(p), float(t))
            for k, p, t in zip(formatted.top_k[:slots], formatted.top_p[:slots], formatted.temperature[:slots])
        }
        if len(requested) > 1:
            raise ValueError(
                f"on-device sampling needs one shared (top_k, top_p, temperature), got {sorted(requested)}"
            )
        top_k, top_p, temperature = requested.pop()
        # top-1 is argmax whatever p and T are; keep the dedicated greedy sampler.
        return _GREEDY_SAMPLING if top_k == 1 else (top_k, top_p, temperature)

    def _capture_sampling_trace(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        force_argmax: bool,
    ):
        output_prefix = tuple(int(tt_out_tok.shape[index]) for index in range(3))
        if output_prefix != (1, 1, 1) or int(logits.shape[-2]) != int(tt_out_tok.shape[-1]):
            raise ValueError("sampling logits [1,1,B,V] require token output [1,1,1,B]")
        key = (int(tt_out_tok.shape[-1]), int(top_k), float(top_p), float(temperature), bool(force_argmax))
        if self._sampling_trace_id is not None and (
            self._sampling_trace_key != key or self._sampling_trace_logits is not logits
        ):
            raise RuntimeError("model and sampling traces must be released together before recapture")
        if self._sampling_trace_id is not None:
            return self._sampling_trace_output

        use_greedy_tp4 = not force_argmax and self._is_semantic_greedy(
            top_k=top_k, top_p=top_p, temperature=temperature
        )
        if not force_argmax and not use_greedy_tp4 and self._sampling_params is None:
            raise RuntimeError("sampling parameters must be allocated before sampler trace capture")
        # prepare_token_out_decode prewarms this exact physical sampler
        # workload before registering the model trace.  Dispatching it eagerly
        # again here would allocate after that trace is live.  Capture the
        # sampler directly; begin_trace_capture establishes a trace-safe
        # allocation region for any address-specific runtime resources.
        ttnn.synchronize_device(self.mesh_device)
        self.model.trace_state.counters["synchronizations"] += 1
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        if force_argmax:
            captured = self.force_argmax_sampler.decode_forward(logits, tt_out_tok=tt_out_tok)
        elif use_greedy_tp4:
            captured = self.greedy_tp4_sampler.decode_forward(logits, tt_out_tok=tt_out_tok)
        else:
            captured = self.sampler.decode_forward(
                logits,
                k=self._sampling_params[0],
                p=self._sampling_params[1],
                temp=self._sampling_params[2],
                tt_out_tok=tt_out_tok,
            )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self._sampling_trace_id = trace_id
        self._sampling_trace_output = (tt_out_tok, captured[1])
        self._sampling_trace_logits = logits
        self._sampling_trace_key = key
        return self._sampling_trace_output

    def _prewarm_split_sampling_workloads(self, *, kv_cache, use_greedy_tp4: bool, force_argmax: bool) -> None:
        """Create batch-specific sampler programs/resources before either trace is live.

        The broadcast-backed candidate gathers own workload semaphores that are
        created on their first exact-shape invocation.  Creating a new batch
        variant after the model trace is registered makes those resources unsafe
        with respect to the trace allocator.  Warm one complete model-to-sampler
        step first, then restore the persistent position inputs before the normal
        model and sampler captures.
        """
        state = self.model.trace_state
        if not force_argmax and not use_greedy_tp4 and self._sampling_params is None:
            raise RuntimeError("sampling parameters must exist before split-sampling prewarm")
        if state.initial_positions is None:
            raise RuntimeError("initial trace positions were lost before split-sampling prewarm")
        if any(
            value is None
            for value in (state.token_input, state.rope_position, state.rope_position_lookup_mask, state.cache_position)
        ):
            raise RuntimeError("trace state must be initialized before split-sampling prewarm")

        warmup_logits = self.model.decode_forward_device_state(
            state.token_input,
            rope_position=state.rope_position,
            rope_position_lookup_mask=state.rope_position_lookup_mask,
            cache_position=state.cache_position,
            page_tables=state.page_tables,
            kv_cache=kv_cache,
            batch_size=state.batch_size,
            advance_position=True,
        )
        warmup_token = self._new_token_buffer(state.batch_size)
        if force_argmax:
            self.force_argmax_sampler.decode_forward(warmup_logits, tt_out_tok=warmup_token)
        elif use_greedy_tp4:
            self.greedy_tp4_sampler.decode_forward(warmup_logits, tt_out_tok=warmup_token)
        else:
            self.sampler.decode_forward(
                warmup_logits,
                k=self._sampling_params[0],
                p=self._sampling_params[1],
                temp=self._sampling_params[2],
                tt_out_tok=warmup_token,
            )
        ttnn.synchronize_device(self.mesh_device)
        state.counters["synchronizations"] += 1
        warmup_logits.deallocate(True)
        warmup_token.deallocate(True)
        self.model.write_trace_positions_from_host(state.initial_positions)

    def _execute_sampling_trace(self):
        if self._sampling_trace_id is None or self._sampling_trace_output is None:
            raise RuntimeError("sampling trace is not captured")
        ttnn.execute_trace(self.mesh_device, self._sampling_trace_id, cq_id=0, blocking=False)
        return self._sampling_trace_output

    def reset(self) -> None:
        # A new request may run prefill, which allocates transient buffers.
        # Release both decode traces before any such allocation so trace DRAM
        # cannot alias the new request's buffers.
        self._release_all_decode_traces()
        if self._cache_dirty and self._owns_standalone_cache:
            self.model.zero_kv_cache(self.kv_cache)
            ttnn.synchronize_device(self.mesh_device)
            self._cache_dirty = False
        self.timings.clear()
        for name in self.model.trace_state.counters:
            self.model.trace_state.counters[name] = 0

    def teardown(self) -> None:
        self._release_all_decode_traces()
        self.greedy_tp4_sampler.teardown()
        self.model.teardown()

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        return_device_logits: bool = False,
        release_decode_traces: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        # The vLLM adapter passes release_decode_traces=False for prefills
        # whose programs and shape caches are already warm: those allocate
        # transients only, which are freed before the next trace execution,
        # so the live decode traces stay valid. Every other caller releases.
        if release_decode_traces and (
            self.model.trace_state.trace_id is not None or self._sampling_trace_id is not None
        ):
            self._release_all_decode_traces()
        # The source-current readiness prefill runner passes one disposable TT
        # tensor with kv_cache=None as an explicit placeholder. It is not a
        # usable per-layer cache mapping, so treat that exact legacy shape as
        # an omitted pair. All real external state remains both-or-neither.
        if kv_cache is None and isinstance(page_table, ttnn.Tensor):
            page_table = None
        page_table, kv_cache, external_state = self._resolve_cache_pair(page_table, kv_cache)
        self._cache_dirty = getattr(self, "_cache_dirty", False) or not external_state
        if return_device_logits:
            return self.model.prefill_forward_device_logits(
                tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens
            )
        return self.model.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            return_all_logits=return_all_logits,
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache,
        enable_trace: bool = False,
        return_device_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        """Explicit eager one-step API; traced callers use the token-out methods below."""
        if enable_trace:
            raise ValueError("use prepare_token_out_decode/decode_next_token_traced for traced decode")
        page_table, kv_cache, external_state = self._resolve_cache_pair(page_table, kv_cache)
        self._cache_dirty = getattr(self, "_cache_dirty", False) or not external_state
        return self.model.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            return_device_logits=return_device_logits,
        )

    def prepare_token_out_decode(
        self,
        *,
        first_input_tokens: Sequence[int] | torch.Tensor | None,
        first_input_device_tokens: ttnn.Tensor | None = None,
        start_positions: Sequence[int] | torch.Tensor,
        page_table=None,
        kv_cache=None,
        page_table_generations: Sequence[int] | None = None,
        prompt_lengths: Sequence[int] | None = None,
        active_batch_size: int | None = None,
        pad_to_max_batch: bool = True,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        force_argmax: bool = False,
        seed: int | Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, Any]:
        positions = torch.as_tensor(start_positions, dtype=torch.int32).reshape(-1)
        if first_input_device_tokens is None:
            if first_input_tokens is None:
                raise ValueError("first_input_tokens are required without a device token buffer")
            tokens = torch.as_tensor(first_input_tokens, dtype=torch.int32).reshape(-1, 1)
        else:
            if first_input_tokens is not None:
                raise ValueError("provide either host or device first-input tokens, not both")
            if tuple(int(first_input_device_tokens.shape[index]) for index in range(3)) != (1, 1, 1):
                raise ValueError("device first-input tokens must have shape [1,1,1,batch]")
            tokens = torch.zeros((positions.numel(), 1), dtype=torch.int32)
        if tokens.shape[0] != positions.numel() or not 1 <= tokens.shape[0] <= self.max_batch_size:
            raise ValueError("token and position rows must match within max_batch_size")
        normalized_prompt_lengths, active_batch_size = self._normalize_decode_slots(
            positions, prompt_lengths=prompt_lengths, active_batch_size=active_batch_size
        )
        if pad_to_max_batch and tokens.shape[0] < self.max_batch_size:
            padding = self.max_batch_size - tokens.shape[0]
            tokens = torch.cat((tokens, torch.zeros((padding, 1), dtype=torch.int32)))
            positions = torch.cat((positions, torch.full((padding,), -1, dtype=torch.int32)))
            normalized_prompt_lengths += (0,) * padding
        if first_input_device_tokens is not None and int(first_input_device_tokens.shape[-1]) != int(tokens.shape[0]):
            raise ValueError("device first-input token buffer must match the padded physical decode batch")
        sampling_seeds = self._normalize_sampling_seeds(
            seed,
            active_slots=[value >= 0 for value in positions.tolist()],
            physical_batch_size=int(tokens.shape[0]),
        )
        page_tables_raw, resolved_kv_cache, external_state = self._resolve_cache_pair(page_table, kv_cache)
        page_tables = self.model._normalize_page_tables(page_tables_raw)
        if external_state and page_table_generations is None:
            raise ValueError("external page tables require explicit page_table_generations")
        if page_table_generations is None:
            page_table_generations = [self.page_table_generation] * len(page_tables)
        elif len(page_table_generations) != len(page_tables):
            raise ValueError("page-table generations must match the layer count")
        cache_identity = _kv_cache_identity(resolved_kv_cache)
        requested_key = (
            int(tokens.shape[0]),
            int(top_k),
            float(top_p),
            float(temperature),
            bool(force_argmax),
        )
        if self.model.trace_state.trace_id is not None and (
            self._sampling_trace_id is None
            or self._sampling_trace_key != requested_key
            or self.model.trace_state.kv_cache_identity != cache_identity
        ):
            self._release_all_decode_traces()

        state = self.model.initialize_trace_state(
            tokens=tokens,
            start_pos=positions,
            page_tables=page_tables,
            page_table_generations=page_table_generations,
            prompt_lengths=normalized_prompt_lengths,
            active_batch_size=active_batch_size,
            refresh_tokens_from_host=first_input_device_tokens is None,
        )
        if first_input_device_tokens is not None:
            self.model.write_trace_tokens_from_device(first_input_device_tokens)
        if state.trace_id is not None:
            self.model.refresh_trace_page_tables(page_tables, generations=page_table_generations)
            if not force_argmax and not self._is_semantic_greedy(top_k=top_k, top_p=top_p, temperature=temperature):
                self._initialize_non_greedy_rng(sampling_seeds)
            self._validate_next_trace_position()
            logits = self.model.execute_decode_trace()
            return self._execute_sampling_trace()

        # Sampling parameters outlive the sampler trace and therefore must be
        # allocated before the model trace makes allocator addresses unsafe.
        use_greedy_tp4 = not force_argmax and self._is_semantic_greedy(
            top_k=top_k, top_p=top_p, temperature=temperature
        )
        if not force_argmax and not use_greedy_tp4:
            self._sampling_params = self._make_sampling_params(
                batch_size=requested_key[0], top_k=top_k, top_p=top_p, temperature=temperature
            )
        self._prewarm_split_sampling_workloads(
            kv_cache=resolved_kv_cache,
            use_greedy_tp4=use_greedy_tp4,
            force_argmax=force_argmax,
        )
        if not force_argmax and not use_greedy_tp4:
            # Prewarm may consume RNG state. Reinitialize the requested stream
            # exactly once, then leave UINT32_MAX in the trace-bound buffer so
            # every replay advances rand_tile() without host seed traffic.
            self._initialize_non_greedy_rng(sampling_seeds)
        self.model.capture_decode_trace(kv_cache=resolved_kv_cache)
        if self.model.trace_state.logits is None:
            raise RuntimeError("model trace capture did not retain persistent logits")
        output = self._capture_sampling_trace(
            self.model.trace_state.logits,
            tt_out_tok=state.token_input,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            force_argmax=force_argmax,
        )
        self._validate_next_trace_position()
        self.model.execute_decode_trace()
        self._execute_sampling_trace()
        return output

    def decode_next_token_traced(
        self,
        *,
        page_table=None,
        kv_cache=None,
        page_table_generations: Sequence[int] | None = None,
    ) -> tuple[ttnn.Tensor, Any]:
        if self.model.trace_state.trace_id is None:
            raise RuntimeError("prepare_token_out_decode must capture the decode traces first")
        if page_table is not None or kv_cache is not None:
            page_table, kv_cache, _ = self._resolve_cache_pair(page_table, kv_cache)
            if _kv_cache_identity(kv_cache) != self.model.trace_state.kv_cache_identity:
                raise ValueError("decode trace is bound to a different KV cache allocation")
            if page_table_generations is None:
                raise ValueError("external page tables require explicit page_table_generations")
            self.model.refresh_trace_page_tables(
                self.model._normalize_page_tables(page_table), generations=page_table_generations
            )
        elif page_table_generations is not None:
            raise ValueError("page_table_generations require explicit page_table and kv_cache handles")
        self._validate_next_trace_position()
        self.model.execute_decode_trace()
        return self._execute_sampling_trace()

    def write_teacher_forced_token(self, token: int | Sequence[int]) -> None:
        self.model.write_trace_tokens_from_host(torch.as_tensor(token, dtype=torch.int32).reshape(-1, 1))

    def read_sampled_token(self, token_tensor: ttnn.Tensor) -> int:
        value = _read_first_mesh_token(token_tensor)
        self.model.trace_state.counters["sampled_token_readbacks"] += 1
        return value

    def benchmark_token_out_no_readback(self, prompt_token_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
        if max_new_tokens < 3:
            raise ValueError("benchmark requires at least three output tokens")
        self._validate_generation_window(len(prompt_token_ids), max_new_tokens)
        self.reset()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        start = time.perf_counter()
        prefill_logits = self.prefill_forward(
            prompt,
            page_table=self.page_tables,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_device_logits=True,
        )
        first_buffer = self._new_token_buffer(1)
        first_output, _ = self._sample_eager(prefill_logits, tt_out_tok=first_buffer)
        # Measure device-complete sampler-ready TTFT without reading the token.
        # This is one request-boundary fence, not a per-token decode fence.
        ttnn.synchronize_device(self.mesh_device)
        self.model.trace_state.counters["synchronizations"] += 1
        prefill_logits.deallocate(True)
        ttft = time.perf_counter()
        self.prepare_token_out_decode(
            first_input_tokens=None,
            first_input_device_tokens=first_output,
            start_positions=[len(prompt_token_ids)],
            pad_to_max_batch=False,
        )
        first_buffer.deallocate(True)
        steady_start = time.perf_counter()
        for _ in range(max_new_tokens - 2):
            self.decode_next_token_traced()
        ttnn.synchronize_device(self.mesh_device)
        end = time.perf_counter()
        return {
            "workload": {"prompt_len": len(prompt_token_ids), "gen_len": max_new_tokens, "batch": 1},
            "ttft_ms": (ttft - start) * 1000.0,
            "decode_t/s/u": (max_new_tokens - 1) / max(end - ttft, 1e-9),
            "steady_decode_t/s/u": (max_new_tokens - 2) / max(end - steady_start, 1e-9),
            "trace_counters": dict(self.model.trace_state.counters),
        }

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        host_sampling_compat: bool | None = None,
        stop_on_eos: bool = False,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return []
        self._validate_generation_window(len(prompt_token_ids), max_new_tokens)
        use_host_sampling = self.host_sampling_compat if host_sampling_compat is None else host_sampling_compat
        if not enable_trace and not use_host_sampling:
            raise ValueError("optimized token-out generation requires enable_trace=True")

        self.reset()
        request_seed = secrets.randbelow(_SAMPLING_SEED_SKIP) if seed is None else int(seed)
        if not 0 <= request_seed < _SAMPLING_SEED_SKIP:
            raise ValueError("sampling seeds must be uint32 values other than UINT32_MAX")
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        start = time.perf_counter()
        if use_host_sampling:
            logits = self.prefill_forward(
                prompt,
                page_table=self.page_tables,
                kv_cache=self.kv_cache,
                prompt_lens=[len(prompt_token_ids)],
            )
            predicted = int(torch.argmax(logits[0, -1]).item())
        else:
            logits = self.prefill_forward(
                prompt,
                page_table=self.page_tables,
                kv_cache=self.kv_cache,
                prompt_lens=[len(prompt_token_ids)],
                return_device_logits=True,
            )
            first_buffer = self._new_token_buffer(1)
            sampled, _ = self._sample_eager(
                logits,
                tt_out_tok=first_buffer,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=request_seed,
            )
            predicted = self.read_sampled_token(sampled)
            logits.deallocate(True)
            first_buffer.deallocate(True)
        first = time.perf_counter()
        self.timings["ttft_ms"] = (first - start) * 1000.0
        outputs = [predicted]
        next_token = next_input(0, predicted) if next_input is not None else predicted

        if max_new_tokens > 1:
            if use_host_sampling or not enable_trace:
                for step in range(1, max_new_tokens):
                    step_logits = self.decode_forward(
                        torch.tensor([[next_token]]),
                        torch.tensor([len(prompt_token_ids) + step - 1], dtype=torch.int32),
                        page_table=self.page_tables,
                        kv_cache=self.kv_cache,
                        enable_trace=False,
                    )
                    predicted = int(torch.argmax(step_logits[0]).item())
                    outputs.append(predicted)
                    next_token = next_input(step, predicted) if next_input is not None else predicted
            else:
                sampled, _ = self.prepare_token_out_decode(
                    first_input_tokens=[next_token],
                    start_positions=[len(prompt_token_ids)],
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=(request_seed + 1) % _SAMPLING_SEED_SKIP,
                )
                predicted = self.read_sampled_token(sampled)
                outputs.append(predicted)
                if next_input is not None:
                    forced = next_input(1, predicted)
                    if max_new_tokens > 2:
                        self.write_teacher_forced_token(forced)
                for step in range(2, max_new_tokens):
                    if stop_on_eos and next_input is None and outputs[-1] in self._eos_token_ids:
                        break
                    sampled, _ = self.decode_next_token_traced()
                    predicted = self.read_sampled_token(sampled)
                    outputs.append(predicted)
                    if next_input is not None:
                        forced = next_input(step, predicted)
                        if step < max_new_tokens - 1:
                            self.write_teacher_forced_token(forced)

        end = time.perf_counter()
        decode_tokens = max(len(outputs) - 1, 0)
        if decode_tokens:
            self.timings["decode_t/s/u"] = decode_tokens / max(end - first, 1e-9)
        self.timings["e2e_t/s/u"] = len(outputs) / max(end - start, 1e-9)
        return outputs


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Gemma4Generator:
    model_config = kwargs.pop("model_config", None)
    explicit_precision_config_path = kwargs.pop("precision_config_path", None)
    precision_config_path = explicit_precision_config_path
    if model_config is None and precision_config_path is None:
        precision_config_path = os.environ.get("GEMMA4_31B_PRECISION_CONFIG")
    if model_config is None and kwargs.get("model") is not None:
        model_config = kwargs["model"].config
    if model_config is None:
        config_fields = {}
        for key in ("max_seq_len", "layer_indices", "max_batch_size"):
            if key in kwargs:
                value = kwargs.pop(key)
                config_fields[key] = tuple(value) if key == "layer_indices" and value is not None else value
        if precision_config_path is None:
            selected = Path(model_dir) / "doc/datatype_sweep/selected_precision_config.json"
            if selected.exists():
                precision_config_path = selected
        model_config = (
            Gemma4FullModelConfig.from_precision_config(precision_config_path, **config_fields)
            if precision_config_path is not None
            else Gemma4FullModelConfig(**config_fields)
        )
    elif explicit_precision_config_path is not None:
        raise ValueError("precision_config_path cannot be combined with an explicit model_config")
    return Gemma4Generator(model_dir=model_dir, mesh_device=mesh_device, model_config=model_config, **kwargs)


__all__ = ["Gemma4Generator", "build_generator"]
