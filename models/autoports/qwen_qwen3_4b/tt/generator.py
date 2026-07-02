# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import HF_MODEL_ID
from models.autoports.qwen_qwen3_4b.tt.model import Qwen3FullModel, Qwen3FullModelConfig
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.readiness_check.contract import Generator, NextInputFn

_ARGMAX_TILE_KERNEL = "models/autoports/qwen_qwen3_4b/tt/kernels/qwen_argmax_tile_local_winner_kernel.cpp"
_ARGMAX_PAIR_REDUCE_KERNEL = "models/autoports/qwen_qwen3_4b/tt/kernels/qwen_argmax_pair_reduce_kernel.cpp"


def _core_grid_for_first_n(device, num_cores: int) -> tuple[ttnn.CoreRangeSet, list[ttnn.CoreCoord]]:
    grid_size = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)][:num_cores]
    return ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in cores}), cores


class Qwen3GreedyTP4Sampler:
    """Greedy-only TP4 sampler that avoids generic TopKDeviceOperation."""

    def __init__(self, *, mesh_device, vocab_per_device: int, max_batch_size: int, num_cores: int = 8) -> None:
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        if tuple(mesh_device.shape) != (1, 4) or self.num_devices != 4:
            raise ValueError(f"Qwen3GreedyTP4Sampler requires 1x4 TP4 mesh, got {tuple(mesh_device.shape)}")
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
        output_pair_page_bytes = 64
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
        tile_scratch = ttnn.CBDescriptor(
            total_size=tile_scratch_bytes,
            core_ranges=self.core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=tile_cb, data_format=ttnn.bfloat16, page_size=tile_scratch_bytes)
            ],
        )
        winner_cb = ttnn.CBDescriptor(
            total_size=winner_page_bytes * len(self.cores) * active_batch_size,
            core_ranges=self.core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=gather_cb, data_format=ttnn.uint32, page_size=winner_page_bytes)
            ],
        )
        return ttnn.ProgramDescriptor(kernels=kernels, cbs=[tile_scratch, winner_cb], semaphores=[])

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
                *gathered_accessor_args.get_compile_time_args(),
                *output_accessor_args.get_compile_time_args(),
            ],
            common_runtime_args=[
                int(gathered_pairs.buffer_address()),
                int(output_token.buffer_address()),
            ],
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
        active_batch_size = int(tt_out_tok.shape[-2])
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
            topology=ttnn.Topology.Ring,
            memory_config=self.gather_mem,
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


class Qwen3Generator(Generator):
    """Readiness and low-level serving generator for Qwen3-4B TP4 full model."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        mesh_device,
        hf_model_id: str = HF_MODEL_ID,
        model: Qwen3FullModel | None = None,
        model_config: Qwen3FullModelConfig | None = None,
        tokenizer=None,
        host_sampling_compat: bool = False,
        max_batch_size: int = 1,
        **model_kwargs,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.hf_model_id = hf_model_id
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        self.model = model or Qwen3FullModel.from_pretrained(
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            config=model_config,
            **model_kwargs,
        )
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self.max_batch_size = int(max_batch_size)
        self.host_sampling_compat = host_sampling_compat
        self.kv_cache = self.model.init_paged_kv_cache()
        self.page_table = self.model.make_identity_page_table(batch_size=self.max_batch_size)
        self.page_table_generation = 0
        self.greedy_sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.model.padded_vocab_size,
                mesh_device=mesh_device,
                max_batch_size=self.max_batch_size,
                max_top_k=32,
                allow_force_argmax=False,
                pad_to_power_of_2=True,
                ag_topology=ttnn.Topology.Linear,
            )
        )
        self.topk_sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.model.padded_vocab_size,
                mesh_device=mesh_device,
                max_batch_size=self.max_batch_size,
                max_top_k=32,
                allow_force_argmax=False,
                pad_to_power_of_2=True,
                ag_topology=ttnn.Topology.Linear,
            )
        )
        self.sampler = self.greedy_sampler
        self.force_argmax_sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.model.padded_vocab_size,
                mesh_device=mesh_device,
                max_batch_size=self.max_batch_size,
                max_top_k=32,
                allow_force_argmax=True,
                pad_to_power_of_2=True,
                ag_topology=ttnn.Topology.Linear,
            )
        )
        self.greedy_sampler.load_device_buffers()
        self.topk_sampler.load_device_buffers()
        self.force_argmax_sampler.load_device_buffers()
        self.greedy_tp4_sampler = Qwen3GreedyTP4Sampler(
            mesh_device=mesh_device,
            vocab_per_device=self.model.vocab_per_device,
            max_batch_size=self.max_batch_size,
        )
        self._sampling_trace_id: int | None = None
        self._sampling_trace_output = None
        self._sampling_params = self._make_sampling_params(batch_size=1, top_k=1, top_p=0.0, temperature=1.0)
        self._sampling_trace_key: tuple[int, float, float, bool] | None = None
        self.timings: dict[str, float] = {}

    def _release_traces(self) -> None:
        if self._sampling_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._sampling_trace_id)
            self._sampling_trace_id = None
            self._sampling_trace_output = None
        if self.model.trace_state.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.model.trace_state.trace_id)
            self.model.trace_state.trace_id = None
            self.model.trace_state.logits = None

    def reset(self) -> None:
        self._release_traces()
        self.kv_cache = self.model.init_paged_kv_cache()
        self.page_table = self.model.make_identity_page_table(batch_size=self.max_batch_size)
        self.page_table_generation += 1
        self.timings.clear()

    def teardown(self) -> None:
        self._release_traces()

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        kv_cache = self.kv_cache if kv_cache is None else kv_cache
        page_table = self.page_table if page_table is None else page_table
        return self.model.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            return_all_logits=return_all_logits,
            **kwargs,
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache,
        enable_trace: bool = True,
        return_device_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        # The eager low-level path remains available for readiness host-sampling
        # compatibility and focused debugging. Optimized token-out generation goes
        # through decode_next_token_on_device once sampler tracing is captured.
        kv_cache = self.kv_cache if kv_cache is None else kv_cache
        page_table = self.page_table if page_table is None else page_table
        if enable_trace and not self.host_sampling_compat:
            return self.model.decode_forward(
                tokens,
                start_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                return_device_logits=return_device_logits,
            )
        return self.model.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            return_device_logits=return_device_logits,
        )

    def _make_sampling_params(self, *, batch_size: int, top_k: int, top_p: float, temperature: float):
        k = ttnn.from_torch(
            torch.full((batch_size,), int(top_k), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        p = ttnn.from_torch(
            torch.full((batch_size,), float(top_p), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        temp = ttnn.from_torch(
            torch.full((batch_size,), float(temperature), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return k, p, temp

    @staticmethod
    def _active_batch_size_from_token_tensor(tt_out_tok: ttnn.Tensor) -> int:
        return int(tt_out_tok.shape[-2])

    def _new_replicated_token_buffer(self, batch_size: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.zeros((1, 1, int(batch_size), 1), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _capture_sampling_trace(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        skip_precompile: bool = False,
        force_argmax: bool = True,
    ):
        if not skip_precompile:
            if force_argmax:
                self._sample_force_argmax_to_output(logits, tt_out_tok=tt_out_tok)
            else:
                self._sample_greedy_tp4_to_output(logits, tt_out_tok=tt_out_tok)
            ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        if force_argmax:
            output = self._sample_force_argmax_to_output(logits, tt_out_tok=tt_out_tok)
        else:
            output = self._sample_greedy_tp4_to_output(logits, tt_out_tok=tt_out_tok)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(self.mesh_device)
        self._sampling_trace_id = trace_id
        self._sampling_trace_output = output
        self._sampling_trace_key = (
            self._active_batch_size_from_token_tensor(tt_out_tok),
            0.0,
            1.0,
            bool(force_argmax),
        )
        return output

    def _sample_force_argmax_to_output(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor):
        sampled, log_probs = self.force_argmax_sampler.decode_forward(logits, tt_out_tok=None)
        sampled_for_copy = sampled
        if list(sampled.shape) != list(tt_out_tok.shape):
            sampled_for_copy = ttnn.reshape(sampled, list(tt_out_tok.shape))
        ttnn.copy(sampled_for_copy, tt_out_tok)
        return tt_out_tok, log_probs

    def _sample_greedy_tp4_to_output(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor):
        return self.greedy_tp4_sampler.decode_forward(logits, tt_out_tok=tt_out_tok)

    def sample_logits_on_device(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        enable_trace: bool = True,
        force_argmax: bool = False,
        skip_trace_precompile: bool = False,
    ):
        if force_argmax:
            if not enable_trace:
                return self._sample_force_argmax_to_output(logits, tt_out_tok=tt_out_tok)
            trace_key = (self._active_batch_size_from_token_tensor(tt_out_tok), 0.0, 1.0, True)
            if self._sampling_trace_id is not None and self._sampling_trace_key != trace_key:
                ttnn.release_trace(self.mesh_device, self._sampling_trace_id)
                self._sampling_trace_id = None
                self._sampling_trace_output = None
                self._sampling_trace_key = None
            if self._sampling_trace_id is None:
                return self._capture_sampling_trace(
                    logits,
                    tt_out_tok=tt_out_tok,
                    skip_precompile=skip_trace_precompile,
                    force_argmax=True,
                )
            ttnn.execute_trace(self.mesh_device, self._sampling_trace_id, cq_id=0, blocking=False)
            return self._sampling_trace_output
        if not enable_trace:
            return self._sample_greedy_tp4_to_output(logits, tt_out_tok=tt_out_tok)
        trace_key = (self._active_batch_size_from_token_tensor(tt_out_tok), 0.0, 1.0, False)
        if self._sampling_trace_id is not None and self._sampling_trace_key != trace_key:
            ttnn.release_trace(self.mesh_device, self._sampling_trace_id)
            self._sampling_trace_id = None
            self._sampling_trace_output = None
            self._sampling_trace_key = None
        if self._sampling_trace_id is None:
            return self._capture_sampling_trace(
                logits,
                tt_out_tok=tt_out_tok,
                skip_precompile=skip_trace_precompile,
                force_argmax=False,
            )
        ttnn.execute_trace(self.mesh_device, self._sampling_trace_id, cq_id=0, blocking=False)
        return self._sampling_trace_output

    def sample_logits_topk_on_device(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        top_k: int,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        if top_k < 1 or top_k > 32:
            raise ValueError(f"top_k must be in [1, 32], got {top_k}")
        k, p, temp = self._make_sampling_params(
            batch_size=self._active_batch_size_from_token_tensor(tt_out_tok),
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        return self.topk_sampler.decode_forward(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)

    @staticmethod
    def _normalize_token_out_inputs(
        *,
        first_input_token: int | None,
        first_input_tokens: list[int] | torch.Tensor | None,
        start_pos: int | None,
        start_positions: list[int] | torch.Tensor | None,
        prompt_len: int | None,
        prompt_lens: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if first_input_tokens is None:
            if first_input_token is None:
                raise ValueError("first_input_token or first_input_tokens is required")
            first_input_tokens = [int(first_input_token)]
        if start_positions is None:
            if start_pos is None:
                raise ValueError("start_pos or start_positions is required")
            start_positions = [int(start_pos)]
        if prompt_lens is None:
            if prompt_len is None:
                raise ValueError("prompt_len or prompt_lens is required")
            prompt_lens = [int(prompt_len)]

        token_tensor = torch.as_tensor(first_input_tokens, dtype=torch.long).reshape(-1, 1)
        pos_tensor = torch.as_tensor(start_positions, dtype=torch.int32).reshape(-1)
        prompt_lens = [int(value) for value in prompt_lens]
        batch_size = int(token_tensor.shape[0])
        if int(pos_tensor.numel()) != batch_size:
            raise ValueError("start_positions must have one entry per token-out row")
        if len(prompt_lens) != batch_size:
            raise ValueError("prompt_lens must have one entry per token-out row")
        return token_tensor, pos_tensor, prompt_lens

    def prepare_token_out_decode(
        self,
        *,
        first_input_token: int | None = None,
        start_pos: int | None = None,
        prompt_len: int | None = None,
        first_input_tokens: list[int] | torch.Tensor | None = None,
        start_positions: list[int] | torch.Tensor | None = None,
        prompt_lens: list[int] | None = None,
        page_table=None,
        read_first_token: bool = True,
    ) -> int | list[int] | None:
        token_input, pos, prompt_lens = self._normalize_token_out_inputs(
            first_input_token=first_input_token,
            first_input_tokens=first_input_tokens,
            start_pos=start_pos,
            start_positions=start_positions,
            prompt_len=prompt_len,
            prompt_lens=prompt_lens,
        )
        if int(token_input.shape[0]) > self.max_batch_size:
            raise ValueError(
                f"token-out batch size {int(token_input.shape[0])} exceeds max_batch_size={self.max_batch_size}"
            )
        page_table = self.page_table if page_table is None else page_table
        self.model.reset_decode_trace_state(
            token_input=token_input,
            start_pos=pos,
            page_table=page_table,
            prompt_lens=prompt_lens,
        )
        state = self.model.trace_state
        if state.token_input is None or state.current_pos is None or state.rope_pos is None or state.page_table is None:
            raise RuntimeError("decode trace state did not initialize")
        warm_logits = self.model.decode_forward_device_state(
            state.token_input,
            current_pos=state.current_pos,
            rope_pos=state.rope_pos,
            page_table=state.page_table,
            kv_cache=self.kv_cache,
            advance_position=True,
        )
        prewarm_output = self._new_replicated_token_buffer(int(token_input.shape[0]))
        self.sample_logits_on_device(
            warm_logits,
            tt_out_tok=prewarm_output,
            enable_trace=False,
            force_argmax=False,
        )
        self.model._reset_trace_positions_from_host()
        self.model.capture_decode_trace(kv_cache=self.kv_cache)
        if self.model.trace_state.logits is None or self.model.trace_state.token_input is None:
            raise RuntimeError("decode trace did not produce logits/token input")
        sampled, _ = self.sample_logits_on_device(
            self.model.trace_state.logits,
            tt_out_tok=self.model.trace_state.token_input,
            enable_trace=True,
            skip_trace_precompile=True,
            force_argmax=False,
        )
        if not read_first_token:
            return None
        tokens = [int(value) for value in ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1).tolist()]
        self.model.trace_state.counters["readbacks"] += 1
        return tokens[0] if len(tokens) == 1 else tokens

    def refresh_decode_page_table(self, page_table, *, generation: int | None = None) -> None:
        self.model.refresh_trace_page_table(page_table, generation=generation)

    def decode_next_token_traced(self, *, page_table=None, page_table_generation: int | None = None):
        if page_table is not None:
            self.refresh_decode_page_table(page_table, generation=page_table_generation)
        return self.decode_next_token_on_device()

    def decode_next_token_on_device(self):
        logits = self.model.execute_decode_trace()
        return self.sample_logits_on_device(
            logits,
            tt_out_tok=self.model.trace_state.token_input,
            enable_trace=True,
            force_argmax=False,
        )

    def benchmark_token_out_no_readback(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        enable_trace: bool = True,
    ) -> dict[str, float | int | dict[str, int]]:
        if max_new_tokens < 2:
            raise ValueError("benchmark_token_out_no_readback requires at least two generated tokens")
        self.reset()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        start_s = time.perf_counter()
        logits = self.prefill_forward(
            prompt,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=False,
        )
        first_input_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        first_s = time.perf_counter()
        if not enable_trace:
            raise ValueError("optimized token-out benchmark requires enable_trace=True")
        self.prepare_token_out_decode(
            first_input_token=first_input_token,
            start_pos=len(prompt_token_ids),
            prompt_len=len(prompt_token_ids),
            read_first_token=False,
        )
        steady_start_s = time.perf_counter()
        for _ in range(2, max_new_tokens):
            self.decode_next_token_on_device()
        ttnn.synchronize_device(self.mesh_device)
        end_s = time.perf_counter()
        decode_tokens = max_new_tokens - 1
        steady_tokens = max_new_tokens - 2
        return {
            "ttft_ms": (first_s - start_s) * 1000.0,
            "decode_t/s/u": decode_tokens / max(end_s - first_s, 1.0e-9),
            "prepare_decode_ms": (steady_start_s - first_s) * 1000.0,
            "steady_decode_t/s/u": steady_tokens / max(end_s - steady_start_s, 1.0e-9) if steady_tokens else 0.0,
            "steady_decode_tokens": steady_tokens,
            "e2e_t/s/u": max_new_tokens / max(end_s - start_s, 1.0e-9),
            "decode_tokens": decode_tokens,
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
        **kwargs: Any,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return []
        use_host_sampling = self.host_sampling_compat if host_sampling_compat is None else host_sampling_compat

        self.reset()
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        start_s = time.perf_counter()
        logits = self.prefill_forward(
            prompt,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=False,
        )
        predicted = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        first_s = time.perf_counter()
        self.timings["ttft_ms"] = (first_s - start_s) * 1000.0

        outputs = [predicted]
        eos_id = self.tokenizer.eos_token_id
        next_token = next_input(0, predicted) if next_input is not None else predicted
        position = len(prompt_token_ids)

        if enable_trace and not use_host_sampling and next_input is None and max_new_tokens > 1:
            captured_token = self.prepare_token_out_decode(
                first_input_token=next_token,
                start_pos=position,
                prompt_len=len(prompt_token_ids),
            )
            outputs.append(captured_token)
            for step in range(2, max_new_tokens):
                sampled, _ = self.decode_next_token_on_device()
                token = int(ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[0].item())
                self.model.trace_state.counters["readbacks"] += 1
                outputs.append(token)
            end_s = time.perf_counter()
            decode_tokens = max(len(outputs) - 1, 0)
            if decode_tokens:
                self.timings["decode_t/s/u"] = decode_tokens / max(end_s - first_s, 1.0e-9)
            self.timings["e2e_t/s/u"] = len(outputs) / max(end_s - start_s, 1.0e-9)
            return outputs

        if enable_trace and not use_host_sampling and next_input is not None and max_new_tokens > 1:
            captured_token = self.prepare_token_out_decode(
                first_input_token=next_token,
                start_pos=position,
                prompt_len=len(prompt_token_ids),
            )
            outputs.append(captured_token)
            next_token = next_input(1, captured_token)
            if max_new_tokens > 2:
                self.model.write_trace_token_from_host(next_token)
            for step in range(2, max_new_tokens):
                sampled, _ = self.decode_next_token_on_device()
                predicted = int(ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[0].item())
                self.model.trace_state.counters["readbacks"] += 1
                outputs.append(predicted)
                next_token = next_input(step, predicted)
                if step < max_new_tokens - 1:
                    self.model.write_trace_token_from_host(next_token)
            end_s = time.perf_counter()
            decode_tokens = max(len(outputs) - 1, 0)
            if decode_tokens:
                self.timings["decode_t/s/u"] = decode_tokens / max(end_s - first_s, 1.0e-9)
            self.timings["e2e_t/s/u"] = len(outputs) / max(end_s - start_s, 1.0e-9)
            return outputs

        for step in range(1, max_new_tokens):
            if stop_on_eos and eos_id is not None and outputs[-1] == eos_id and next_input is None:
                break
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            pos_tensor = torch.tensor([position], dtype=torch.int32)
            logits = self.decode_forward(
                token_tensor,
                pos_tensor,
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                enable_trace=enable_trace,
                return_device_logits=False,
            )
            if use_host_sampling:
                predicted = int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
            else:
                # Until split sampling is captured, this path still normalizes the
                # output to the same greedy semantics expected by readiness.
                predicted = int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
            outputs.append(predicted)
            next_token = next_input(step, predicted) if next_input is not None else predicted
            position += 1

        end_s = time.perf_counter()
        decode_tokens = max(len(outputs) - 1, 0)
        if decode_tokens:
            self.timings["decode_t/s/u"] = decode_tokens / max(end_s - first_s, 1.0e-9)
        self.timings["e2e_t/s/u"] = len(outputs) / max(end_s - start_s, 1.0e-9)
        return outputs


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Qwen3Generator:
    return Qwen3Generator(model_dir=model_dir, mesh_device=mesh_device, **kwargs)
