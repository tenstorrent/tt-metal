# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the TP=4 Gemma-4 26B A4B full model."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

import torch
from transformers import AutoConfig, AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import DECODE_SLOT_COUNT, FullModelState, Gemma4FullModel
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.readiness_check.contract import Generator, NextInputFn
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


@dataclass
class TraceCounters:
    replays: int = 0
    token_refreshes: int = 0
    position_refreshes: int = 0
    rope_refreshes: int = 0
    page_table_refreshes: int = 0
    synchronizations: int = 0
    token_readbacks: int = 0


@dataclass(frozen=True)
class SamplingSpec:
    """Semantic sampling configuration used to key captured sampler graphs."""

    greedy: bool
    top_k: tuple[int, ...] = ()
    top_p: tuple[float, ...] = ()
    temperature: tuple[float, ...] = ()
    seeds: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.greedy, self.top_k, self.top_p, self.temperature, self.seeds)


@dataclass
class DecodeTrace:
    model_trace_id: int
    sampling_trace_id: int | None
    token_input: ttnn.Tensor
    current_pos: ttnn.Tensor
    position_ids: ttnn.Tensor
    logits: ttnn.Tensor
    sampled_tokens: ttnn.Tensor
    state: FullModelState
    batch_size: int
    sampling_mode: str
    page_table_ids: tuple[int, ...]
    sampling_spec: SamplingSpec
    sampling_params: tuple[ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None]


def _padded_prefill_len(logical_len: int) -> int:
    if logical_len < 1:
        raise ValueError("prompt must contain at least one token")
    return max(ttnn.TILE_SIZE, ((logical_len + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE)


class Gemma4Generator(Generator):
    """Two-level generator with explicit state and split traced sampling.

    The optimized mode is ``sampling_mode='device'``.  ``'host'`` is an
    explicit compatibility mode for readiness checks that require logits.
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_mixed_prompt_lens": True,
        "supports_inactive_rows": True,
        "supports_on_device_sampling": True,
    }

    def __init__(self, model: Gemma4FullModel, tokenizer: Any, *, sampling_mode: str = "device") -> None:
        if sampling_mode not in ("device", "host"):
            raise ValueError("sampling_mode must be 'device' or 'host'")
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.sampling_mode = sampling_mode
        # TTTv2 was selected over the stateful TTTv1 SamplingGenerator: its
        # per-call parameters, native 1D topology, power-of-two local-vocab
        # padding, and direct tt_out_tok contract match this model exactly.
        self.sampler = Sampling1D(
            vocab_size=model.vocab_size,
            mesh_device=self.mesh_device,
            # Sampling always consumes the fixed 32-row terminal tile even
            # when model compute is sliced to a smaller logical batch.
            max_batch_size=DECODE_SLOT_COUNT,
            max_top_k=32,
            allow_force_argmax=True,
            pad_to_power_of_2=True,
            num_gather_links=2,
        )
        self._trace_cache: dict[tuple[int, str, tuple[Any, ...], int, tuple[int, ...]], DecodeTrace] = {}
        self.trace_counters = TraceCounters()
        self.last_perf: dict[str, float] = {}

    @staticmethod
    def _state_from_args(
        model: Gemma4FullModel,
        *,
        kv_cache: Any,
        page_table: Any,
        batch_size: int,
        prompt_lens: Sequence[int] | None = None,
    ) -> FullModelState:
        owns_cache = kv_cache is None
        if isinstance(kv_cache, FullModelState):
            state = kv_cache
        elif kv_cache is None:
            state = model.state or model.allocate_state(max_batch_size=batch_size)
        else:
            state = model.allocate_state(max_batch_size=batch_size)
            state.kv_cache = list(kv_cache)
        if page_table is not None:
            if isinstance(page_table, (list, tuple)):
                if len(page_table) != model.num_layers:
                    raise ValueError(f"expected {model.num_layers} per-layer page tables")
                state.page_tables = list(page_table)
            elif model.num_layers > 0:
                # One table is legal only when all layer geometries match.
                geometries = {(s.block_size, s.blocks_per_slot) for s in state.cache_specs}
                if len(geometries) != 1:
                    if not owns_cache:
                        raise ValueError("Gemma4 mixed cache geometry requires one page table per layer")
                    # The common readiness runner passes a geometry-agnostic
                    # placeholder table with kv_cache=None. In that case this
                    # generator owns cache allocation and must retain its
                    # matching per-layer page tables.
                    page_table = None
                if page_table is not None:
                    state.page_tables = [page_table] * model.num_layers
        if prompt_lens is not None:
            state.prompt_lens[: len(prompt_lens)] = [int(x) for x in prompt_lens]
        return state

    def _host_tokens_to_device(self, tokens: torch.Tensor, *, rank4: bool = False) -> ttnn.Tensor:
        values = tokens.to(torch.int32)
        if rank4:
            flat = values.reshape(-1)
            if flat.numel() > DECODE_SLOT_COUNT:
                raise ValueError(f"decode batch exceeds fixed {DECODE_SLOT_COUNT}-slot contract")
            values = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.int32)
            values[: flat.numel()] = flat
            values = values.reshape(1, 1, 1, DECODE_SLOT_COUNT)
        return ttnn.from_torch(
            values,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _positions_to_device(
        self, positions: torch.Tensor, *, dtype: Any = ttnn.int32, layout: Any = ttnn.ROW_MAJOR_LAYOUT
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            positions.to(torch.int32),
            dtype=dtype,
            layout=layout,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _refresh_device_input(self, target: ttnn.Tensor, values: torch.Tensor, *, rank4: bool = False) -> None:
        values = values.to(torch.int32)
        if rank4:
            flat = values.reshape(-1)
            values = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.int32)
            values[: flat.numel()] = flat
            values = values.reshape(1, 1, 1, DECODE_SLOT_COUNT)
        host = ttnn.from_torch(
            values,
            dtype=target.dtype,
            layout=target.layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, target)

    @staticmethod
    def _expand_sampling_value(value: Any, batch_size: int, cast: Any, name: str) -> tuple[Any, ...]:
        values = list(value) if isinstance(value, (list, tuple, torch.Tensor)) else [value] * batch_size
        if len(values) != batch_size:
            raise ValueError(f"{name} must be scalar or have {batch_size} entries")
        return tuple(cast(item) for item in values)

    def _sampling_spec(
        self,
        batch_size: int,
        *,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
    ) -> SamplingSpec:
        temperatures = self._expand_sampling_value(temperature, batch_size, float, "temperature")
        if all(value <= 0.0 for value in temperatures):
            return SamplingSpec(greedy=True)
        if any(value <= 0.0 for value in temperatures):
            raise ValueError("temperature must be positive for every sampled row")
        ks = self._expand_sampling_value(top_k, batch_size, int, "top_k")
        ps = self._expand_sampling_value(top_p, batch_size, float, "top_p")
        seed_values = self._expand_sampling_value(seeds, batch_size, int, "seeds")
        if any(value < 1 or value > 32 for value in ks):
            raise ValueError("top_k must be in [1, 32]")
        if any(value < 0.0 or value > 1.0 for value in ps):
            raise ValueError("top_p must be in [0, 1]")
        return SamplingSpec(False, ks, ps, temperatures, seed_values)

    def _sampling_params(
        self, spec: SamplingSpec
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None]:
        if spec.greedy:
            # Native force-argmax is exactly greedy and supports direct output
            # into the device feedback tensor. On faithful TP4 [1,1,32,262144]
            # logits it measured 2.343 ms versus 10.736 ms for the semantically
            # equivalent k=1/p=0/temp=1 path (sampler_comparison.json).
            return None, None, None, None
        padded_k = torch.ones(DECODE_SLOT_COUNT, dtype=torch.int32)
        padded_p = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.bfloat16)
        padded_temp = torch.ones(DECODE_SLOT_COUNT, dtype=torch.bfloat16)
        padded_seeds = torch.zeros(DECODE_SLOT_COUNT, dtype=torch.int32)
        active = len(spec.top_k)
        padded_k[:active] = torch.tensor(spec.top_k, dtype=torch.int32)
        padded_p[:active] = torch.tensor(spec.top_p, dtype=torch.bfloat16)
        padded_temp[:active] = torch.tensor(spec.temperature, dtype=torch.bfloat16)
        padded_seeds[:active] = torch.tensor(spec.seeds, dtype=torch.int32)

        def make(values: torch.Tensor, dtype: Any) -> ttnn.Tensor:
            return ttnn.from_torch(
                values,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, None), mesh_shape=tuple(self.mesh_device.shape)
                ),
            )

        return (
            make(padded_k, ttnn.uint32),
            make(padded_p, ttnn.bfloat16),
            make(padded_temp, ttnn.bfloat16),
            make(padded_seeds, ttnn.uint32),
        )

    def _read_tokens(self, tokens: ttnn.Tensor, batch_size: int) -> torch.Tensor:
        shard = ttnn.get_device_tensors(tokens)[0] if isinstance(tokens.device(), ttnn.MeshDevice) else tokens
        self.trace_counters.token_readbacks += 1
        return ttnn.to_torch(shard).reshape(-1)[:batch_size].to(torch.long)

    @staticmethod
    def _pad_sampling_logits(logits: ttnn.Tensor) -> ttnn.Tensor:
        if logits.shape[-2] == DECODE_SLOT_COUNT:
            return logits
        return ttnn.pad(
            logits,
            padding=[(0, 0), (0, 0), (0, DECODE_SLOT_COUNT - logits.shape[-2]), (0, 0)],
            value=0.0,
        )

    def _gather_logits_to_torch(self, logits: ttnn.Tensor, *, logical_len: int | None = None) -> torch.Tensor:
        """Explicit host-logits compatibility boundary for readiness checks."""
        gathered = ttnn.all_gather(logits, dim=3, cluster_axis=1, num_links=2, topology=ttnn.Topology.Ring)
        shard = ttnn.get_device_tensors(gathered)[0]
        host = ttnn.to_torch(shard)
        if host.ndim == 4 and host.shape[1] == 1:
            host = host[:, 0]
        if logical_len is not None:
            host = host[:, :logical_len]
        return host

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: Any,
        kv_cache: Any,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        del kwargs
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens):
            raise ValueError("tokens must be [batch,padded_prompt] with one prompt_lens entry per row")
        if any(n < 1 or n > self.model.max_seq_len for n in prompt_lens):
            raise ValueError(f"prompt lengths must be in [1, {self.model.max_seq_len}]")
        state = self._state_from_args(
            self.model,
            kv_cache=kv_cache,
            page_table=page_table,
            batch_size=tokens.shape[0],
            prompt_lens=prompt_lens,
        )
        state.active_mask.zero_()
        state.active_mask[: len(prompt_lens)] = True
        state.positions.fill_(-1)
        state.positions[: len(prompt_lens)] = torch.tensor(prompt_lens, dtype=torch.int32)
        for row, logical_len in enumerate(prompt_lens):
            if logical_len > state.slot_context_lengths[row]:
                raise ValueError(
                    f"prompt row {row} length {logical_len} exceeds allocated slot capacity "
                    f"{state.slot_context_lengths[row]}"
                )
        if len(set(prompt_lens)) != 1:
            # Preserve mixed prompts through the same explicit state contract;
            # each logical row owns its slot and cache/page-table row.
            outputs = []
            for row, logical_len in enumerate(prompt_lens):
                row_tokens = tokens[row : row + 1, :logical_len]
                tt_tokens = self._host_tokens_to_device(row_tokens)
                tt_pos = self._positions_to_device(torch.arange(logical_len).reshape(1, logical_len), dtype=ttnn.uint32)
                outputs.append(
                    self.model.prefill_forward(
                        tt_tokens,
                        state=state,
                        prompt_lens=[logical_len],
                        position_ids=tt_pos,
                        user_id=row,
                        return_all_logits=return_all_logits,
                    )
                )
            return outputs

        logical_len = int(prompt_lens[0])
        tt_tokens = self._host_tokens_to_device(tokens[:, :logical_len])
        positions = torch.arange(logical_len).reshape(1, logical_len).repeat(tokens.shape[0], 1)
        tt_pos = self._positions_to_device(positions, dtype=ttnn.uint32)
        logits = self.model.prefill_forward(
            tt_tokens,
            state=state,
            prompt_lens=prompt_lens,
            position_ids=tt_pos,
            return_all_logits=return_all_logits,
        )
        if return_all_logits:
            return self._gather_logits_to_torch(logits, logical_len=logical_len)
        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: Any,
        kv_cache: Any,
        sampling_mode: str | None = None,
        enable_trace: bool = True,
        active_mask: torch.Tensor | None = None,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
        **kwargs: Any,
    ) -> torch.Tensor | ttnn.Tensor:
        del kwargs
        batch = tokens.shape[0]
        mode = sampling_mode or self.sampling_mode
        if mode == "host":
            # Compatibility mode intentionally gathers logits on the host.
            # Keep it eager so those transient gather buffers can never
            # invalidate the stable-address optimized decode trace.
            enable_trace = False
        state = self._state_from_args(self.model, kv_cache=kv_cache, page_table=page_table, batch_size=batch)
        if active_mask is not None:
            state.active_mask[:batch] = active_mask.to(torch.bool)
        spec = self._sampling_spec(batch, top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        if mode in ("host", "teacher"):
            spec = SamplingSpec(greedy=True)
        if enable_trace:
            trace = self._get_or_capture_decode_trace(
                tokens, start_pos, state=state, sampling_mode=mode, sampling_spec=spec
            )
            if mode in ("host", "teacher"):
                # Teacher forcing deliberately replaces device feedback, but
                # still executes the full decode graph through its trace.
                self._refresh_device_input(trace.token_input, tokens, rank4=True)
                self.trace_counters.token_refreshes += 1
            ttnn.execute_trace(self.mesh_device, trace.model_trace_id, cq_id=0, blocking=False)
            ttnn.plus_one(trace.current_pos, skip_negative_entries=True)
            ttnn.plus_one(trace.position_ids, skip_negative_entries=True)
            if trace.sampling_trace_id is not None:
                ttnn.execute_trace(self.mesh_device, trace.sampling_trace_id, cq_id=0, blocking=False)
            self.trace_counters.replays += 1
            return trace.sampled_tokens if mode in ("device", "teacher") else trace.logits

        tt_tokens = self._host_tokens_to_device(tokens, rank4=True)
        tt_current = self._positions_to_device(start_pos)
        tt_position_ids = self._positions_to_device(start_pos, dtype=ttnn.uint32)
        logits = self.model.decode_forward(
            tt_tokens, state=state, current_pos=tt_current, position_ids=tt_position_ids, batch_size=batch
        )
        if mode == "host":
            return logits
        logits = self._pad_sampling_logits(logits)
        k, p, temp, seed_tensor = self._sampling_params(spec)
        return self.sampler.decode_forward(logits, k=k, p=p, temp=temp, seeds=seed_tensor)[0]

    def _get_or_capture_decode_trace(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        state: FullModelState,
        sampling_mode: str,
        sampling_spec: SamplingSpec,
    ) -> DecodeTrace:
        page_table_ids = tuple(id(table) for table in state.page_tables)
        key = (tokens.shape[0], sampling_mode, sampling_spec.key, id(state), page_table_ids)
        if key in self._trace_cache:
            return self._trace_cache[key]
        # TT-Metal forbids allocating any device buffer while an active trace
        # pins allocator addresses. Sampling-mode/parameter transitions need
        # new persistent tensors, so intentionally release the prior trace set
        # before allocating the next one. The semantic key still prevents
        # accidental graph reuse; transitions safely recapture by design.
        page_table_changed = any(
            cached_key[0] == key[0]
            and cached_key[1] == key[1]
            and cached_key[3] == key[3]
            and cached_key[4] != page_table_ids
            for cached_key in self._trace_cache
        )
        for stale in self._trace_cache.values():
            ttnn.release_trace(self.mesh_device, stale.model_trace_id)
            if stale.sampling_trace_id is not None:
                ttnn.release_trace(self.mesh_device, stale.sampling_trace_id)
        self._trace_cache.clear()
        if page_table_changed:
            self.trace_counters.page_table_refreshes += 1
        batch = tokens.shape[0]
        token_input = self._host_tokens_to_device(tokens, rank4=True)
        padded_positions = torch.full((DECODE_SLOT_COUNT,), -1, dtype=torch.int32)
        padded_positions[:batch] = start_pos.reshape(-1).to(torch.int32)
        if bool(state.active_mask[:batch].any()):
            padded_positions[:batch] = torch.where(
                state.active_mask[:batch], padded_positions[:batch], torch.full((batch,), -1, dtype=torch.int32)
            )
        current_pos = self._positions_to_device(padded_positions)
        # UINT32 has no negative inactive sentinel; zero is safe because all
        # cache/attention updates are gated by the INT32 current_pos=-1 rows.
        rope_positions = padded_positions.clamp_min(0)
        position_ids = self._positions_to_device(rope_positions, dtype=ttnn.uint32)
        sampled_mode = sampling_mode in ("device", "teacher")
        k, p, temp, seed_tensor = self._sampling_params(sampling_spec) if sampled_mode else (None, None, None, None)

        # Exact warm path, including in-place device position advance.
        logits = self.model.decode_forward(
            token_input, state=state, current_pos=current_pos, position_ids=position_ids, batch_size=batch
        )
        sampled_tokens = token_input
        if sampled_mode:
            sampling_logits = self._pad_sampling_logits(logits)
            # Compile the exact optional-output graph that capture will use.
            # Warming the allocator-output variant is insufficient: optional
            # output changes the program hash and would upload binaries during
            # capture, which the mesh command queue correctly rejects as a write.
            sampled_tokens, _ = self.sampler.decode_forward(
                sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=token_input
            )
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(position_ids, skip_negative_entries=True)
        ttnn.synchronize_device(self.mesh_device)

        model_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits = self.model.decode_forward(
            token_input, state=state, current_pos=current_pos, position_ids=position_ids, batch_size=batch
        )
        ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)

        sampling_trace_id = None
        if sampled_mode:
            sampling_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            sampling_logits = self._pad_sampling_logits(logits)
            sampled_tokens, _ = self.sampler.decode_forward(
                sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=token_input
            )
            ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)

        # Warmup and capture execute the graph and mutate all three persistent
        # inputs. Restore the caller's logical first-step state so replay zero
        # consumes exactly `tokens` at exactly `start_pos`.
        self._refresh_device_input(token_input, tokens, rank4=True)
        self._refresh_device_input(current_pos, padded_positions)
        self._refresh_device_input(position_ids, rope_positions)
        trace = DecodeTrace(
            model_trace_id=model_trace_id,
            sampling_trace_id=sampling_trace_id,
            token_input=token_input,
            current_pos=current_pos,
            position_ids=position_ids,
            logits=logits,
            sampled_tokens=sampled_tokens,
            state=state,
            batch_size=batch,
            sampling_mode=sampling_mode,
            page_table_ids=page_table_ids,
            sampling_spec=sampling_spec,
            sampling_params=(k, p, temp, seed_tensor),
        )
        self._trace_cache[key] = trace
        return trace

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        sampling_mode: str | None = None,
        top_k: Any = 1,
        top_p: Any = 0.0,
        temperature: Any = 0.0,
        seeds: Any = 0,
        **kwargs: Any,
    ) -> List[int]:
        del kwargs
        if max_new_tokens < 1:
            return []
        start_s = time.perf_counter()
        mode = sampling_mode or ("teacher" if next_input is not None else self.sampling_mode)
        if next_input is not None and mode not in ("teacher", "host"):
            raise ValueError("teacher forcing requires sampling_mode='teacher' or explicit compatibility mode 'host'")
        state = self.model.state or self.model.allocate_state(max_batch_size=1)
        spec = self._sampling_spec(1, top_k=top_k, top_p=top_p, temperature=temperature, seeds=seeds)
        if mode in ("host", "teacher"):
            spec = SamplingSpec(greedy=True)
        logical_len = len(prompt_token_ids)
        required_context = logical_len + max_new_tokens - 1
        if required_context > state.slot_context_lengths[0]:
            raise ValueError(
                f"prompt plus generation requires {required_context} cache positions, "
                f"but slot 0 owns {state.slot_context_lengths[0]}"
            )
        physical = _padded_prefill_len(logical_len)
        prompt = torch.tensor(prompt_token_ids, dtype=torch.long).reshape(1, logical_len)
        padded = torch.nn.functional.pad(prompt, (0, physical - logical_len))
        logits = self.prefill_forward(
            padded,
            page_table=state.page_tables,
            kv_cache=state,
            prompt_lens=[logical_len],
        )
        if mode in ("device", "teacher"):
            k, p, temp, seed_tensor = self._sampling_params(spec)
            sampling_logits = ttnn.pad(
                logits,
                padding=[(0, 0), (0, 0), (0, DECODE_SLOT_COUNT - logits.shape[-2]), (0, 0)],
                value=0.0,
            )
            # Use the same explicit output contract as traced decode. The
            # allocator-output sampler variant can expose an uninitialized
            # asynchronous result on the first token on P300C.
            first_tt = self._host_tokens_to_device(torch.zeros((1, 1), dtype=torch.long), rank4=True)
            first_tt, _ = self.sampler.decode_forward(
                sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=first_tt
            )
            ttnn.synchronize_device(self.mesh_device)
            self.trace_counters.synchronizations += 1
            first_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            first_tt, _ = self.sampler.decode_forward(
                sampling_logits, k=k, p=p, temp=temp, seeds=seed_tensor, tt_out_tok=first_tt
            )
            ttnn.end_trace_capture(self.mesh_device, first_trace_id, cq_id=0)
            ttnn.execute_trace(self.mesh_device, first_trace_id, cq_id=0, blocking=True)
            ttnn.release_trace(self.mesh_device, first_trace_id)
            predicted = int(self._read_tokens(first_tt, 1)[0])
        else:
            full = ttnn.all_gather(logits, dim=3, cluster_axis=1, num_links=2, topology=ttnn.Topology.Ring)
            predicted = int(ttnn.to_torch(ttnn.get_device_tensors(full)[0]).reshape(-1).argmax())
        outputs = [predicted]
        first_token_s = time.perf_counter()
        feed = next_input(0, predicted) if next_input is not None else predicted
        eos_ids = set(self.model.eos_token_ids)
        for step in range(1, max_new_tokens):
            if next_input is None and predicted in eos_ids:
                break
            sampled = self.decode_forward(
                torch.tensor([[feed]], dtype=torch.long),
                torch.tensor([logical_len + step - 1], dtype=torch.int32),
                page_table=state.page_tables,
                kv_cache=state,
                sampling_mode=mode,
                enable_trace=enable_trace,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seeds=seeds,
            )
            if mode in ("device", "teacher"):
                predicted = int(self._read_tokens(sampled, 1)[0])
            else:
                full = ttnn.all_gather(sampled, dim=3, cluster_axis=1, num_links=2, topology=ttnn.Topology.Ring)
                predicted = int(ttnn.to_torch(ttnn.get_device_tensors(full)[0]).reshape(-1).argmax())
            outputs.append(predicted)
            feed = next_input(step, predicted) if next_input is not None else predicted
        end_s = time.perf_counter()
        decode_tokens = max(len(outputs) - 1, 0)
        decode_elapsed_s = max(end_s - first_token_s, 0.0)
        self.last_perf = {
            "ttft_ms": (first_token_s - start_s) * 1000.0,
            "elapsed_s": end_s - start_s,
            "e2e_t/s/u": len(outputs) / max(end_s - start_s, 1e-12),
            "decode_tokens": float(decode_tokens),
            "decode_elapsed_s": decode_elapsed_s,
            "decode_t/s/u": decode_tokens / max(decode_elapsed_s, 1e-12),
            "trace_replays": float(self.trace_counters.replays),
            "token_refreshes": float(self.trace_counters.token_refreshes),
            "position_refreshes": float(self.trace_counters.position_refreshes),
            "rope_refreshes": float(self.trace_counters.rope_refreshes),
            "page_table_refreshes": float(self.trace_counters.page_table_refreshes),
            "synchronizations": float(self.trace_counters.synchronizations),
            "token_readbacks": float(self.trace_counters.token_readbacks),
        }
        return outputs

    def reset(self) -> None:
        for trace in self._trace_cache.values():
            ttnn.release_trace(self.mesh_device, trace.model_trace_id)
            if trace.sampling_trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace.sampling_trace_id)
        self._trace_cache.clear()
        self.trace_counters = TraceCounters()
        self.model.state = self.model.allocate_state(max_batch_size=self.model.max_batch_size)


def build_generator(model_dir: str | Path, mesh_device: Any, **kwargs: Any) -> Gemma4Generator:
    model_dir = Path(model_dir)
    model_path = kwargs.pop("model_path", None) or kwargs.pop("hf_model", None) or "google/gemma-4-26B-A4B-it"
    max_seq_len = int(kwargs.pop("max_seq_len", 262_144))
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    num_layers = kwargs.pop("num_layers", None)
    layer_indices = kwargs.pop("layer_indices", None)
    sampling_mode = kwargs.pop("sampling_mode", "device")
    precision_config_path = kwargs.pop("precision_config_path", None)
    tensor_cache_path = Path(
        kwargs.pop("tensor_cache_path", os.environ.get("GEMMA4_TENSOR_CACHE_PATH", "/tmp/gemma4_full_model_cache"))
    )
    state_dict = kwargs.pop("state_dict", None)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unknown generator options: {unknown}")
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if state_dict is None:
        state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        layer_indices=layer_indices,
        tensor_cache_path=tensor_cache_path,
        precision_config_path=precision_config_path,
    )
    return Gemma4Generator(model, tokenizer, sampling_mode=sampling_mode)


__all__ = ["Gemma4Generator", "SamplingSpec", "TraceCounters", "build_generator"]
