# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readiness generator for the Llama-3.2-1B-Instruct TTNN full model."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.model import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_PAGE_BLOCK_SIZE,
    MODEL_ID,
    Llama32FullModel,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.multichip_decoder import (
    set_multichip_ccl_dtype_policy,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoderPrecisionPolicy,
    precision_policy_from_config,
)
from models.common.readiness_check.contract import Generator, NextInputFn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _upper_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


PRECISION_CONFIG_ENV = "MD_LLAMA32_PRECISION_CONFIG"
DEFAULT_SELECTED_PRECISION_CONFIG_RELPATH = Path("doc/datatype_sweep/selected_precision_config.json")
VLLM_UNBOUNDED_TOP_K = 1
VLLM_UNBOUNDED_TOP_P = 0.0


def _resolve_precision_config_path(model_dir: Path, precision_config_path: str | Path | None) -> Path | None:
    raw_path = precision_config_path if precision_config_path is not None else os.getenv(PRECISION_CONFIG_ENV)
    if raw_path is not None:
        raw_text = str(raw_path).strip()
        if raw_text.lower() in {"", "none", "default", "builtin"}:
            return None
        path = Path(raw_text)
        return path if path.is_absolute() else (model_dir / path)

    selected_path = model_dir / DEFAULT_SELECTED_PRECISION_CONFIG_RELPATH
    return selected_path if selected_path.exists() else None


def _load_precision_config(
    model_dir: Path,
    *,
    precision_policy: OptimizedDecoderPrecisionPolicy | None,
    precision_config_path: str | Path | None,
) -> tuple[OptimizedDecoderPrecisionPolicy, Path | None, dict[str, Any] | None]:
    if precision_policy is not None:
        set_multichip_ccl_dtype_policy(all_gather_dtype=ttnn.bfloat8_b, reduce_scatter_dtype=ttnn.bfloat8_b)
        return precision_policy, None, None

    resolved_path = _resolve_precision_config_path(model_dir, precision_config_path)
    if resolved_path is None:
        set_multichip_ccl_dtype_policy(all_gather_dtype=ttnn.bfloat8_b, reduce_scatter_dtype=ttnn.bfloat8_b)
        return OptimizedDecoderPrecisionPolicy(), None, None

    config = json.loads(resolved_path.read_text())
    policy = precision_policy_from_config(config)
    ccl_config = config.get("ccl", {})
    for key, value in ccl_config.get("runtime_env_overrides", {}).items():
        os.environ.setdefault(str(key), str(value))
    set_multichip_ccl_dtype_policy(
        all_gather_dtype=ccl_config.get("all_gather_dtype", ttnn.bfloat8_b),
        reduce_scatter_dtype=ccl_config.get("reduce_scatter_dtype", ttnn.bfloat8_b),
    )
    return policy, resolved_path, config


@dataclass
class TraceLoopCounters:
    model_trace_captures: int = 0
    model_trace_replays: int = 0
    sampling_trace_captures_or_replays: int = 0
    token_refreshes: int = 0
    current_position_refreshes: int = 0
    rope_index_refreshes: int = 0
    page_table_refreshes: int = 0
    synchronizations: int = 0
    readbacks: int = 0
    host_argmax_decode_steps: int = 0
    full_logits_decode_readbacks: int = 0
    device_position_advances: int = 0
    device_token_feedback_steps: int = 0
    async_decode_reads: int = 0
    host_decode_process_calls: int = 0


@dataclass
class DecodeTraceState:
    trace_id: int | None = None
    inputs: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None
    logits: ttnn.Tensor | None = None
    page_table_signature: tuple[int, int] | None = None


@dataclass
class GreedyTraceState:
    trace_id: int | None = None
    logits: ttnn.Tensor | None = None
    output_token: ttnn.Tensor | None = None
    persistent_tensors: tuple[ttnn.Tensor, ...] = field(default_factory=tuple)


class SplitGreedySampler:
    """Split-vocab greedy sampler that avoids full-vocab all-gather."""

    def __init__(self, *, mesh_device: ttnn.MeshDevice, tt_sampling: Any, tt_ccl: Any) -> None:
        self.mesh_device = mesh_device
        self.tt_sampling = tt_sampling
        self.tt_ccl = tt_ccl
        self.trace_state = GreedyTraceState()
        self.device_offsets: ttnn.Tensor | None = None
        self._last_eager_tensors: tuple[ttnn.Tensor, ...] = ()

    def _device_offsets(self) -> ttnn.Tensor:
        if self.device_offsets is not None:
            return self.device_offsets
        tt_sampling = self.tt_sampling
        cluster_shape = tuple(int(v) for v in tt_sampling.cluster_shape)
        num_devices = max(cluster_shape)
        per_device_vocab = int(tt_sampling.padded_vocab_size) // int(num_devices)
        offsets = torch.zeros(1, 1, tt_sampling.max_batch_size, num_devices, dtype=torch.int64)
        for device_id in range(num_devices):
            offsets[:, :, :, device_id] = device_id * per_device_vocab
        self.device_offsets = ttnn.from_torch(
            offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self.device_offsets

    def _all_gather_topk(
        self,
        tensor: ttnn.Tensor,
        *,
        buffer_key: str,
        dtype: Any | None = None,
    ) -> ttnn.Tensor:
        sampler = self.tt_sampling
        cluster_axis = sampler._get_sampling_cluster_axis()
        topology = ttnn.Topology.Ring if self.mesh_device.get_num_devices() >= 8 else ttnn.Topology.Linear
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=sampler.num_gather_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=cluster_axis,
            topology=topology,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def _forward(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        retain_intermediates: bool = False,
    ) -> tuple[ttnn.Tensor, tuple[ttnn.Tensor, ...]]:
        sampler = self.tt_sampling
        retained: list[ttnn.Tensor] = []

        def release(tensor: ttnn.Tensor) -> None:
            if retain_intermediates:
                retained.append(tensor)
            else:
                ttnn.deallocate(tensor)

        logits_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16, sub_core_grids=sampler.sub_core_grids)
        topk_input = logits_bf16
        if sampler.pad_to_power_of_2 and not _is_power_of_two(int(topk_input.shape[-1])):
            padded_width = _upper_power_of_two(int(topk_input.shape[-1]))
            topk_input = ttnn.pad(
                topk_input,
                [(0, 0), (0, 0), (0, 0), (0, padded_width - int(topk_input.shape[-1]))],
                value=-sys.float_info.max,
                sub_core_grids=sampler.sub_core_grids,
            )
            release(logits_bf16)
        local_values, local_indices = ttnn.topk(
            topk_input,
            k=sampler.max_top_k,
            dim=-1,
            sub_core_grids=sampler.sub_core_grid_topk,
            indices_tensor=sampler.tt_indices_tensor,
        )
        release(topk_input)

        gathered_values = self._all_gather_topk(
            local_values,
            buffer_key="GREEDY_VALUES",
        )
        release(local_values)
        gathered_indices = self._all_gather_topk(
            local_indices,
            buffer_key="GREEDY_INDICES",
            dtype=ttnn.uint16,
        )
        release(local_indices)

        gathered_indices_int32 = ttnn.typecast(gathered_indices, dtype=ttnn.int32, sub_core_grids=sampler.sub_core_grids)
        release(gathered_indices)
        global_indices = ttnn.add(
            sampler.tt_indices_device_offsets,
            gathered_indices_int32,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        release(gathered_indices_int32)

        gathered_values_rm = ttnn.untilize(
            gathered_values, use_multicore=True, sub_core_grids=sampler.sub_core_grids
        )
        release(gathered_values)
        winner_slots_rm = ttnn.argmax(gathered_values_rm, dim=-1, keepdim=True, use_multicore=True)
        release(gathered_values_rm)

        winner_slots_tile = ttnn.to_layout(winner_slots_rm, ttnn.TILE_LAYOUT)
        release(winner_slots_rm)
        selected_tokens_tile = ttnn.gather(
            global_indices,
            dim=-1,
            index=winner_slots_tile,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        release(winner_slots_tile)
        release(global_indices)

        selected_tokens_rm = ttnn.to_layout(selected_tokens_tile, ttnn.ROW_MAJOR_LAYOUT)
        release(selected_tokens_tile)
        selected_tokens = ttnn.reshape(selected_tokens_rm, tuple(tt_out_tok.shape))
        release(selected_tokens_rm)
        sampled = ttnn.to_memory_config(
            selected_tokens,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=tt_out_tok,
        )
        release(selected_tokens)
        if retain_intermediates and sampled is not tt_out_tok:
            retained.append(sampled)
        return tt_out_tok, tuple(retained)

    def sample(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor, enable_trace: bool) -> ttnn.Tensor:
        if not enable_trace:
            sampled, self._last_eager_tensors = self._forward(
                logits, tt_out_tok=tt_out_tok, retain_intermediates=True
            )
            return sampled
        if self.trace_state.trace_id is None:
            _, warm_tensors = self._forward(logits, tt_out_tok=tt_out_tok, retain_intermediates=True)
            self._last_eager_tensors = warm_tensors
            ttnn.synchronize_device(self.mesh_device)
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            output_token, persistent_tensors = self._forward(
                logits, tt_out_tok=tt_out_tok, retain_intermediates=True
            )
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            self.trace_state = GreedyTraceState(
                trace_id=trace_id,
                logits=logits,
                output_token=tt_out_tok,
                persistent_tensors=persistent_tensors,
            )
            return tt_out_tok
        if logits is not self.trace_state.logits or tt_out_tok is not self.trace_state.output_token:
            raise ValueError("split-greedy trace inputs changed; release the trace before replaying")
        ttnn.execute_trace(self.mesh_device, self.trace_state.trace_id, cq_id=0, blocking=False)
        return self.trace_state.output_token

    def reset_trace(self) -> None:
        if self.trace_state.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_state.trace_id)
        self.trace_state = GreedyTraceState()
        self._last_eager_tensors = ()


class Llama32Generator(Generator):
    """High-level and low-level generator for readiness checks."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = MODEL_ID,
        revision: str | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        num_layers: int | None = None,
        cache_path: str | Path | None = None,
        precision_policy: OptimizedDecoderPrecisionPolicy | None = None,
        precision_config_path: str | Path | None = None,
        use_vllm_paged_kv_cache: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.max_seq_len = int(max_seq_len)
        self.max_batch_size = int(max_batch_size)
        self.page_block_size = int(page_block_size)
        self.use_vllm_paged_kv_cache = bool(use_vllm_paged_kv_cache)
        self.precision_policy, self.precision_config_path, self.precision_config = _load_precision_config(
            self.model_dir,
            precision_policy=precision_policy,
            precision_config_path=precision_config_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, revision=revision, local_files_only=True)
        self.model = Llama32FullModel.from_pretrained(
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            revision=revision,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_block_size=page_block_size,
            num_layers=num_layers,
            cache_path=cache_path or (self.model_dir / ".ttnn_cache"),
            precision_policy=self.precision_policy,
            use_vllm_paged_kv_cache=use_vllm_paged_kv_cache,
        )
        self.model.load_device_weights()
        self.kv_cache = self.model.kv_cache
        self.sampling = SamplingGenerator(
            args=self.model.sampling_args,
            mesh_device=mesh_device,
            tt_ccl=self.model.tt_ccl,
            enable_internal_trace=True,
        )
        self.greedy_sampler = SplitGreedySampler(
            mesh_device=mesh_device,
            tt_sampling=self.sampling.tt_sampling,
            tt_ccl=self.model.tt_ccl,
        )
        self.trace_state = DecodeTraceState()
        self.counters = TraceLoopCounters()
        self._last_page_table: torch.Tensor | None = None
        self._vllm_last_page_table_signature: tuple[int, int, int, int] | None = None
        self._vllm_last_sampling_params_signature: tuple[Any, ...] | None = None
        self._last_generation_meta: dict[str, Any] = {}
        self._last_prefill_ms: float | None = None
        self._last_decode_s: list[float] = []
        self._active_sampling_params = self._make_sampling_params(top_k=1, top_p=0.0, temperature=1.0)

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _make_sampling_params(
        self,
        *,
        top_k: int,
        top_p: float,
        temperature: float,
        seed: int | None = None,
    ) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            enable_log_probs=False,
            num_logprobs=0,
        )

    def _sampling_value_list(self, value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().reshape(-1).tolist()
        if hasattr(value, "tolist"):
            as_list = value.tolist()
            return as_list if isinstance(as_list, list) else [as_list]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _format_sampling_params_for_model(self, params: SamplingParams) -> SamplingParams:
        formatted = format_sampling_params(params, self.max_batch_size)
        if not self.use_vllm_paged_kv_cache:
            return formatted

        raw_top_k_values = self._sampling_value_list(getattr(params, "top_k"))
        top_k = list(getattr(formatted, "top_k"))
        top_p = list(getattr(formatted, "top_p"))
        vocab_size = int(self.model.hf_config.vocab_size)
        changed = False
        for idx, raw_top_k in enumerate(raw_top_k_values[: len(top_k)]):
            try:
                raw_top_k_int = int(raw_top_k)
            except (TypeError, ValueError):
                continue
            if raw_top_k_int >= vocab_size:
                top_k[idx] = min(int(top_k[idx]), VLLM_UNBOUNDED_TOP_K)
                top_p[idx] = VLLM_UNBOUNDED_TOP_P
                changed = True
        if not changed:
            return formatted
        return replace(formatted, top_k=top_k, top_p=top_p)

    def _apply_sampling_params(self, params: SamplingParams) -> None:
        formatted = self._format_sampling_params_for_model(params)
        self.sampling.reset_sampling_params(formatted)
        self._active_sampling_params = params

    def _apply_prefill_sampling_state(
        self,
        params: SamplingParams,
        *,
        prompt_tokens: torch.Tensor | None,
        empty_slots: list[int],
    ) -> None:
        formatted = self._format_sampling_params_for_model(params)
        self.sampling.apply_prefill_state(
            sampling_params=formatted,
            prompt_tokens=prompt_tokens,
            empty_slots=empty_slots,
            replicate_seeds=len(empty_slots) == 1,
        )
        self._active_sampling_params = formatted
        self._vllm_last_sampling_params_signature = self._sampling_params_signature(formatted)

    def _apply_decode_sampling_state(
        self,
        params: SamplingParams,
        *,
        active_slots: list[int],
        start_pos: torch.Tensor,
        reset_batch: bool,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap: torch.Tensor | None = None,
    ) -> None:
        formatted = self._format_sampling_params_for_model(params)
        if slot_remap is not None:
            self.sampling.seed_manager.apply_slot_remap(slot_remap)
        signature = self._sampling_params_signature(formatted)
        needs_penalty_state = prompt_tokens is not None or output_tokens is not None
        if reset_batch or needs_penalty_state or signature != self._vllm_last_sampling_params_signature:
            self.sampling.apply_decode_state(
                [formatted],
                reset_batch=reset_batch,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
            self._vllm_last_sampling_params_signature = signature
        self.sampling.seed_manager.reset_seed_from_slots_if_needed(formatted.seed, active_slots)
        self.sampling.seed_manager.align_seed_counters_to_positions(formatted.seed, active_slots, start_pos, offset=1)
        self.sampling.seed_manager.get_new_values(active_slots)
        self._active_sampling_params = formatted

    def _sampling_params_signature(self, params: SamplingParams) -> tuple[Any, ...]:
        def freeze(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return tuple(value.reshape(-1).tolist())
            if isinstance(value, list):
                return tuple(value)
            return value

        return tuple(freeze(getattr(params, field)) for field in params.__dataclass_fields__)

    def _is_greedy_sampling_params(self, params: SamplingParams) -> bool:
        def first(value):
            if isinstance(value, torch.Tensor):
                return value.reshape(-1)[0].item()
            return value[0] if isinstance(value, list) else value

        return (
            int(first(params.top_k)) == 1
            and (float(first(params.top_p)) == 0.0 or float(first(params.top_p)) == 1.0)
        )

    def _sampling_requests_log_probs(self, params: SamplingParams) -> bool:
        value = getattr(params, "enable_log_probs", False)
        if isinstance(value, torch.Tensor):
            return bool(value.bool().any().item())
        if isinstance(value, list):
            return any(bool(item) for item in value)
        return bool(value)

    def _normalize_page_table(self, page_table: torch.Tensor | ttnn.Tensor | None) -> torch.Tensor:
        if page_table is None:
            return self.model.make_page_table(batch_size=self.max_batch_size, max_seq_len=self.max_seq_len)
        if isinstance(page_table, torch.Tensor):
            if page_table.shape[0] == self.max_batch_size:
                return page_table.to(torch.int32)
            out = self.model.make_page_table(batch_size=self.max_batch_size, max_seq_len=self.max_seq_len)
            out[: page_table.shape[0], : page_table.shape[1]] = page_table.to(torch.int32)
            return out
        raise TypeError(
            "page_table must be a torch.Tensor at the generator boundary when a trace may need refreshing; "
            "pass None for the internally owned contiguous table"
        )

    def _active_blocks_from_prompt_lens(self, prompt_lens: list[int]) -> list[int]:
        return [max(1, math.ceil(int(prompt_len) / self.page_block_size)) for prompt_len in prompt_lens]

    def _active_blocks_from_decode_positions(self, start_pos: torch.Tensor) -> list[int]:
        blocks: list[int] = []
        for pos in start_pos.reshape(-1).to(torch.int64).tolist()[: self.max_batch_size]:
            if int(pos) < 0:
                blocks.append(0)
            else:
                blocks.append(max(1, int(pos) // self.page_block_size + 1))
        return blocks

    def _sanitize_page_table_tail(self, page_table: torch.Tensor, active_blocks: list[int]) -> torch.Tensor:
        sanitized = page_table.to(torch.int32).clone()
        cols = int(sanitized.shape[1])
        block_offset = self._vllm_page_table_block_offset()
        for row, active in enumerate(active_blocks[: sanitized.shape[0]]):
            active = int(active)
            if active <= 0:
                continue
            active = min(active, cols)
            if self.use_vllm_paged_kv_cache:
                sanitized[row, :active] = torch.arange(block_offset, block_offset + active, dtype=torch.int32)
            if active < cols:
                sanitized[row, active:] = 0
        return sanitized

    def _vllm_page_table_block_offset(self) -> int:
        if not self.use_vllm_paged_kv_cache:
            return 0
        try:
            num_blocks = int(self.kv_cache[0][0].shape[0])
        except (AttributeError, IndexError, TypeError):
            return 0
        required_blocks = math.ceil(self.max_seq_len / self.page_block_size)
        return 1 if num_blocks > required_blocks else 0

    def _page_table_signature(self, page_table: torch.Tensor) -> tuple[int, int]:
        return int(page_table.data_ptr()), int(page_table.sum().item())

    def _page_table_value_signature(self, page_table: torch.Tensor) -> tuple[int, int, int, int]:
        normalized = page_table.to(torch.int32).contiguous()
        checksum = zlib.crc32(normalized.numpy().tobytes())
        return int(normalized.shape[0]), int(normalized.shape[1]), int(normalized.sum().item()), int(checksum)

    def _pad_prefill_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        if tokens.dim() != 2 or tokens.shape[0] != 1:
            raise ValueError(f"prefill tokens must be [1, seq_len], got {tuple(tokens.shape)}")
        real_len = int(tokens.shape[1])
        padded_len = max(128, _ceil_to_multiple(real_len, 128))
        if padded_len > self.max_seq_len:
            raise ValueError(f"prompt length {real_len} pads to {padded_len}, exceeds max_seq_len={self.max_seq_len}")
        if padded_len == real_len:
            return tokens.to(torch.long), real_len
        pad = torch.zeros(1, padded_len - real_len, dtype=torch.long)
        return torch.cat([tokens.to(torch.long), pad], dim=1), real_len

    def _read_token(self, tt_tokens: ttnn.Tensor) -> int:
        host = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0])
        self.counters.readbacks += 1
        return int(host.flatten()[0].item())

    def token_out_to_torch(self, tt_tokens: ttnn.Tensor) -> torch.Tensor:
        host = ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0])
        self.counters.readbacks += 1
        return host.reshape(-1)[: self.max_batch_size].to(torch.int32)

    def log_probs_to_torch(self, tt_log_probs: ttnn.Tensor | None) -> torch.Tensor | None:
        if tt_log_probs is None:
            return None
        host = ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0])
        self.counters.readbacks += 1
        return host.reshape(-1)[: self.max_batch_size].to(torch.float32)

    def _sample_logits_with_log_probs(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        enable_trace: bool,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        use_split_greedy = (
            self._is_greedy_sampling_params(self._active_sampling_params)
            and not self._sampling_requests_log_probs(self._active_sampling_params)
        )
        if use_split_greedy:
            before = self.greedy_sampler.trace_state.trace_id is not None
            sampled = self.greedy_sampler.sample(logits, enable_trace=enable_trace, tt_out_tok=tt_out_tok)
            after = self.greedy_sampler.trace_state.trace_id is not None
            self.counters.sampling_trace_captures_or_replays += 1
            if after and not before:
                self.counters.sampling_trace_captures_or_replays += 0
            return sampled, None

        before = sum(1 for slot in self.sampling._trace_states.values() if slot.get("id") is not None)
        sampled = self.sampling.sample(
            logits,
            enable_trace=enable_trace,
            tt_out_tok=tt_out_tok,
        )
        after = sum(1 for slot in self.sampling._trace_states.values() if slot.get("id") is not None)
        self.counters.sampling_trace_captures_or_replays += 1
        if after > before:
            self.counters.sampling_trace_captures_or_replays += 0
        if isinstance(sampled, tuple):
            return sampled[0], sampled[1]
        return sampled, None

    def _sample_logits(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor, enable_trace: bool) -> ttnn.Tensor:
        return self._sample_logits_with_log_probs(logits, tt_out_tok=tt_out_tok, enable_trace=enable_trace)[0]

    def bind_external_kv_cache(self, kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]]) -> None:
        self.model.bind_external_kv_cache(kv_cache)
        self.kv_cache = kv_cache

    def _active_slots_from_positions(self, start_pos: torch.Tensor) -> list[int]:
        positions = start_pos.reshape(-1).to(torch.int64)
        return [idx for idx, pos in enumerate(positions.tolist()[: self.max_batch_size]) if int(pos) >= 0]

    # ------------------------------------------------------------------
    # Low-level API
    # ------------------------------------------------------------------

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor | None,
        kv_cache: Any,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kv_cache is not None and kv_cache is not self.kv_cache:
            raise ValueError("external kv_cache must be the cache returned by this generator/model")
        if len(prompt_lens) != 1:
            raise ValueError("this full-model readiness generator currently supports batch-1 prefill")

        padded_tokens, real_len = self._pad_prefill_tokens(tokens)
        if isinstance(page_table, ttnn.Tensor):
            page_table_tt = page_table
        else:
            page_table_host = self._normalize_page_table(page_table)
            page_table_host = self._sanitize_page_table_tail(
                page_table_host,
                self._active_blocks_from_prompt_lens(prompt_lens),
            )
            page_table_tt = self.model.page_table_to_device(page_table_host)
        tokens_tt = self.model.prepare_prefill_tokens_device(padded_tokens)

        start = time.perf_counter()
        logits_tt = self.model.prefill_forward_device(tokens_tt, page_table=page_table_tt, start_pos=0, user_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1
        self._last_prefill_ms = (time.perf_counter() - start) * 1000.0

        logits = self.model.logits_to_torch(logits_tt, rows=int(padded_tokens.shape[1]))
        if return_all_logits:
            return logits[:, 0, :real_len, :].contiguous()
        last_idx = int(prompt_lens[0]) - 1
        return logits[:, 0, last_idx : last_idx + 1, :].contiguous()

    def prefill_token_out_host(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor | None,
        kv_cache: Any,
        prompt_lens: list[int],
        sampling_params: SamplingParams,
        empty_slots: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is not None and kv_cache is not self.kv_cache:
            raise ValueError("external kv_cache must be the cache returned by this generator/model")
        if len(prompt_lens) != 1:
            raise ValueError("this vLLM adapter currently supports batch-1 prefill")

        empty_slots = [0] if empty_slots is None else [int(slot) for slot in empty_slots]
        self._apply_prefill_sampling_state(
            sampling_params,
            prompt_tokens=tokens.to(torch.int32),
            empty_slots=empty_slots,
        )

        padded_tokens, real_len = self._pad_prefill_tokens(tokens)
        if isinstance(page_table, ttnn.Tensor):
            page_table_tt = page_table
        else:
            page_table_host = self._normalize_page_table(page_table)
            page_table_host = self._sanitize_page_table_tail(
                page_table_host,
                self._active_blocks_from_prompt_lens(prompt_lens),
            )
            page_table_tt = self.model.page_table_to_device(page_table_host)
        tokens_tt = self.model.prepare_prefill_tokens_device(padded_tokens)

        start = time.perf_counter()
        logits_tt = self.model.prefill_forward_device(tokens_tt, page_table=page_table_tt, start_pos=0, user_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1
        self._last_prefill_ms = (time.perf_counter() - start) * 1000.0

        last_idx = int(prompt_lens[0]) - 1
        if last_idx < 0 or last_idx >= real_len:
            raise ValueError(f"invalid prompt_lens={prompt_lens} for prefill length {real_len}")
        tile_start = (last_idx // 32) * 32
        active_slot = last_idx - tile_start
        last_logits = ttnn.slice(logits_tt, (0, 0, tile_start, 0), (1, 1, tile_start + 32, logits_tt.shape[-1]))
        tt_out_tok = ttnn.from_torch(
            torch.zeros(1, 1, 1, self.max_batch_size, dtype=torch.uint32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tt_tokens, tt_log_probs = self._sample_logits_with_log_probs(
            last_logits,
            tt_out_tok=tt_out_tok,
            enable_trace=False,
        )
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1
        host_tokens = self.token_out_to_torch(tt_tokens)
        host_tokens[0] = host_tokens[active_slot]
        host_log_probs = self.log_probs_to_torch(tt_log_probs)
        if host_log_probs is not None:
            host_log_probs[0] = host_log_probs[active_slot]
            return host_tokens, host_log_probs
        return host_tokens

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        sampling_params: SamplingParams | None = None,
        return_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kv_cache is not None and kv_cache is not self.kv_cache:
            raise ValueError("external kv_cache must be the cache returned by this generator/model")
        params = sampling_params or self._active_sampling_params
        self._apply_sampling_params(params)
        page_table_host = self._normalize_page_table(page_table)
        page_table_host = self._sanitize_page_table_tail(
            page_table_host,
            self._active_blocks_from_decode_positions(start_pos.reshape(-1)),
        )
        if return_logits:
            logits = self._decode_logits_host(tokens.reshape(-1), start_pos.reshape(-1), page_table_host, enable_trace)
            return logits
        tt_tokens = self._decode_token_out_device(
            tokens.reshape(-1),
            start_pos.reshape(-1),
            page_table_host,
            enable_trace=enable_trace,
            reset_inputs=True,
            page_table_changed=True,
        )
        return torch.tensor([self._read_token(tt_tokens)], dtype=torch.long)

    def _decode_logits_host(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        enable_trace: bool,
    ) -> torch.Tensor:
        if enable_trace:
            logits = self._decode_model_trace(
                tokens,
                start_pos,
                page_table,
                reset_inputs=True,
                page_table_changed=True,
            )
        else:
            host_inputs = self.model.prepare_decode_inputs_host(tokens, start_pos, page_table)
            device_inputs = self.model.copy_decode_inputs_to_device(host_inputs)
            logits = self.model.decode_forward_device(*device_inputs)
        self.counters.full_logits_decode_readbacks += 1
        return self.model.logits_to_torch(logits, rows=int(tokens.numel()))[:, 0, : int(tokens.numel()), :]

    def decode_token_out_device_for_vllm(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool,
        sampling_params: SamplingParams,
        reset_batch: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap: torch.Tensor | None = None,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        if kv_cache is not None and kv_cache is not self.kv_cache:
            raise ValueError("external kv_cache must be the cache returned by this generator/model")

        tokens_flat = tokens.reshape(-1)
        start_pos_flat = start_pos.reshape(-1)
        raw_page_table_host = self._normalize_page_table(page_table)
        page_table_host = self._sanitize_page_table_tail(
            raw_page_table_host,
            self._active_blocks_from_decode_positions(start_pos_flat),
        )
        active_slots = self._active_slots_from_positions(start_pos_flat)
        self._apply_decode_sampling_state(
            sampling_params,
            active_slots=active_slots,
            start_pos=start_pos_flat,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
        )

        if not enable_trace:
            tt_tokens, tt_log_probs = self._decode_token_out_device_no_trace(
                tokens_flat,
                start_pos_flat,
                page_table_host,
            )
        else:
            page_table_signature = self._page_table_value_signature(raw_page_table_host)
            reset_inputs = bool(reset_batch or self.trace_state.trace_id is None)
            page_table_changed = self._vllm_last_page_table_signature != page_table_signature
            tt_tokens, tt_log_probs = self._decode_token_out_device_with_log_probs(
                tokens_flat,
                start_pos_flat,
                page_table_host,
                enable_trace=True,
                reset_inputs=reset_inputs,
                page_table_changed=page_table_changed,
            )
            self._vllm_last_page_table_signature = page_table_signature

        if tt_log_probs is not None:
            return tt_tokens, tt_log_probs
        return tt_tokens

    def _decode_token_out_device_no_trace(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        host_inputs = self.model.prepare_decode_inputs_host(tokens, start_pos, page_table)
        device_inputs = self.model.copy_decode_inputs_to_device(host_inputs)
        self.counters.token_refreshes += 1
        self.counters.current_position_refreshes += 1
        self.counters.rope_index_refreshes += 1
        self.counters.page_table_refreshes += 1
        logits = self.model.decode_forward_device(*device_inputs)
        tt_tokens, tt_log_probs = self._sample_logits_with_log_probs(
            logits,
            tt_out_tok=device_inputs[0],
            enable_trace=False,
        )
        self.counters.device_position_advances += 1
        self.counters.device_token_feedback_steps += 1
        return tt_tokens, tt_log_probs

    # ------------------------------------------------------------------
    # Traced split-sampling decode
    # ------------------------------------------------------------------

    def _capture_decode_trace(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
    ) -> ttnn.Tensor:
        host_inputs = self.model.prepare_decode_inputs_host(tokens, start_pos, page_table)
        device_inputs = self.model.copy_decode_inputs_to_device(host_inputs)
        self.counters.token_refreshes += 1
        self.counters.current_position_refreshes += 1
        self.counters.rope_index_refreshes += 1
        self.counters.page_table_refreshes += 1

        warm_logits = self.model.decode_forward_device(*device_inputs)
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1
        del warm_logits

        host_inputs = self.model.prepare_decode_inputs_host(tokens, start_pos, page_table)
        self.model.copy_decode_inputs_to_device(host_inputs, device_inputs)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits = self.model.decode_forward_device(*device_inputs)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1

        self.trace_state = DecodeTraceState(
            trace_id=trace_id,
            inputs=device_inputs,
            logits=logits,
            page_table_signature=self._page_table_signature(page_table),
        )
        self.counters.model_trace_captures += 1
        self.counters.device_position_advances += 1
        return logits

    def _decode_model_trace(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        reset_inputs: bool,
        page_table_changed: bool,
    ) -> ttnn.Tensor:
        if self.trace_state.trace_id is None:
            return self._capture_decode_trace(tokens, start_pos, page_table)

        assert self.trace_state.inputs is not None
        assert self.trace_state.logits is not None
        if reset_inputs:
            host_inputs = self.model.prepare_decode_inputs_host(tokens, start_pos, page_table)
            self.model.copy_decode_inputs_to_device(host_inputs, self.trace_state.inputs)
            self.counters.token_refreshes += 1
            self.counters.current_position_refreshes += 1
            self.counters.rope_index_refreshes += 1
            self.counters.page_table_refreshes += 1
        elif page_table_changed:
            host_inputs = self.model.prepare_decode_inputs_host(tokens, start_pos, page_table)
            self.model.copy_page_table_to_device(host_inputs[3], self.trace_state.inputs[3])
            self.counters.page_table_refreshes += 1

        ttnn.execute_trace(self.mesh_device, self.trace_state.trace_id, cq_id=0, blocking=False)
        self.counters.model_trace_replays += 1
        self.counters.device_position_advances += 1
        return self.trace_state.logits

    def _decode_token_out_device(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,
        reset_inputs: bool,
        page_table_changed: bool,
    ) -> ttnn.Tensor:
        return self._decode_token_out_device_with_log_probs(
            tokens,
            start_pos,
            page_table,
            enable_trace=enable_trace,
            reset_inputs=reset_inputs,
            page_table_changed=page_table_changed,
        )[0]

    def _decode_token_out_device_with_log_probs(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        enable_trace: bool,
        reset_inputs: bool,
        page_table_changed: bool,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        if not enable_trace:
            raise RuntimeError("enable_trace=False is a debug-only path and is not valid for readiness decode")
        logits = self._decode_model_trace(
            tokens,
            start_pos,
            page_table,
            reset_inputs=reset_inputs,
            page_table_changed=page_table_changed,
        )
        assert self.trace_state.inputs is not None
        tt_tokens, tt_log_probs = self._sample_logits_with_log_probs(
            logits,
            tt_out_tok=self.trace_state.inputs[0],
            enable_trace=True,
        )
        self.counters.device_token_feedback_steps += 1
        return tt_tokens, tt_log_probs

    # ------------------------------------------------------------------
    # High-level generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_on_eos: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []
        if not enable_trace:
            raise RuntimeError("generate(enable_trace=False) is debug-only and not accepted for readiness")

        self._apply_sampling_params(self._make_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature))
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        page_table = self.model.make_page_table(batch_size=self.max_batch_size, max_seq_len=self.max_seq_len)
        self._last_page_table = page_table.clone()
        self._last_decode_s = []
        self._last_generation_meta = {
            "prompt_len": len(prompt_token_ids),
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "teacher_forcing": next_input is not None,
        }

        prefill_logits = self.prefill_forward(
            prompt,
            page_table=page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=False,
        )
        first = int(torch.argmax(prefill_logits[0, -1]).item())
        predictions = [first]
        if stop_on_eos and first == self.tokenizer.eos_token_id and next_input is None:
            return predictions

        token_to_feed = next_input(0, first) if next_input is not None else first
        current_pos_value = len(prompt_token_ids)
        reset_inputs = True
        page_table_changed = True

        for step in range(1, max_new_tokens):
            step_start = time.perf_counter()
            tt_tokens = self._decode_token_out_device(
                torch.tensor([token_to_feed], dtype=torch.long),
                torch.tensor([current_pos_value], dtype=torch.long),
                page_table,
                enable_trace=enable_trace,
                reset_inputs=reset_inputs,
                page_table_changed=page_table_changed,
            )
            predicted = self._read_token(tt_tokens)
            self._last_decode_s.append(time.perf_counter() - step_start)
            predictions.append(predicted)
            if stop_on_eos and predicted == self.tokenizer.eos_token_id and next_input is None:
                break
            if next_input is not None:
                token_to_feed = next_input(step, predicted)
                reset_inputs = True
            else:
                # This value is returned to the caller only; device feedback owns the next decode input.
                token_to_feed = predicted
                reset_inputs = False
            current_pos_value += 1
            page_table_changed = False

        return predictions

    def benchmark_token_out_no_readback(
        self,
        prompt_token_ids: list[int],
        decode_steps: int,
        *,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        warmup_steps: int = 1,
        signpost_labels: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Measure steady-state token-out replay without per-token host readback.

        The prompt/prefill and first decode step initialize the KV cache, persistent
        trace inputs, model trace, and sampling trace. The measured loop then
        replays model decode plus split sampling with `tt_out_tok` feedback; it
        does not refresh token/current-position/RoPE/page-table tensors from host
        and performs one final synchronization after all replay work is queued.
        """

        if decode_steps <= 0:
            raise ValueError(f"decode_steps must be positive, got {decode_steps}")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")

        self._apply_sampling_params(self._make_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature))
        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        page_table = self.model.make_page_table(batch_size=self.max_batch_size, max_seq_len=self.max_seq_len)
        self._last_page_table = page_table.clone()
        self._last_decode_s = []
        self._last_generation_meta = {
            "prompt_len": len(prompt_token_ids),
            "max_new_tokens": decode_steps,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "teacher_forcing": False,
            "no_readback_benchmark": True,
            "warmup_steps": warmup_steps,
        }

        prefill_logits = self.prefill_forward(
            prompt,
            page_table=page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=False,
        )
        first_token = int(torch.argmax(prefill_logits[0, -1]).item())
        current_pos_value = len(prompt_token_ids)

        warmup_start_counters = asdict(self.counters)
        token_to_feed = first_token
        for warmup_idx in range(warmup_steps):
            self._decode_token_out_device(
                torch.tensor([token_to_feed], dtype=torch.long),
                torch.tensor([current_pos_value], dtype=torch.long),
                page_table,
                enable_trace=True,
                reset_inputs=warmup_idx == 0,
                page_table_changed=warmup_idx == 0,
            )
            token_to_feed = 0
            current_pos_value += 1

        dummy_token = torch.zeros(1, dtype=torch.long)
        dummy_pos = torch.tensor([current_pos_value], dtype=torch.long)
        decode_start_counters = asdict(self.counters)

        if signpost_labels is not None:
            from tracy import signpost

            signpost(signpost_labels[0])
        start = time.perf_counter()
        for _ in range(decode_steps):
            self._decode_token_out_device(
                dummy_token,
                dummy_pos,
                page_table,
                enable_trace=True,
                reset_inputs=False,
                page_table_changed=False,
            )
        ttnn.synchronize_device(self.mesh_device)
        elapsed_s = time.perf_counter() - start
        self.counters.synchronizations += 1
        if signpost_labels is not None:
            signpost(signpost_labels[1])

        mean_step_s = elapsed_s / decode_steps
        self._last_decode_s = [mean_step_s] * decode_steps
        final_counters = asdict(self.counters)

        def counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
            return {key: int(after[key]) - int(before.get(key, 0)) for key in after}

        return {
            "model": MODEL_ID,
            "prompt_len": len(prompt_token_ids),
            "decode_steps": decode_steps,
            "warmup_steps": warmup_steps,
            "precision_policy": self.precision_policy.to_dict(),
            "precision_config_path": str(self.precision_config_path) if self.precision_config_path is not None else None,
            "precision_config_id": (self.precision_config or {}).get("config_id"),
            "prefill_ttft_ms": self._last_prefill_ms,
            "decode_elapsed_s": elapsed_s,
            "decode_latency_ms_mean": mean_step_s * 1000.0,
            "decode_token_out_no_readback_t_s_u": decode_steps / elapsed_s if elapsed_s > 0 else None,
            "sampling": {
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "greedy": self._is_greedy_sampling_params(self._active_sampling_params),
                "split_greedy_trace_captured": self.greedy_sampler.trace_state.trace_id is not None,
                "common_sampling_internal_traces": sum(
                    1 for slot in self.sampling._trace_states.values() if slot.get("id") is not None
                ),
                "force_argmax_enabled": self.sampling.tt_sampling.force_argmax_sampling,
                "pad_logits_to_power_of_2": self.sampling.tt_sampling.pad_to_power_of_2,
                "local_logits_width": int(self.model.per_device_vocab_size),
                "local_topk_input_width": _upper_power_of_two(int(self.model.per_device_vocab_size))
                if self.sampling.tt_sampling.pad_to_power_of_2
                else int(self.model.per_device_vocab_size),
            },
            "counter_deltas": {
                "warmup": counter_delta(decode_start_counters, warmup_start_counters),
                "measured_decode": counter_delta(final_counters, decode_start_counters),
            },
            "final_counters": final_counters,
            "host_boundary_audit": {
                "per_token_token_refreshes": 0,
                "per_token_current_position_refreshes": 0,
                "per_token_rope_index_refreshes": 0,
                "per_token_page_table_refreshes": 0,
                "per_token_readbacks": 0,
                "final_synchronizations": 1,
                "trace_replay_blocking": False,
            },
        }

    def reset(self) -> None:
        self.model.reset_kv_cache()
        self.sampling.reset_output_state()
        self._last_page_table = None
        self._vllm_last_page_table_signature = None
        self._vllm_last_sampling_params_signature = None
        self._last_prefill_ms = None
        self._last_decode_s = []
        self._last_generation_meta = {}
        self.counters = TraceLoopCounters()

    def teardown(self) -> None:
        if self.trace_state.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_state.trace_id)
        self.greedy_sampler.reset_trace()
        self.sampling.reset_trace()
        self.trace_state = DecodeTraceState()
        self._vllm_last_page_table_signature = None
        self._vllm_last_sampling_params_signature = None

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def trace_audit(self) -> dict[str, Any]:
        decode_times = self._last_decode_s
        return {
            "model": MODEL_ID,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "page_block_size": self.page_block_size,
            "prefill_ttft_ms": self._last_prefill_ms,
            "decode_token_out_t_s_u": (len(decode_times) / sum(decode_times)) if decode_times and sum(decode_times) else None,
            "decode_latency_ms_mean": (sum(decode_times) / len(decode_times) * 1000.0) if decode_times else None,
            "trace_state": {
                "model_trace_captured": self.trace_state.trace_id is not None,
                "sampling_internal_traces": (
                    sum(1 for slot in self.sampling._trace_states.values() if slot.get("id") is not None)
                    + int(self.greedy_sampler.trace_state.trace_id is not None)
                ),
                "split_greedy_trace_captured": self.greedy_sampler.trace_state.trace_id is not None,
            },
            "counters": asdict(self.counters),
            "last_generation": self._last_generation_meta,
            "sampling_policy": {
                "canonical": "split greedy top-1 for greedy; SamplingGenerator split top-k/top-p for sampled requests",
                "force_argmax_enabled": self.sampling.tt_sampling.force_argmax_sampling,
                "allow_force_argmax": getattr(self.sampling.tt_sampling, "_allow_force_argmax_sampling", False),
                "max_top_k": self.sampling.tt_sampling.max_top_k,
            },
            "precision_policy": self.precision_policy.to_dict(),
            "precision_config_path": str(self.precision_config_path) if self.precision_config_path is not None else None,
            "precision_config_id": (self.precision_config or {}).get("config_id"),
        }

    def write_trace_audit(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.trace_audit(), indent=2) + "\n")


def build_generator(model_dir: str | Path, mesh_device, **kwargs: Any) -> Llama32Generator:
    return Llama32Generator(model_dir=model_dir, mesh_device=mesh_device, **kwargs)


__all__ = ["Llama32Generator", "build_generator"]
