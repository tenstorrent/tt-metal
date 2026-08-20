# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for ``Qwen/Qwen3.6-35B-A3B``."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import MODEL_ID, QwenFullAttentionCache
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenReadinessGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenFullModelCache
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder
from models.common.sampling import format_sampling_params
from models.common.utility_functions import nearest_32


def _model_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _context_contract_len() -> int:
    with (_model_dir() / "doc" / "context_contract.json").open(encoding="utf-8") as f:
        data = json.load(f)
    return int(data["supported_context"])


def _tt_int(tensor: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.to(torch.int32).contiguous(),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class Qwen3_5MoeForConditionalGeneration(QwenReadinessGenerator):
    """TT vLLM bridge using QwenReadinessGenerator low-level methods."""

    is_text_generation_model = True
    is_pooling_model = False
    supports_multimodal = False
    supports_pp = False
    is_hybrid = True

    model_capabilities = {
        "supports_async_decode": True,
        "supports_async_decode_overlap": False,
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "supports_slot_independent_device_seeds": False,
        "supports_on_device_penalties": False,
        "supports_mixed_greedy_random_device_sampling": False,
    }

    def __init__(self, *args, vllm_max_batch_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_max_batch_size = int(vllm_max_batch_size or self.model.max_batch_size)
        self._last_page_table_host: torch.Tensor | None = None
        self._vllm_decode_warmed = False
        self._vllm_trace_warmed_cache_ids: set[int] = set()

    def reset(self) -> None:
        self._last_page_table_host = None
        self._vllm_trace_warmed_cache_ids.clear()
        super().reset()

    # vLLM's registry inspects these methods to classify external models as
    # generative. The TT plugin never executes this path; serving goes through
    # prefill_forward / decode_forward below.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            f"{type(self).__name__} is a TT bridge; embeddings happen on TT through prefill_forward / decode_forward."
        )

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            f"{type(self).__name__} is a TT bridge; the TT runner invokes prefill_forward / decode_forward."
        )

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(f"{type(self).__name__} is a TT bridge; sampled token output is produced on TT.")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=None,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        if optimizations is not None:
            raise ValueError(f"{cls.__name__} does not support custom optimizations={optimizations!r}")
        if int(tt_data_parallel) != 1:
            raise ValueError(f"{cls.__name__} expects a single 2x2 TT mesh, got tt_data_parallel={tt_data_parallel}")
        context_len = _context_contract_len()
        resolved_max_seq_len = int(max_seq_len or context_len)
        if resolved_max_seq_len != context_len:
            raise ValueError(
                f"served max_model_len must match context_contract.json ({context_len}); got {resolved_max_seq_len}"
            )
        model_id = getattr(hf_config, "_name_or_path", None) or MODEL_ID
        serving_batch = nearest_32(int(max_batch_size))
        return cls(
            model_dir=_model_dir(),
            mesh_device=mesh_device,
            model_id=model_id,
            local_files_only=True,
            max_batch_size=serving_batch,
            max_seq_len=resolved_max_seq_len,
            host_sampling_compat=True,
            vllm_max_batch_size=int(max_batch_size),
        )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        model_name: str,
        num_devices: int,
        tt_data_parallel: int,
        max_model_len: int,
        max_num_seqs: int,
    ) -> int:
        del model_name, num_devices, tt_data_parallel, max_num_seqs
        return min(int(max_model_len), _context_contract_len())

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        hf_config = vllm_config.model_config.hf_text_config
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            getattr(hf_config, "mamba_ssm_dtype", None),
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            parallel_config.tensor_parallel_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        del dtype, num_layers
        num_blocks, _local_kv_heads, block_size, head_dim = (int(dim) for dim in kv_cache_shape)
        if head_dim != int(self.model.cfg.head_dim):
            raise ValueError(f"vLLM KV head_dim {head_dim} does not match model head_dim {self.model.cfg.head_dim}")

        self.model.block_size = block_size
        if self.model.prefill_chunk_size % block_size != 0:
            chunks = math.ceil(self.model.prefill_chunk_size / block_size)
            self.model.prefill_chunk_size = max(block_size, chunks * block_size)

        cache_shape = (
            num_blocks,
            int(self.model.cfg.num_key_value_heads),
            block_size,
            head_dim,
        )
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 1), mesh_shape=self.mesh_device.shape)
        full_attention = {}
        for layer_idx in self.model.full_attention_layers:
            keys = ttnn.as_tensor(
                torch.zeros(cache_shape, dtype=torch.bfloat16),
                device=self.mesh_device,
                dtype=self.model.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            values = ttnn.as_tensor(
                torch.zeros(cache_shape, dtype=torch.bfloat16),
                device=self.mesh_device,
                dtype=self.model.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            full_attention[layer_idx] = QwenFullAttentionCache(
                keys=keys,
                values=values,
                block_size=block_size,
                max_num_blocks=num_blocks,
                max_batch_size=self.model.max_batch_size,
                max_seq_len=self.model.max_seq_len,
            )

        linear_attention = {
            layer_idx: MultichipDecoder.allocate_linear_attention_state(
                hf_config=self.model.text_config,
                mesh_device=self.mesh_device,
                batch_size=self.model.max_batch_size,
                dtype=self.model.linear_state_dtype,
            )
            for layer_idx in self.model.linear_attention_layers
        }
        page_table_width = math.ceil(self.model.max_seq_len / block_size)
        page_table_host = torch.zeros((self.model.max_batch_size, page_table_width), dtype=torch.int32)
        cache = QwenFullModelCache(
            full_attention=full_attention,
            linear_attention=linear_attention,
            page_table=_tt_int(page_table_host, self.mesh_device),
            page_table_host=page_table_host,
            max_batch_size=self.model.max_batch_size,
            max_seq_len=self.model.max_seq_len,
            block_size=block_size,
        )
        self.cache = cache
        self._vllm_kv_cache_shape = tuple(int(dim) for dim in kv_cache_shape)
        return cache

    def warmup_model_prefill(self, *, kv_cache, can_sample_on_device: bool, enable_trace: bool, **kwargs) -> None:
        del enable_trace, kwargs
        if can_sample_on_device:
            self._reset_sampling_state()
        self.cache = kv_cache

    def warmup_model_decode(
        self,
        *,
        kv_cache,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs,
    ) -> None:
        decode_width = int(max_batch_size)
        del num_blocks, enable_trace, kwargs
        self.cache = kv_cache
        self._release_decode_trace()
        if can_sample_on_device and self.model.sampling is not None:
            self._reset_sampling_state()
            self._warmup_decode_for_trace(width=decode_width)

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table,
        kv_cache,
        enable_trace: bool,
        prompt_lens,
        start_pos=None,
        sampling_params=None,
        empty_slots=None,
        **kwargs,
    ):
        del enable_trace, kwargs
        if start_pos is not None and torch.as_tensor(start_pos).reshape(-1).ne(0).any():
            raise NotImplementedError("chunked/prefix prefill is disabled for this Qwen vLLM adapter")
        self._release_decode_trace()
        prompt_lens = [int(length) for length in prompt_lens]
        rope_deltas = torch.zeros((len(prompt_lens),), dtype=torch.int32)
        empty_slots = list(range(tokens.shape[0])) if empty_slots is None else [int(slot) for slot in empty_slots]
        cache = self._prepare_serving_cache(kv_cache, page_table, force=True, page_table_slots=empty_slots)
        if sampling_params is not None:
            sampled = self.vllm_prefill_sample_on_device(
                tokens,
                cache=cache,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
                empty_slots=empty_slots,
            )
            return sampled, rope_deltas
        if not self.host_sampling_compat:
            raise RuntimeError("host-logits prefill is disabled; vLLM must pass sampling_params for device sampling")
        if empty_slots:
            self.model.reset_linear_attention_state(cache, empty_slots)
        logits = self._prefill_host_logits(tokens, cache=cache, prompt_lens=prompt_lens, empty_slots=empty_slots)
        return logits, rope_deltas

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table,
        kv_cache,
        start_pos,
        enable_trace: bool,
        read_from_device: bool,
        sampling_params=None,
        prompt_tokens=None,
        output_tokens=None,
        reset_batch: bool | None = None,
        slot_remap=None,
        **kwargs,
    ):
        del kwargs
        cache = self._prepare_serving_cache(kv_cache, page_table, force=bool(reset_batch))
        slot_remap_moves_state = self._slot_remap_moves_state(slot_remap)
        if slot_remap is not None:
            self.model.remap_linear_attention_state(cache, slot_remap)
            if self.model.sampling is not None:
                self.model.sampling.seed_manager.apply_slot_remap(slot_remap)

        if sampling_params is None:
            if not self.host_sampling_compat:
                raise RuntimeError("host-logits decode is disabled; vLLM must pass sampling_params for device sampling")
            return self._decode_host_logits(tokens, start_pos, cache=cache)

        if self.model.sampling is None:
            raise RuntimeError("vLLM decode sampling requires on-device sampling")
        max_batch = int(self.model.sampling.tt_sampling.max_batch_size)
        active_slots = self._active_decode_slots(start_pos, max_batch=max_batch)
        if enable_trace and self._trace is None:
            self._warmup_decode_for_trace(width=max_batch)
        reset_inputs = (
            bool(reset_batch) or slot_remap_moves_state or self._trace is None or self._trace.cache is not cache
        )
        formatted_params = format_sampling_params(sampling_params, max_batch)
        if reset_inputs:
            self.model.sampling.reset_sampling_params(formatted_params)
            self.model.sampling.reset_prompt_tokens(prompt_tokens)
            self.model.sampling.reset_output_state(output_tokens)
            self.model.sampling.seed_manager.reset_seed_from_slots_if_needed(formatted_params.seed, active_slots)
            self.model.sampling.seed_manager.align_seed_counters_to_positions(
                formatted_params.seed,
                active_slots,
                start_pos,
            )
        self.model.sampling.seed_manager.get_new_values(active_slots)

        if enable_trace and self._trace is None and id(cache) not in self._vllm_trace_warmed_cache_ids:
            tt_out = self._run_untraced_decode_sample_for_current_step(
                tokens=tokens,
                start_pos=start_pos,
                cache=cache,
                max_batch=max_batch,
            )
        elif enable_trace and self._trace is not None and self._trace.cache is cache:
            self._reset_decode_trace_inputs(tokens=tokens, start_pos=start_pos, max_batch=max_batch)
            ttnn.execute_trace(self.mesh_device, self._trace.trace_id, cq_id=0, blocking=False)
            self._trace.generated += 1
            tt_out = self._trace.token_output
        elif enable_trace:
            tt_out = self._capture_decode_trace_for_current_step(
                tokens=tokens,
                start_pos=start_pos,
                cache=cache,
                max_batch=max_batch,
            )
        else:
            token_buffer = self._sample_token_buffer(0, width=max_batch)
            tt_out = self.model.decode_forward(
                tokens,
                start_pos,
                page_table=cache.page_table,
                kv_cache=cache,
                sample_on_device=True,
                tt_out_tok=token_buffer,
            )
            if isinstance(tt_out, tuple):
                tt_out = tt_out[0]

        if not read_from_device:
            return tt_out
        host = self.read_decode_output(tt_out, async_read=False)
        return self.process_decode_output_host(host, is_tokens=True)

    def read_decode_output(self, tt_out, async_read: bool = False):
        if tt_out is None:
            return None, []
        if isinstance(tt_out, tuple):
            tokens, log_probs = tt_out
            tokens_cpu, token_events = self.read_decode_output(tokens, async_read=async_read)
            log_probs_cpu, log_prob_events = self.read_decode_output(log_probs, async_read=async_read)
            return (tokens_cpu, log_probs_cpu), token_events + log_prob_events
        if isinstance(tt_out, torch.Tensor):
            return tt_out, []
        if async_read:
            host = tt_out.cpu(blocking=False, cq_id=0)
            return host, [ttnn.record_event(self.mesh_device, 0)]
        return tt_out.cpu(), []

    def process_decode_output_host(self, tt_out, is_tokens: bool = True):
        if isinstance(tt_out, tuple) and len(tt_out) == 2 and isinstance(tt_out[1], list):
            tt_out = tt_out[0]
        if tt_out is None:
            return None
        if isinstance(tt_out, tuple):
            tokens, log_probs = tt_out
            return self.process_decode_output_host(tokens, is_tokens=True), self.process_decode_output_host(
                log_probs,
                is_tokens=False,
            )
        if isinstance(tt_out, torch.Tensor):
            host = tt_out.reshape(-1)
        elif not ttnn.is_tensor_storage_on_device(tt_out):
            host = self._host_ttnn_to_torch(tt_out).reshape(-1)
        elif is_tokens:
            host = self._read_token_buffer(tt_out)
        else:
            host = self._read_log_probs(tt_out)
        return host.to(torch.int32) if is_tokens else host.float()

    def _make_decode_trace(self, trace_id, cache, token_input, token_output, current_pos):
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import _DecodeTrace

        return _DecodeTrace(
            trace_id=trace_id,
            cache=cache,
            token_input=token_input,
            token_output=token_output,
            current_pos=current_pos,
            prompt_len=0,
        )

    def _decode_step_tensors(
        self,
        *,
        tokens: torch.Tensor,
        start_pos,
        max_batch: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        token_values = torch.zeros((max_batch,), dtype=torch.int32)
        flat_tokens = tokens.reshape(-1).to(torch.int32)
        token_values[: min(max_batch, flat_tokens.numel())] = flat_tokens[:max_batch]
        pos_values = torch.full((max_batch,), -1, dtype=torch.int32)
        flat_pos = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        pos_values[: min(max_batch, flat_pos.numel())] = flat_pos[:max_batch]
        return (
            self._active_token_buffer(token_values, on_host=False),
            self._sample_token_buffer(0, width=max_batch),
            self.model._positions_to_tt(pos_values),
        )

    def _prepare_serving_cache(
        self,
        kv_cache,
        page_table,
        *,
        force: bool,
        page_table_slots: list[int] | None = None,
    ) -> QwenFullModelCache:
        cache = kv_cache or self.cache
        if cache is None:
            raise RuntimeError("vLLM did not provide a KV cache")
        cache_changed = cache is not self.cache
        self.cache = cache
        host = self._normalize_page_table(page_table, cache)
        if page_table_slots is not None:
            host = self._scatter_page_table_to_slots(host, cache, page_table_slots)
        if (
            cache_changed
            or force
            or self._last_page_table_host is None
            or not torch.equal(host, self._last_page_table_host)
        ):
            self._release_decode_trace()
            self._vllm_trace_warmed_cache_ids.discard(id(cache))
            host_tt = ttnn.from_torch(
                host,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host_tt, cache.page_table)
            if tuple(cache.page_table_host.shape) == tuple(host.shape):
                cache.page_table_host.copy_(host)
            self._last_page_table_host = host.clone()
        return cache

    @staticmethod
    def _slot_remap_moves_state(slot_remap) -> bool:
        if slot_remap is None:
            return False
        remap = torch.as_tensor(slot_remap, dtype=torch.int64).reshape(-1)
        if remap.numel() == 0:
            return False
        identity = torch.arange(remap.numel(), dtype=torch.int64)
        active = remap >= 0
        return bool(torch.any(active & (remap != identity)).item())

    @staticmethod
    def _host_ttnn_to_torch(tensor: ttnn.Tensor) -> torch.Tensor:
        try:
            shards = ttnn.get_device_tensors(tensor)
        except RuntimeError:
            return ttnn.to_torch(tensor)
        return ttnn.to_torch(shards[0])

    def _capture_decode_trace_for_current_step(
        self,
        *,
        tokens: torch.Tensor,
        start_pos,
        cache: QwenFullModelCache,
        max_batch: int,
    ) -> ttnn.Tensor:
        token_input, token_output, current_pos = self._decode_step_tensors(
            tokens=tokens,
            start_pos=start_pos,
            max_batch=max_batch,
        )
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_sample_body(
            token_input=token_input,
            token_output=token_output,
            current_pos=current_pos,
            cache=cache,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self._trace = self._make_decode_trace(trace_id, cache, token_input, token_output, current_pos)
        self._reset_decode_trace_inputs(tokens=tokens, start_pos=start_pos, max_batch=max_batch)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        self._trace.generated = 1
        return token_output

    def _run_untraced_decode_sample_for_current_step(
        self,
        *,
        tokens: torch.Tensor,
        start_pos,
        cache: QwenFullModelCache,
        max_batch: int,
    ) -> ttnn.Tensor:
        token_input, token_output, current_pos = self._decode_step_tensors(
            tokens=tokens,
            start_pos=start_pos,
            max_batch=max_batch,
        )
        self._decode_sample_body(
            token_input=token_input,
            token_output=token_output,
            current_pos=current_pos,
            cache=cache,
        )
        ttnn.synchronize_device(self.mesh_device)
        self._vllm_trace_warmed_cache_ids.add(id(cache))
        return token_output

    def _warmup_decode_for_trace(self, *, width: int) -> None:
        if self._vllm_decode_warmed or self.model.sampling is None:
            return
        max_batch = min(
            nearest_32(int(width)),
            int(self.model.max_batch_size),
            int(self.model.sampling.tt_sampling.max_batch_size),
        )
        if max_batch <= 0:
            return

        old_cache = self.cache
        compile_cache = None
        token_input = None
        token_output = None
        current_pos = None
        try:
            compile_cache = self.model.allocate_cache(
                max_batch_size=max_batch,
                max_seq_len=max(1, int(self.model.block_size)),
            )
            token_values = torch.zeros((max_batch,), dtype=torch.int32)
            positions = torch.zeros((max_batch,), dtype=torch.int32)
            self._reset_sampling_state()
            token_input = self._active_token_buffer(token_values, on_host=False)
            token_output = self._sample_token_buffer(0, width=max_batch)
            current_pos = self.model._positions_to_tt(positions)
            self._decode_sample_body(
                token_input=token_input,
                token_output=token_output,
                current_pos=current_pos,
                cache=compile_cache,
            )
            ttnn.synchronize_device(self.mesh_device)
            self._reset_sampling_state()
            self._vllm_decode_warmed = True
        finally:
            self.cache = old_cache
            del compile_cache, token_input, token_output, current_pos
            gc.collect()

    def _normalize_page_table(self, page_table, cache: QwenFullModelCache) -> torch.Tensor:
        host = page_table.to(torch.int32).contiguous()
        host = torch.where(host > 0, host - 1, host)
        target_rows, target_cols = tuple(cache.page_table_host.shape)
        if host.shape[0] > target_rows or host.shape[1] > target_cols:
            raise ValueError(f"page_table shape {tuple(host.shape)} exceeds cache table {(target_rows, target_cols)}")
        if host.shape != (target_rows, target_cols):
            padded = torch.zeros((target_rows, target_cols), dtype=torch.int32)
            padded[: host.shape[0], : host.shape[1]] = host
            host = padded
        return host

    def _scatter_page_table_to_slots(
        self,
        page_table_host: torch.Tensor,
        cache: QwenFullModelCache,
        slots: list[int],
    ) -> torch.Tensor:
        target_rows, target_cols = tuple(cache.page_table_host.shape)
        scattered = cache.page_table_host.clone()
        if tuple(scattered.shape) != (target_rows, target_cols):
            scattered = torch.zeros((target_rows, target_cols), dtype=torch.int32)
        if len(slots) > int(page_table_host.shape[0]):
            raise ValueError(f"page table has {page_table_host.shape[0]} rows for {len(slots)} scheduled slots")
        width = min(target_cols, int(page_table_host.shape[1]))
        for request_idx, slot in enumerate(slots):
            slot = int(slot)
            if slot < 0 or slot >= target_rows:
                raise ValueError(f"page_table slot {slot} exceeds cache table rows {target_rows}")
            scattered[slot, :] = 0
            scattered[slot, :width] = page_table_host[request_idx, :width]
        return scattered

    def _prefill_host_logits(
        self,
        tokens: torch.Tensor,
        *,
        cache: QwenFullModelCache,
        prompt_lens: list[int],
        empty_slots: list[int],
    ) -> torch.Tensor:
        rows = []
        for request_idx, (prompt_len, slot) in enumerate(zip(prompt_lens, empty_slots, strict=True)):
            if prompt_len <= 0:
                rows.append(torch.zeros((1, self.model.vocab_size), dtype=torch.float32))
                continue
            row = self.model.prefill_user(
                tokens[request_idx : request_idx + 1, :prompt_len],
                cache=cache,
                user_id=int(slot),
                page_table_user_id=int(slot),
                return_all_logits=False,
            )
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _decode_host_logits(self, tokens: torch.Tensor, start_pos, *, cache: QwenFullModelCache) -> torch.Tensor:
        logits = self.model.decode_forward(tokens, start_pos, page_table=cache.page_table, kv_cache=cache)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        return logits

    def _reset_decode_trace_inputs(self, *, tokens: torch.Tensor, start_pos, max_batch: int) -> None:
        if self._trace is None:
            return
        token_values = torch.zeros((max_batch,), dtype=torch.int32)
        flat_tokens = tokens.reshape(-1).to(torch.int32)
        token_values[: min(max_batch, flat_tokens.numel())] = flat_tokens[:max_batch]
        pos_values = torch.full((max_batch,), -1, dtype=torch.int32)
        flat_pos = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        pos_values[: min(max_batch, flat_pos.numel())] = flat_pos[:max_batch]
        ttnn.copy_host_to_device_tensor(self._active_token_buffer(token_values, on_host=True), self._trace.token_input)
        ttnn.copy_host_to_device_tensor(self._positions_host(pos_values), self._trace.current_pos)

    def _active_decode_slots(self, start_pos, *, max_batch: int) -> list[int]:
        positions = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)
        active = [idx for idx, pos in enumerate(positions[:max_batch].tolist()) if int(pos) >= 0]
        return active or []


Qwen3_5MoeForCausalLM = Qwen3_5MoeForConditionalGeneration

__all__ = ["Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"]
