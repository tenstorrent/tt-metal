# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM TT plugin adapter for the Llama-3.2-1B-Instruct autoport."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.generator import Llama32Generator
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.model import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_PAGE_BLOCK_SIZE,
    MODEL_ID,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import dtype_to_config_name


class Llama32ForCausalLM(Llama32Generator):
    """Shared vLLM path around the datatype-selected full-model generator."""

    model_capabilities = {
        "supports_async_decode": True,
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "supports_host_sampling_fallback": False,
        "supports_device_logprobs": False,
        "tt_async_decode_allows_overlap": False,
    }

    def _debug_vllm_boundary(self, event: str, **payload: Any) -> None:
        path = os.environ.get("TT_LLAMA32_VLLM_DEBUG_PATH")
        if not path:
            return
        count = int(getattr(self, "_debug_vllm_boundary_count", 0))
        limit = int(os.environ.get("TT_LLAMA32_VLLM_DEBUG_LIMIT", "96"))
        if count >= limit:
            return
        self._debug_vllm_boundary_count = count + 1

        def compact(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                flat = value.detach().cpu().reshape(-1)
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "values": flat[:16].tolist(),
                }
            if isinstance(value, (list, tuple)):
                return [compact(item) for item in list(value)[:16]]
            if isinstance(value, dict):
                return {str(key): compact(item) for key, item in value.items()}
            if hasattr(value, "tolist"):
                return compact(value.tolist())
            return value

        record = {"event": event, "index": count, **compact(payload)}
        debug_path = Path(path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _debug_sampling_params(self, sampling_params: Any | None) -> dict[str, Any] | None:
        if sampling_params is None:
            return None
        out: dict[str, Any] = {}
        for field in ("temperature", "top_k", "top_p", "seed", "enable_log_probs", "num_logprobs"):
            if not hasattr(sampling_params, field):
                continue
            value = getattr(sampling_params, field)
            if isinstance(value, torch.Tensor):
                out[field] = value.detach().cpu().reshape(-1)[:16].tolist()
            elif hasattr(value, "tolist"):
                flat = value.reshape(-1) if hasattr(value, "reshape") else value
                out[field] = flat[:16].tolist() if hasattr(flat, "__getitem__") else value.tolist()
            elif isinstance(value, list):
                out[field] = value[:16]
            else:
                out[field] = value
        return out

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ) -> "Llama32ForCausalLM":
        if optimizations is not None:
            raise ValueError("custom vLLM optimizations are not supported by this autoport adapter")
        if int(tt_data_parallel) != 1:
            raise ValueError(f"{MODEL_ID} autoport adapter currently supports tt_data_parallel=1, got {tt_data_parallel}")
        if int(max_batch_size) != 1:
            raise ValueError(
                f"{MODEL_ID} autoport vLLM adapter currently supports batch-1 serving only; "
                f"run vLLM with --max_num_seqs 1, got {max_batch_size}"
            )
        if int(max_seq_len) > DEFAULT_MAX_SEQ_LEN:
            raise ValueError(
                f"{MODEL_ID} autoport adapter supports max_seq_len <= {DEFAULT_MAX_SEQ_LEN}; "
                f"run vLLM with --max_model_len {DEFAULT_MAX_SEQ_LEN}"
            )

        model_dir = Path(__file__).resolve().parents[1]
        hf_model_id = getattr(hf_config, "_name_or_path", None) or MODEL_ID
        return cls(
            model_dir=model_dir,
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            max_seq_len=int(max_seq_len),
            max_batch_size=DEFAULT_MAX_BATCH_SIZE,
            page_block_size=DEFAULT_PAGE_BLOCK_SIZE,
            num_layers=n_layers,
            cache_path=model_dir / ".ttnn_cache",
            use_vllm_paged_kv_cache=True,
        )

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        model_name: str | None = None,
        num_devices: int | None = None,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
    ) -> int:
        del model_name, num_devices, tt_data_parallel
        model_len = int(max_model_len or DEFAULT_MAX_SEQ_LEN)
        num_seqs = int(max_num_seqs or 1)
        return min(DEFAULT_MAX_BATCH_SIZE * DEFAULT_MAX_SEQ_LEN, model_len * num_seqs)

    @property
    def cache_path(self) -> Path:
        return self.model_dir / ".ttnn_cache"

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, int, int, int],
        dtype: torch.dtype,
        num_layers: int,
    ) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        del dtype
        if int(num_layers) != self.model.n_layers:
            raise ValueError(f"vLLM requested {num_layers} KV-cache layers, model has {self.model.n_layers}")

        expected_local_kv_heads = int(self.model.hf_config.num_key_value_heads) // self.mesh_device.get_num_devices()
        if int(kv_cache_shape[1]) != expected_local_kv_heads:
            raise ValueError(
                f"KV cache shape {kv_cache_shape} has {kv_cache_shape[1]} local KV heads, "
                f"expected {expected_local_kv_heads}"
            )
        if int(kv_cache_shape[2]) != self.page_block_size:
            raise ValueError(
                f"KV cache block size {kv_cache_shape[2]} does not match page_block_size={self.page_block_size}"
            )
        if int(kv_cache_shape[3]) != int(self.model.hf_config.head_dim):
            raise ValueError(f"KV cache head_dim {kv_cache_shape[3]} does not match {self.model.hf_config.head_dim}")

        kv_dtype = self.precision_policy.kv_cache_dtype
        cache_source = torch.zeros(kv_cache_shape, dtype=torch.bfloat16)
        shape_tag = "x".join(str(dim) for dim in kv_cache_shape)
        dtype_tag = dtype_to_config_name(kv_dtype)
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for layer_idx in range(num_layers):
            key_cache = ttnn.as_tensor(
                cache_source,
                device=self.mesh_device,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.cache_path / f"vllm_layer{layer_idx}_k_cache_{shape_tag}_{dtype_tag}",
            )
            value_cache = ttnn.as_tensor(
                cache_source,
                device=self.mesh_device,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.cache_path / f"vllm_layer{layer_idx}_v_cache_{shape_tag}_{dtype_tag}",
            )
            kv_cache.append((key_cache, value_cache))

        self.bind_external_kv_cache(kv_cache)
        return kv_cache

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: list[int],
        sampling_params: Any | None = None,
        empty_slots: list[int] | None = None,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if sampling_params is None:
            raise RuntimeError("vLLM autoport path requires on-device sampling; host logits fallback is disabled")
        self._debug_vllm_boundary(
            "prefill_input",
            tokens=tokens,
            prompt_lens=prompt_lens,
            page_table=page_table,
            sampling_params=self._debug_sampling_params(sampling_params),
            empty_slots=empty_slots,
        )
        out = self.prefill_token_out_host(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
            empty_slots=empty_slots,
        )
        self._debug_vllm_boundary("prefill_output", output=out[0] if isinstance(out, tuple) else out)
        return out

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        sampling_params: Any | None = None,
        read_from_device: bool = False,
        reset_batch: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        if sampling_params is None:
            raise RuntimeError("vLLM autoport path requires on-device sampling; host logits fallback is disabled")
        self._debug_vllm_boundary(
            "decode_input",
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=self._debug_sampling_params(sampling_params),
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
        )
        tt_out = self.decode_token_out_device_for_vllm(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
        )
        if read_from_device:
            return self.process_decode_output_host(tt_out, is_tokens=True)
        return tt_out

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> tuple[Any, list[Any]]:
        if not async_read:
            self._debug_vllm_boundary("read_decode_output", async_read=False)
            return tt_out, []
        self.counters.async_decode_reads += 1
        self._debug_vllm_boundary("read_decode_output", async_read=True)
        cpu_out = self._cpu_async(tt_out)
        return cpu_out, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = True) -> Any:
        if not is_tokens:
            raise RuntimeError("host logits processing is disabled for the autoport vLLM adapter")
        self.counters.host_decode_process_calls += 1
        if isinstance(tt_out, tuple):
            tokens = self._tt_or_torch_to_flat_torch(tt_out[0], dtype=torch.int32)
            log_probs = None if tt_out[1] is None else self._tt_or_torch_to_flat_torch(tt_out[1], dtype=torch.float32)
            self._debug_vllm_boundary("decode_output", output=tokens, log_probs=log_probs)
            return tokens if log_probs is None else (tokens, log_probs)
        tokens = self._tt_or_torch_to_flat_torch(tt_out, dtype=torch.int32)
        self._debug_vllm_boundary("decode_output", output=tokens)
        return tokens

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,
        enable_trace: bool,
        can_sample_on_device: bool,
    ) -> None:
        del enable_trace
        if getattr(self, "already_warmed_up_prefill", False):
            return
        if not can_sample_on_device:
            raise RuntimeError("vLLM autoport warmup requires sample_on_device_mode='all'")
        page_table = self._warmup_page_table(kv_cache, batch_size=1)
        tokens = torch.zeros(1, 128, dtype=torch.long)
        self.prefill_token_out_host(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[128],
            sampling_params=self._make_sampling_params(top_k=1, top_p=0.0, temperature=1.0),
            empty_slots=[0],
        )
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,
        max_batch_size: int,
        num_blocks: int,
        enable_trace: bool,
        can_sample_on_device: bool,
    ) -> None:
        if not can_sample_on_device:
            raise RuntimeError("vLLM autoport warmup requires on-device decode sampling")
        if int(max_batch_size) > DEFAULT_MAX_BATCH_SIZE:
            raise ValueError(f"warmup max_batch_size {max_batch_size} exceeds {DEFAULT_MAX_BATCH_SIZE}")
        tokens = torch.zeros(int(max_batch_size), 1, dtype=torch.long)
        start_pos = torch.zeros(int(max_batch_size), dtype=torch.long)
        page_table = torch.zeros(int(max_batch_size), int(num_blocks), dtype=torch.int32)
        if int(num_blocks) > 0:
            page_table[:] = torch.arange(int(num_blocks), dtype=torch.int32)
        self.decode_token_out_device_for_vllm(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            sampling_params=self._make_sampling_params(top_k=1, top_p=0.0, temperature=1.0),
            reset_batch=True,
        )
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1

    def _warmup_page_table(self, kv_cache: Any, *, batch_size: int) -> torch.Tensor:
        num_blocks = int(kv_cache[0][0].shape[0]) if kv_cache else math.ceil(self.max_seq_len / self.page_block_size)
        blocks_per_req = min(math.ceil(self.max_seq_len / self.page_block_size), num_blocks)
        page_table = torch.zeros(batch_size, blocks_per_req, dtype=torch.int32)
        if blocks_per_req > 0:
            page_table[:] = torch.arange(blocks_per_req, dtype=torch.int32)
        return page_table

    def _cpu_async(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(None if item is None else self._cpu_async(item) for item in value)
        if isinstance(value, torch.Tensor):
            return value
        return value.cpu(blocking=False, cq_id=0)

    def _tt_or_torch_to_flat_torch(self, value: Any, *, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.reshape(-1)[: self.max_batch_size].to(dtype)
        try:
            tensors = ttnn.get_device_tensors(value)
            if tensors:
                host = ttnn.to_torch(tensors[0])
            else:
                host = ttnn.to_torch(value)
        except Exception:
            host = ttnn.to_torch(value)
        return host.reshape(-1)[: self.max_batch_size].to(dtype)


__all__ = ["Llama32ForCausalLM"]
