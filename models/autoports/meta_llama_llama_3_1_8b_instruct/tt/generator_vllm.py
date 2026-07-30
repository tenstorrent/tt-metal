# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the Llama 3.1 8B Instruct autoport.

The adapter is intentionally thin: it translates vLLM's runner protocol to the
readiness generator's low-level prefill/decode methods and preserves vLLM-owned
attention KV-cache ownership.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import (
    Llama31_8B_InstructGenerator,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.model import (
    MODEL_DIR,
    MODEL_ID,
    TARGET_MESH_SHAPE,
    Llama31_8B_InstructFullModel,
)
from models.common.sampling import SamplingParams, format_sampling_params
from models.common.sampling.tt_log_probs import LogProbsResult
from models.common.tensor_utils import TILE_SIZE


class Llama31_8B_InstructForCausalLM(Llama31_8B_InstructGenerator):
    """TT vLLM bridge for ``meta-llama/Llama-3.1-8B-Instruct``."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        # Async split submit/read is supported. Scheduler overlap is a
        # separate proof-backed capability and remains disabled for this
        # adapter because model inputs are still refreshed from vLLM host
        # scheduler state at trace/layout changes.
        "tt_async_decode_allows_overlap": False,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vllm_kv_cache: Any | None = None
        self._expected_next_decode_token: int | None = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: ttnn.MeshDevice,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "Llama31_8B_InstructForCausalLM":
        if optimizations is not None:
            raise ValueError("Custom vLLM optimizations are not supported by this autoport adapter")
        if tt_data_parallel != 1:
            raise ValueError(f"Llama 3.1 8B autoport vLLM bridge supports DP=1 only, got {tt_data_parallel}")
        if kwargs:
            raise TypeError(f"Unexpected initialize_vllm_model kwargs: {sorted(kwargs)}")
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(f"Llama 3.1 8B autoport requires T3K mesh shape {TARGET_MESH_SHAPE}, got {mesh_device.shape}")

        model_id = getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", None) or MODEL_ID
        internal_max_seq_len = max(int(max_seq_len), int(getattr(hf_config, "max_position_embeddings", max_seq_len)))
        model = Llama31_8B_InstructFullModel.from_pretrained(
            mesh_device=mesh_device,
            model_id=model_id,
            max_batch_size=int(max_batch_size),
            max_seq_len=internal_max_seq_len,
            page_block_size=64,
            cache_dir=MODEL_DIR / "tt_cache" / "vllm",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        return cls(model_dir=MODEL_DIR, mesh_device=mesh_device, model=model, tokenizer=tokenizer)

    @property
    def cache_path(self) -> Path:
        return self.model_dir / "tt_cache" / "vllm"

    def allocate_kv_cache(self, kv_cache_shape: tuple[int, int, int, int], dtype: torch.dtype, num_layers: int):
        del dtype
        if int(num_layers) != self.model.n_layers:
            raise ValueError(f"Expected {self.model.n_layers} KV-cache layers, got {num_layers}")

        shape = tuple(int(dim) for dim in kv_cache_shape)
        expected_local_kv_heads = self.model.hf_config.num_key_value_heads // self.mesh_device.get_num_devices()
        expected_head_dim = int(self.model.hf_config.head_dim)
        if shape[1] != expected_local_kv_heads or shape[3] != expected_head_dim:
            raise ValueError(
                f"Unexpected vLLM KV-cache shape {shape}; expected local_kv_heads={expected_local_kv_heads}, "
                f"head_dim={expected_head_dim}"
            )

        self.cache_path.mkdir(parents=True, exist_ok=True)
        host_cache = torch.zeros(shape, dtype=torch.bfloat16)
        kv_cache = []
        for layer_idx in range(num_layers):
            layer_cache = []
            for name in ("k", "v"):
                layer_cache.append(
                    ttnn.as_tensor(
                        host_cache,
                        device=self.mesh_device,
                        dtype=self.model.policy.kv_cache_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        cache_file_name=self.cache_path / f"empty_{name}_cache_layer_{layer_idx}_{shape}",
                    )
                )
            kv_cache.append(tuple(layer_cache))
        self._vllm_kv_cache = kv_cache
        return kv_cache

    def _require_batch1(self, tokens: torch.Tensor, *, context: str) -> None:
        if tokens.shape[0] != 1:
            raise NotImplementedError(f"{context} currently supports max-num-seqs=1, got batch {tokens.shape[0]}")

    def _format_sampling_params(self, sampling_params: Any | None) -> Any:
        if sampling_params is None:
            raise RuntimeError("vLLM serving requires on-device sampling params")
        return format_sampling_params(sampling_params, self.sampling.tt_sampling.max_batch_size)

    def _host_vector(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.reshape(-1).cpu()
        if isinstance(value, ttnn.Tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).reshape(-1).cpu()
        raise TypeError(f"Unsupported tensor value {type(value)!r}")

    def _deallocate_device_value(self, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, ttnn.Tensor):
            log_probs_calculator = self.sampling.tt_sampling.log_probs_calculator
            persistent_sampler_tensors = (
                getattr(log_probs_calculator, "output_tensor", None),
                getattr(log_probs_calculator, "topk_logprobs_output", None),
                getattr(log_probs_calculator, "topk_indices_output", None),
            )
            if any(value is tensor for tensor in persistent_sampler_tensors):
                return
            value.deallocate(True)
            return
        if isinstance(value, LogProbsResult):
            self._deallocate_device_value(value.topk_logprobs)
            self._deallocate_device_value(value.topk_indices)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                self._deallocate_device_value(item)

    def _read_token_tuple(self, tt_out: Any, *, row: int | None = None) -> torch.Tensor | tuple[torch.Tensor, Any]:
        tt_tokens = tt_out[0] if isinstance(tt_out, tuple) else tt_out
        tt_log_probs = tt_out[1] if isinstance(tt_out, tuple) else None

        tokens = self._host_vector(tt_tokens).to(torch.int64)
        if row is not None:
            tokens = tokens[row : row + 1]

        if tt_log_probs is None:
            return tokens
        if isinstance(tt_log_probs, LogProbsResult):
            tt_log_probs = tt_log_probs.cpu(blocking=True)
            if row is not None:
                tt_log_probs = tt_log_probs.extract_user(row)
            return tokens, tt_log_probs

        log_probs = self._host_vector(tt_log_probs).float()
        if row is not None:
            log_probs = log_probs[row : row + 1]
        return tokens, log_probs

    def _prepare_prefill_sampling(self, sampling_params: Any, tokens: torch.Tensor) -> Any:
        formatted = self._format_sampling_params(sampling_params)
        self.sampling.apply_prefill_state(
            sampling_params=formatted,
            prompt_tokens=tokens.to(torch.int64),
            empty_slots=[0],
            replicate_seeds=False,
        )
        return formatted

    def _remember_decode_feedback(self, result: Any) -> None:
        tokens = result[0] if isinstance(result, tuple) else result
        if isinstance(tokens, torch.Tensor) and tokens.numel() > 0:
            self._expected_next_decode_token = int(tokens.reshape(-1)[0].item())

    def _prepare_decode_sampling(
        self,
        sampling_params: Any,
        *,
        start_pos: torch.Tensor,
        reset_batch: bool,
        prompt_tokens: torch.Tensor | None,
        output_tokens: torch.Tensor | None,
        slot_remap: torch.Tensor | None,
    ) -> Any:
        formatted = self._format_sampling_params(sampling_params)
        if slot_remap is not None:
            self.sampling.seed_manager.apply_slot_remap(slot_remap)
        self.sampling.apply_decode_state(
            [formatted],
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        self.sampling.seed_manager.reset_seed_from_slots_if_needed(formatted.seed, [0])
        self.sampling.seed_manager.align_seed_counters_to_positions(formatted.seed, [0], start_pos, offset=0)
        self.sampling.seed_manager.get_new_values([0])
        return formatted

    def _release_decode_state_before_prefill(self) -> None:
        if self._decode_trace.trace_id is None and self._decode_trace.tokens is None:
            self.sampling.reset_trace()
            self.model.release_decode_persistent_buffers()
            self._expected_next_decode_token = None
            return
        ttnn.synchronize_device(self.mesh_device)
        self.sampling.reset_trace()
        ttnn.synchronize_device(self.mesh_device)
        self._release_decode_trace()
        ttnn.synchronize_device(self.mesh_device)
        self.model.release_decode_persistent_buffers()
        ttnn.synchronize_device(self.mesh_device)
        self._prev_page_table = None
        self._decode_host_position = None
        self._expected_next_decode_token = None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._expected_next_decode_token = None
        return super().reset(*args, **kwargs)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: list[int] | torch.Tensor,
        enable_trace: bool = False,
        sampling_params: Any | None = None,
        start_pos: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        del enable_trace, kwargs
        self._require_batch1(tokens, context="prefill")
        if start_pos is not None and any(int(pos) != 0 for pos in torch.as_tensor(start_pos).reshape(-1).tolist()):
            raise AssertionError(f"Prefix caching is not supported, got start_pos={start_pos}")
        self._release_decode_state_before_prefill()

        prompt_lens_list = [int(x) for x in torch.as_tensor(prompt_lens).reshape(-1).tolist()]
        if sampling_params is None:
            raise RuntimeError("vLLM prefill requires on-device sampling params")

        tt_logits, _real_len, last_token_idx = self.prefill_forward_device(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens_list,
            return_all_logits=False,
        )
        assert isinstance(tt_logits, ttnn.Tensor)
        self._prepare_prefill_sampling(sampling_params, tokens)
        tt_out = self.sampling.sample(tt_logits, enable_trace=False)
        row = int(last_token_idx) % TILE_SIZE
        result = self._read_token_tuple(tt_out, row=row)
        self._remember_decode_feedback(result)
        self._deallocate_device_value(tt_out)
        tt_logits.deallocate(True)
        return result

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: Any | None = None,
        reset_batch: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        del kwargs
        self._require_batch1(tokens, context="decode")
        if sampling_params is None:
            raise RuntimeError("vLLM decode requires on-device sampling params")
        if not enable_trace:
            raise RuntimeError("vLLM decode requires trace replay; use trace_mode=decode_only")

        self._prepare_decode_sampling(
            sampling_params,
            start_pos=start_pos,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
        )
        token_id = int(tokens.reshape(-1)[0].item())
        current_pos = int(start_pos.reshape(-1)[0].item())
        page_table_torch = self._coerce_page_table_torch(page_table)
        token_from_host = bool(
            reset_batch
            or self._decode_trace.trace_id is None
            or self._expected_next_decode_token is None
            or token_id != self._expected_next_decode_token
        )
        tt_out = self._decode_trace_sample(
            token_id,
            current_pos,
            page_table=page_table_torch,
            kv_cache=kv_cache,
            enable_trace=enable_trace,
            token_from_host=token_from_host,
            refresh_sampled_hidden=False,
            readback=False,
        )
        if not read_from_device:
            return tt_out
        return self.process_decode_output_host(self.read_decode_output(tt_out, async_read=False), is_tokens=True)

    def read_decode_output(self, tt_out: Any, async_read: bool = False):
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out

        read_event = None
        if isinstance(tt_out, tuple):
            tt_tokens, tt_log_probs = tt_out
            tt_tokens = tt_tokens.cpu(blocking=not async_read, cq_id=0)
            if isinstance(tt_log_probs, LogProbsResult):
                tt_log_probs = tt_log_probs.cpu(blocking=not async_read)
            elif tt_log_probs is not None:
                tt_log_probs = tt_log_probs.cpu(blocking=not async_read, cq_id=0)
            read_value = (tt_tokens, tt_log_probs)
        elif isinstance(tt_out, ttnn.Tensor):
            read_value = tt_out.cpu(blocking=not async_read, cq_id=0)
        else:
            raise TypeError(f"Unsupported decode output type {type(tt_out)!r}")

        if async_read:
            read_event = ttnn.record_event(self.mesh_device, 0)
            return read_value, [read_event]
        return read_value

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = True):
        if not is_tokens:
            raise RuntimeError("vLLM adapter only exposes sampled token outputs")
        result = self._read_token_tuple(tt_out)
        self._remember_decode_feedback(result)
        return result

    def warmup_model_prefill(self, kv_cache: Any, enable_trace: bool, can_sample_on_device: bool, **kwargs: Any) -> None:
        del kv_cache, enable_trace, can_sample_on_device, kwargs

    def _decode_warmup_sampling_params(self, can_sample_on_device: bool) -> list[SamplingParams]:
        if not can_sample_on_device:
            return []

        active = 1
        configs: list[SamplingParams] = []
        for penalties_on in (False, True):
            for log_probs_on in (False, True):
                penalty_kwargs = {}
                if penalties_on:
                    penalty_kwargs = {
                        "presence_penalty": [1.2] * active,
                        "frequency_penalty": [1.2] * active,
                        "repetition_penalty": [1.5] * active,
                    }
                configs.append(
                    SamplingParams(
                        temperature=[1.0] * active,
                        top_k=[10] * active,
                        top_p=[0.9] * active,
                        enable_log_probs=[log_probs_on] * active,
                        num_logprobs=[0] * active,
                        **penalty_kwargs,
                    )
                )

        configs.append(
            SamplingParams(
                temperature=[0.0] * active,
                top_k=[1] * active,
                top_p=[1.0] * active,
                enable_log_probs=[False] * active,
                num_logprobs=[0] * active,
            )
        )
        return configs

    def _precompile_decode_sampling(self, kv_cache: Any, num_blocks: int, can_sample_on_device: bool) -> None:
        del kv_cache, num_blocks
        if self._decode_sampling_precompiled or not can_sample_on_device:
            return

        batch_rows = self.sampling.tt_sampling.max_batch_size
        logits = ttnn.from_torch(
            torch.zeros((1, 1, batch_rows, self.model.padded_vocab_size), dtype=torch.bfloat16),
            device=self.mesh_device,
            dtype=self.model.full_model_config.lm_head_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, -1),
                mesh_shape=tuple(self.mesh_device.shape),
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_out_tok = ttnn.from_torch(
            torch.zeros((1, 1, 1, batch_rows), dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate_mapper(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(self.mesh_device)

        start_pos = torch.zeros(1, dtype=torch.int32)
        prompt_tokens = torch.zeros((1, 1), dtype=torch.int64)
        for sampling_params in self._decode_warmup_sampling_params(can_sample_on_device):
            self._prepare_decode_sampling(
                sampling_params,
                start_pos=start_pos,
                reset_batch=True,
                prompt_tokens=prompt_tokens,
                output_tokens=None,
                slot_remap=None,
            )
            self.sampling.sample(logits, enable_trace=False, tt_out_tok=tt_out_tok)
            ttnn.synchronize_device(self.mesh_device)

        logits.deallocate(True)
        tt_out_tok.deallocate(True)
        ttnn.synchronize_device(self.mesh_device)
        self.sampling.reset_trace()
        self._decode_sampling_precompiled = True

    def warmup_model_decode(
        self,
        kv_cache: Any,
        enable_trace: bool,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._require_batch1(torch.zeros(int(max_batch_size), 1, dtype=torch.int32), context="decode warmup")
        self._precompile_decode_sampling(kv_cache, num_blocks, can_sample_on_device)
        del enable_trace


__all__ = ["Llama31_8B_InstructForCausalLM"]
