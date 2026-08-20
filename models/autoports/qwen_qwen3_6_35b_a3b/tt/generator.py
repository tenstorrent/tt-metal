# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Readiness generator for the Qwen3.6-35B-A3B full TTNN model."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import MODEL_ID
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    QwenFullModel,
    QwenFullModelCache,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision_config import load_precision_config
from models.common.readiness_check.contract import Generator
from models.common.sampling import SamplingParams, format_sampling_params
from models.common.utility_functions import nearest_32

NextInputFn = Callable[[int, int], int]


@dataclass
class _DecodeTrace:
    trace_id: Any
    cache: QwenFullModelCache
    token_input: ttnn.Tensor
    token_output: ttnn.Tensor
    current_pos: ttnn.Tensor
    prompt_len: int
    generated: int = 0


class QwenReadinessGenerator(Generator):
    """Standard Metal readiness generator for full-model checks."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        mesh_device,
        model_id: str = MODEL_ID,
        local_files_only: bool = True,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int | None = None,
        prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
        host_sampling_compat: bool = False,
        model: QwenFullModel | None = None,
        precision_config: str | Path | dict[str, Any] | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.model_id = model_id
        self.precision_config, self.precision_config_source = load_precision_config(
            model_dir=self.model_dir,
            precision_config=precision_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        self.model = model or QwenFullModel.from_hf(
            mesh_device=mesh_device,
            model_id=model_id,
            local_files_only=local_files_only,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            precision_config=self.precision_config,
            precision_config_source=self.precision_config_source,
        )
        self.host_sampling_compat = host_sampling_compat
        self.cache: QwenFullModelCache | None = None
        self._trace: _DecodeTrace | None = None
        self.last_timings: dict[str, float] = {}
        self.last_trace_counters: dict[str, int | bool] = {}
        self._reset_sampling_state()

    def reset(self) -> None:
        self._release_decode_trace()
        self._trace = None
        self.cache = None
        self.last_trace_counters = {}
        self._reset_sampling_state()

    def _release_decode_trace(self) -> None:
        released_trace = False
        if self._trace is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._trace.trace_id)
                released_trace = True
            except Exception:
                pass
        if self.model.sampling is not None:
            self.model.sampling.reset_trace()
        if released_trace:
            ttnn.synchronize_device(self.mesh_device)
        self._trace = None

    def teardown(self) -> None:
        self.reset()

    def allocate_kv_cache(
        self,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_seq_len: int | None = None,
        page_table: torch.Tensor | ttnn.Tensor | None = None,
    ) -> QwenFullModelCache:
        return self.model.allocate_cache(max_batch_size=max_batch_size, max_seq_len=max_seq_len, page_table=page_table)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        cache = kv_cache or self.allocate_kv_cache(
            max_batch_size=tokens.shape[0],
            max_seq_len=max(max(prompt_lens), 1),
            page_table=page_table,
        )
        self.cache = cache
        return self.model.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=cache,
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
        **kwargs,
    ) -> torch.Tensor:
        cache = kv_cache or self.cache
        if cache is None:
            cache = self.allocate_kv_cache(max_batch_size=tokens.shape[0], page_table=page_table)
            self.cache = cache
        return self.model.decode_forward(tokens, start_pos, page_table=page_table, kv_cache=cache)

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: NextInputFn | None = None,
        enable_trace: bool = True,
        **kwargs,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return []
        self.reset()

        prompt_len = len(prompt_token_ids)
        if prompt_len <= 0:
            raise ValueError("generate requires at least one prompt token")
        required_context = prompt_len + max(max_new_tokens - 1, 0)
        if required_context > self.model.max_seq_len:
            raise ValueError(
                f"requested prompt/generation needs context {required_context}, "
                f"but the model supports {self.model.max_seq_len}"
            )
        max_seq_len = kwargs.get("max_seq_len") or max(required_context, self.model.block_size)
        traced_decode = max_new_tokens > 1 and not self.host_sampling_compat and enable_trace
        optimized_token_out = traced_decode and next_input is None
        if traced_decode:
            self._warmup_traced_decode(prompt_len=prompt_len, max_seq_len=max_seq_len)
        cache = self.allocate_kv_cache(max_batch_size=1, max_seq_len=max_seq_len)
        self.cache = cache

        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        t0 = time.perf_counter()
        if optimized_token_out:
            first_token = self._prefill_sample_on_device(prompt, cache)
        else:
            prefill_logits = self.model.prefill_forward(
                prompt,
                page_table=cache.page_table,
                kv_cache=cache,
                prompt_lens=[prompt_len],
                return_all_logits=False,
            )
            first_token = int(torch.argmax(prefill_logits[0, 0]).item())
        t1 = time.perf_counter()
        predictions = [first_token]
        forced = next_input(0, first_token) if next_input is not None else first_token

        if max_new_tokens > 1:
            if next_input is not None and traced_decode:
                self._generate_traced_teacher_forcing(
                    predictions=predictions,
                    first_input=forced,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    next_input=next_input,
                    cache=cache,
                )
            elif self.host_sampling_compat or not enable_trace:
                self._generate_host_sampling(
                    predictions=predictions,
                    first_input=forced,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    next_input=next_input,
                    cache=cache,
                )
            else:
                self._generate_traced_sampling(
                    predictions=predictions,
                    first_input=forced,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    cache=cache,
                )
        t2 = time.perf_counter()
        self.last_timings = {
            "ttft_s": t1 - t0,
            "decode_s": t2 - t1,
            "e2e_s": t2 - t0,
            "generated_tokens": float(len(predictions)),
            "decode_tokens": float(max(len(predictions) - 1, 0)),
        }
        return predictions

    def measure_token_out_no_readback(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        max_seq_len: int | None = None,
        validate_final_token: bool = True,
    ) -> dict[str, Any]:
        """Measure traced token-out replay without per-token host sync/readback.

        This is the serving-style hot loop: the captured graph owns token
        feedback and position advance on device, and the host enqueues replay
        commands back-to-back. The optional final token read is outside the
        steady-state loop and exists only to prove the loop produced a token.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            self.last_timings = {
                "ttft_s": 0.0,
                "trace_capture_s": 0.0,
                "decode_replay_s": 0.0,
                "decode_s": 0.0,
                "decode_s_including_capture": 0.0,
                "e2e_s": 0.0,
                "generated_tokens": 0.0,
                "decode_tokens": 0.0,
            }
            self.last_trace_counters = _no_readback_counters(0, validate_final_token=False)
            return {
                "prompt_len": len(prompt_token_ids),
                "max_new_tokens": 0,
                "first_token": None,
                "final_token": None,
                "trace_present": False,
                "trace_generated_steps": 0,
                "raw_timings": dict(self.last_timings),
                "host_boundary_counters": dict(self.last_trace_counters),
            }

        self.reset()
        prompt_len = len(prompt_token_ids)
        if prompt_len <= 0:
            raise ValueError("measure_token_out_no_readback requires at least one prompt token")
        required_context = prompt_len + max(max_new_tokens - 1, 0)
        if required_context > self.model.max_seq_len:
            raise ValueError(
                f"requested prompt/generation needs context {required_context}, "
                f"but the model supports {self.model.max_seq_len}"
            )
        resolved_max_seq_len = max_seq_len or max(
            required_context, self.model.prefill_chunk_size, self.model.block_size
        )
        self._warmup_traced_decode(prompt_len=prompt_len, max_seq_len=resolved_max_seq_len)
        cache = self.allocate_kv_cache(max_batch_size=1, max_seq_len=resolved_max_seq_len)
        self.cache = cache

        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        e2e_start = time.perf_counter()
        ttft_start = time.perf_counter()
        first_token = self._prefill_sample_on_device(prompt, cache)
        ttft_end = time.perf_counter()

        trace_capture_s = 0.0
        decode_replay_s = 0.0
        final_token = first_token
        decode_tokens = max_new_tokens - 1
        if decode_tokens > 0:
            capture_start = time.perf_counter()
            trace = self._capture_decode_trace(first_token, prompt_len, cache)
            trace_capture_s = time.perf_counter() - capture_start

            replay_start = time.perf_counter()
            for _ in range(decode_tokens):
                ttnn.execute_trace(self.mesh_device, trace.trace_id, cq_id=0, blocking=False)
                trace.generated += 1
            ttnn.synchronize_device(self.mesh_device)
            decode_replay_s = time.perf_counter() - replay_start
            if validate_final_token:
                final_token = self._read_active_token(trace.token_input)

        e2e_s = time.perf_counter() - e2e_start
        self.last_timings = {
            "ttft_s": ttft_end - ttft_start,
            "trace_capture_s": trace_capture_s,
            "decode_replay_s": decode_replay_s,
            "decode_s": decode_replay_s,
            "decode_s_including_capture": trace_capture_s + decode_replay_s,
            "e2e_s": e2e_s,
            "generated_tokens": float(max_new_tokens),
            "decode_tokens": float(decode_tokens),
        }
        self.last_trace_counters = _no_readback_counters(decode_tokens, validate_final_token=validate_final_token)
        return {
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
            "first_token": int(first_token),
            "final_token": int(final_token) if final_token is not None else None,
            "trace_present": self._trace is not None,
            "trace_generated_steps": self._trace.generated if self._trace is not None else 0,
            "position_end_expected_exclusive": prompt_len + max_new_tokens,
            "raw_timings": dict(self.last_timings),
            "host_boundary_counters": dict(self.last_trace_counters),
        }

    def _generate_host_sampling(
        self,
        *,
        predictions: list[int],
        first_input: int,
        prompt_len: int,
        max_new_tokens: int,
        next_input: NextInputFn | None,
        cache: QwenFullModelCache,
    ) -> None:
        token = int(first_input)
        for step in range(1, max_new_tokens):
            logits = self.model.decode_forward(
                torch.tensor([[token]], dtype=torch.long),
                torch.tensor([prompt_len + step - 1], dtype=torch.int32),
                page_table=cache.page_table,
                kv_cache=cache,
            )
            pred = int(torch.argmax(logits[0]).item())
            predictions.append(pred)
            token = int(next_input(step, pred)) if next_input is not None else pred

    def _generate_traced_sampling(
        self,
        *,
        predictions: list[int],
        first_input: int,
        prompt_len: int,
        max_new_tokens: int,
        cache: QwenFullModelCache,
    ) -> None:
        trace = self._capture_decode_trace(first_input, prompt_len, cache)
        per_token_syncs = 0
        per_token_readbacks = 0
        while len(predictions) < max_new_tokens:
            ttnn.execute_trace(self.mesh_device, trace.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            per_token_syncs += 1
            predictions.append(self._read_active_token(trace.token_input))
            per_token_readbacks += 1
            trace.generated += 1
        self.last_trace_counters = {
            "trace_replays": int(trace.generated),
            "execute_trace_blocking": False,
            "steady_state_token_refreshes": 0,
            "steady_state_position_refreshes": 0,
            "steady_state_rope_refreshes": 0,
            "steady_state_page_table_refreshes": 0,
            "steady_state_synchronizations": per_token_syncs,
            "steady_state_token_readbacks": per_token_readbacks,
            "terminal_validation_synchronizations": 0,
            "terminal_validation_token_readbacks": 0,
            "host_sampling": False,
            "full_logits_readbacks": 0,
        }

    def _generate_traced_teacher_forcing(
        self,
        *,
        predictions: list[int],
        first_input: int,
        prompt_len: int,
        max_new_tokens: int,
        next_input: NextInputFn,
        cache: QwenFullModelCache,
    ) -> None:
        trace = self._capture_decode_trace(first_input, prompt_len, cache)
        per_token_syncs = 0
        per_token_readbacks = 0
        token_refreshes = 0
        for step in range(1, max_new_tokens):
            ttnn.execute_trace(self.mesh_device, trace.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            per_token_syncs += 1
            pred = self._read_active_token(trace.token_input)
            per_token_readbacks += 1
            predictions.append(pred)
            trace.generated += 1
            forced = next_input(step, pred)
            if step + 1 < max_new_tokens:
                self._write_active_token(trace.token_input, forced)
                token_refreshes += 1
        self.last_trace_counters = {
            "trace_replays": int(trace.generated),
            "execute_trace_blocking": False,
            "steady_state_token_refreshes": token_refreshes,
            "steady_state_position_refreshes": 0,
            "steady_state_rope_refreshes": 0,
            "steady_state_page_table_refreshes": 0,
            "steady_state_synchronizations": per_token_syncs + token_refreshes,
            "steady_state_token_readbacks": per_token_readbacks,
            "terminal_validation_synchronizations": 0,
            "terminal_validation_token_readbacks": 0,
            "host_sampling": False,
            "full_logits_readbacks": 0,
        }

    def _capture_decode_trace(self, first_input: int, prompt_len: int, cache: QwenFullModelCache) -> _DecodeTrace:
        token_input = self._active_token_buffer(first_input)
        token_output = self._sample_token_buffer(0)
        current_pos = self.model._positions_to_tt(torch.tensor([prompt_len], dtype=torch.int32))
        self._reset_sampling_state()
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._decode_sample_body(
            token_input=token_input, token_output=token_output, current_pos=current_pos, cache=cache
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        reset_token = self._active_token_buffer(first_input, on_host=True)
        reset_pos = self._positions_host(torch.tensor([prompt_len], dtype=torch.int32))
        ttnn.copy_host_to_device_tensor(reset_token, token_input)
        ttnn.copy_host_to_device_tensor(reset_pos, current_pos)
        ttnn.synchronize_device(self.mesh_device)
        self._trace = _DecodeTrace(
            trace_id=trace_id,
            cache=cache,
            token_input=token_input,
            token_output=token_output,
            current_pos=current_pos,
            prompt_len=prompt_len,
        )
        return self._trace

    def _prefill_sample_on_device(self, prompt: torch.Tensor, cache: QwenFullModelCache) -> int:
        if self.model.sampling is None:
            raise RuntimeError("optimized token-out generation requires on-device sampling")
        logits = self.model.prefill_user(
            prompt,
            cache=cache,
            user_id=0,
            return_all_logits=False,
            return_tt_logits=True,
        )
        logits = self._pad_logits_batch_for_sampling(logits)
        token_buffer = self._sample_token_buffer(0)
        self._reset_sampling_state()
        sampled = self.model.sampling.sample(logits, enable_trace=False, tt_out_tok=token_buffer)
        if isinstance(sampled, tuple):
            sampled = sampled[0]
        ttnn.synchronize_device(self.mesh_device)
        return self._read_active_token(token_buffer)

    def vllm_prefill_sample_on_device(
        self,
        tokens: torch.Tensor,
        *,
        cache: QwenFullModelCache,
        prompt_lens: list[int],
        sampling_params,
        empty_slots: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Prefill through the canonical on-device sampler and return token IDs."""

        audit_inc = getattr(self, "_audit_inc", None)
        audit_write = getattr(self, "_write_vllm_audit", None)

        def audit(stage: str) -> None:
            if callable(audit_inc):
                audit_inc(stage)
            if callable(audit_write):
                audit_write()

        if self.model.sampling is None:
            raise RuntimeError("vLLM prefill sampling requires on-device sampling")
        batch = int(tokens.shape[0])
        if empty_slots is None:
            empty_slots = list(range(batch))
        empty_slots = [int(slot) for slot in empty_slots]
        if len(empty_slots) != batch:
            raise ValueError(f"empty_slots length {len(empty_slots)} must match prefill batch {batch}")

        max_batch = int(self.model.sampling.tt_sampling.max_batch_size)
        if any(slot < 0 or slot >= max_batch for slot in empty_slots):
            raise ValueError(f"empty_slots {empty_slots} exceed sampling batch {max_batch}")
        if any(slot < 0 or slot >= cache.max_batch_size for slot in empty_slots):
            raise ValueError(f"empty_slots {empty_slots} exceed cache batch {cache.max_batch_size}")

        audit("prefill_sample_reset_linear_state_before")
        self.model.reset_linear_attention_state(cache, empty_slots)
        audit("prefill_sample_reset_linear_state_after")

        formatted_params = format_sampling_params(sampling_params, max_batch)
        slot_params = self._scatter_sampling_params_to_slots(formatted_params, empty_slots, max_batch=max_batch)
        self.model.sampling.reset_sampling_params(slot_params, empty_slots=empty_slots)
        self.model.sampling.seed_manager.reset_seed_from_slots(slot_params.seed, empty_slots)
        self.model.sampling.seed_manager.get_new_values(empty_slots)
        self.model.sampling.reset_prompt_tokens(
            self._sampling_prompt_tokens(tokens, prompt_lens=prompt_lens, slots=empty_slots, max_batch=max_batch)
        )
        self.model.sampling.reset_output_state()
        audit("prefill_sample_sampling_state_after")

        logits_by_slot: list[ttnn.Tensor | None] = [None] * max_batch
        for request_idx, (prompt_len, slot) in enumerate(zip(prompt_lens, empty_slots, strict=True)):
            prompt_len = int(prompt_len)
            if prompt_len <= 0:
                continue
            if prompt_len > tokens.shape[1]:
                raise ValueError(f"prompt_lens[{request_idx}]={prompt_len} exceeds token width {tokens.shape[1]}")
            audit("prefill_sample_prefill_user_before")
            logits_by_slot[slot] = self.model.prefill_user(
                tokens[request_idx : request_idx + 1, :prompt_len],
                cache=cache,
                user_id=slot,
                page_table_user_id=slot,
                return_all_logits=False,
                return_tt_logits=True,
            )
            audit("prefill_sample_prefill_user_after")
        first_logits = next((logits for logits in logits_by_slot if logits is not None), None)
        if first_logits is None:
            return torch.zeros((batch,), dtype=torch.int32)
        logits_filled = [logits if logits is not None else first_logits for logits in logits_by_slot]
        audit("prefill_sample_logits_concat_before")
        logits = ttnn.concat(logits_filled, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        audit("prefill_sample_logits_concat_after")
        token_buffer = self._sample_token_buffer(0, width=max_batch)
        audit("prefill_sample_sample_before")
        sampled = self.model.sampling.sample(logits, enable_trace=False, tt_out_tok=token_buffer)
        audit("prefill_sample_sample_after")
        tt_log_probs = None
        if isinstance(sampled, tuple):
            _, tt_log_probs = sampled
        audit("prefill_sample_sync_before")
        ttnn.synchronize_device(self.mesh_device)
        audit("prefill_sample_sync_after")
        audit("prefill_sample_token_read_before")
        sampled_tokens = self._read_token_buffer(token_buffer)
        audit("prefill_sample_token_read_after")
        output_tokens = sampled_tokens[empty_slots].to(torch.int32)
        if tt_log_probs is None:
            return output_tokens
        audit("prefill_sample_logprob_read_before")
        host_log_probs = self._read_log_probs(tt_log_probs)
        audit("prefill_sample_logprob_read_after")
        return output_tokens, host_log_probs

    def _warmup_traced_decode(self, *, prompt_len: int, max_seq_len: int) -> None:
        compile_cache = self.allocate_kv_cache(max_batch_size=1, max_seq_len=max_seq_len)
        token_buffer = self._sample_token_buffer(0)
        current_pos = self.model._positions_to_tt(torch.tensor([min(prompt_len, max_seq_len - 1)], dtype=torch.int32))
        self._reset_sampling_state()
        token_input = self._active_token_buffer(0)
        self._decode_sample_body(
            token_input=token_input, token_output=token_buffer, current_pos=current_pos, cache=compile_cache
        )
        ttnn.synchronize_device(self.mesh_device)
        del compile_cache, token_input, token_buffer, current_pos
        gc.collect()
        self._reset_sampling_state()

    def _decode_sample_body(
        self,
        *,
        token_input: ttnn.Tensor,
        token_output: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        cache: QwenFullModelCache,
    ) -> None:
        logits = self.model.decode_logits_tt(token_input, current_pos, cache=cache)
        logits = self._pad_logits_batch_for_sampling(logits)
        assert self.model.sampling is not None
        sampled = self.model.sampling.sample(logits, enable_trace=False, tt_out_tok=token_output)
        if isinstance(sampled, tuple):
            sampled = sampled[0]
        width = int(token_input.shape[-1])
        ttnn.slice(sampled, (0, 0, 0, 0), (1, 1, 1, width), output_tensor=token_input)
        ttnn.plus_one(current_pos, skip_negative_entries=True)

    def _pad_logits_batch_for_sampling(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        shape = tuple(int(dim) for dim in logits.shape)
        batch = shape[-2]
        padded_batch = nearest_32(batch)
        if padded_batch == batch:
            return logits
        return ttnn.pad(
            logits,
            padding=[(0, 0), (0, 0), (0, padded_batch - batch), (0, 0)],
            value=0.0,
        )

    def _active_token_buffer(self, token: int | list[int] | torch.Tensor, *, on_host: bool = False) -> ttnn.Tensor:
        width = 1 if isinstance(token, int) else max(1, int(torch.as_tensor(token).numel()))
        return self._token_buffer(token, width=width, on_host=on_host)

    def _sample_token_buffer(
        self,
        token: int | list[int] | torch.Tensor,
        *,
        width: int = 32,
        on_host: bool = False,
    ) -> ttnn.Tensor:
        return self._token_buffer(token, width=width, on_host=on_host)

    def _token_buffer(self, token: int | list[int] | torch.Tensor, *, width: int, on_host: bool = False) -> ttnn.Tensor:
        data = torch.zeros((1, 1, 1, width), dtype=torch.uint32)
        values = torch.as_tensor([token] if isinstance(token, int) else token, dtype=torch.uint32).reshape(-1)
        data[0, 0, 0, : min(width, values.numel())] = values[:width]
        return ttnn.from_torch(
            data,
            device=None if on_host else self.mesh_device,
            dtype=self.model.sampling_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _positions_host(self, positions: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            positions.to(torch.int32).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _read_active_token(self, token_buffer: ttnn.Tensor) -> int:
        local = ttnn.get_device_tensors(token_buffer)[0]
        return int(ttnn.to_torch(local.cpu()).reshape(-1)[0].item())

    def _read_token_buffer(self, token_buffer: ttnn.Tensor) -> torch.Tensor:
        local = ttnn.get_device_tensors(token_buffer)[0]
        return ttnn.to_torch(local.cpu()).reshape(-1).to(torch.int32)

    def _read_log_probs(self, tt_log_probs) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(tt_log_probs, tuple):
            return tuple(self._read_log_probs(part) for part in tt_log_probs)  # type: ignore[return-value]
        local = ttnn.get_device_tensors(tt_log_probs)[0]
        return ttnn.to_torch(local.cpu()).reshape(-1).float()

    def _write_active_token(self, token_buffer: ttnn.Tensor, token: int) -> None:
        host_token = self._token_buffer(token, width=int(token_buffer.shape[-1]), on_host=True)
        ttnn.copy_host_to_device_tensor(host_token, token_buffer)
        ttnn.synchronize_device(self.mesh_device)

    def _reset_sampling_state(self) -> None:
        if self.model.sampling is None:
            return
        params = format_sampling_params(
            SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=0),
            self.model.sampling.tt_sampling.max_batch_size,
        )
        self.model.sampling.reset_sampling_params(params, empty_slots=[])
        self.model.sampling.reset_output_state()

    def _scatter_sampling_params_to_slots(self, sampling_params, slots: list[int], *, max_batch: int):
        inactive_defaults = {
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": None,
            "enable_log_probs": False,
            "num_logprobs": 0,
        }

        def scatter(value):
            if not isinstance(value, list):
                return value
            values = list(value)
            user_values = values[: len(slots)]
            if not user_values:
                return values
            filler = inactive_defaults.get(field.name, user_values[-1])
            scattered = [filler for _ in range(max_batch)]
            for request_value, slot in zip(user_values, slots, strict=True):
                scattered[int(slot)] = request_value
            return scattered

        updates = {}
        for field in fields(sampling_params):
            updates[field.name] = scatter(getattr(sampling_params, field.name))
        return replace(sampling_params, **updates)

    def _sampling_prompt_tokens(
        self,
        tokens: torch.Tensor,
        *,
        prompt_lens: list[int],
        slots: list[int],
        max_batch: int,
    ) -> torch.Tensor:
        max_prompt_len = max(max(int(length) for length in prompt_lens), 1)
        prompt_tokens = torch.full((max_batch, max_prompt_len), -1, dtype=torch.long, device=tokens.device)
        for request_idx, (prompt_len, slot) in enumerate(zip(prompt_lens, slots, strict=True)):
            prompt_len = min(int(prompt_len), tokens.shape[1], max_prompt_len)
            if prompt_len > 0:
                prompt_tokens[int(slot), :prompt_len] = tokens[request_idx, :prompt_len]
        return prompt_tokens


def _no_readback_counters(decode_tokens: int, *, validate_final_token: bool) -> dict[str, int | bool]:
    return {
        "trace_replays": int(decode_tokens),
        "trace_decode_steps": int(decode_tokens),
        "execute_trace_blocking": False,
        "steady_state_token_refreshes": 0,
        "steady_state_position_refreshes": 0,
        "steady_state_rope_refreshes": 0,
        "steady_state_page_table_refreshes": 0,
        "steady_state_synchronizations": 0,
        "steady_state_token_readbacks": 0,
        "terminal_validation_synchronizations": 1 if decode_tokens > 0 else 0,
        "terminal_validation_token_readbacks": 1 if validate_final_token and decode_tokens > 0 else 0,
        "host_sampling": False,
        "full_logits_readbacks": 0,
    }


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> QwenReadinessGenerator:
    return QwenReadinessGenerator(model_dir=model_dir, mesh_device=mesh_device, **kwargs)


__all__ = ["QwenReadinessGenerator", "build_generator"]
