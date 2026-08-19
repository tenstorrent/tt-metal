# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Readiness generator for the Qwen3.6-35B-A3B full TTNN model."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
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
    ):
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.model_id = model_id
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
        )
        self.host_sampling_compat = host_sampling_compat
        self.cache: QwenFullModelCache | None = None
        self._trace: _DecodeTrace | None = None
        self.last_timings: dict[str, float] = {}
        self._reset_sampling_state()

    def reset(self) -> None:
        if self._trace is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._trace.trace_id)
            except Exception:
                pass
        self._trace = None
        self.cache = None
        self._reset_sampling_state()

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
        while len(predictions) < max_new_tokens:
            ttnn.execute_trace(self.mesh_device, trace.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            predictions.append(self._read_active_token(trace.token_input))
            trace.generated += 1

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
        for step in range(1, max_new_tokens):
            ttnn.execute_trace(self.mesh_device, trace.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh_device)
            pred = self._read_active_token(trace.token_input)
            predictions.append(pred)
            trace.generated += 1
            forced = next_input(step, pred)
            if step + 1 < max_new_tokens:
                self._write_active_token(trace.token_input, forced)

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
        active_output = ttnn.slice(token_output, (0, 0, 0, 0), (1, 1, 1, 1))
        ttnn.copy(active_output, token_input)
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

    def _active_token_buffer(self, token: int, *, on_host: bool = False) -> ttnn.Tensor:
        return self._token_buffer(token, width=1, on_host=on_host)

    def _sample_token_buffer(self, token: int, *, on_host: bool = False) -> ttnn.Tensor:
        return self._token_buffer(token, width=32, on_host=on_host)

    def _token_buffer(self, token: int, *, width: int, on_host: bool = False) -> ttnn.Tensor:
        data = torch.zeros((1, 1, 1, width), dtype=torch.uint32)
        data[0, 0, 0, 0] = int(token)
        return ttnn.from_torch(
            data,
            device=None if on_host else self.mesh_device,
            dtype=ttnn.uint32,
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


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> QwenReadinessGenerator:
    return QwenReadinessGenerator(model_dir=model_dir, mesh_device=mesh_device, **kwargs)


__all__ = ["QwenReadinessGenerator", "build_generator"]
