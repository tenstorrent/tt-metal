# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Readiness generator for the Phi-3.5-mini full TTNN autoport."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.model import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_REVISION,
    MODEL_ID,
    SAMPLING_BATCH_SIZE,
    Phi35MiniForCausalLMTT,
)
from models.common.readiness_check.contract import Generator, NextInputFn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.sampling.tt_log_probs import LogProbsResult


@dataclass
class DecodeTraceState:
    trace_id: int | None = None
    token_input: ttnn.Tensor | None = None
    current_pos: ttnn.Tensor | None = None
    page_table: ttnn.Tensor | None = None
    logits: ttnn.Tensor | None = None
    sampled: Any | None = None
    rope_sequence_length: int | None = None
    sampling_mode_key: tuple[bool, bool, bool] | None = None
    capture_sampling: bool = False


@dataclass
class TraceCounters:
    model_trace_captures: int = 0
    model_trace_replays: int = 0
    sampling_trace_captures: int = 0
    token_setup_refreshes: int = 0
    token_teacher_forcing_refreshes: int = 0
    position_setup_refreshes: int = 0
    position_steady_state_refreshes: int = 0
    page_table_setup_refreshes: int = 0
    page_table_changed_refreshes: int = 0
    sampled_token_readbacks: int = 0
    no_readback_decode_steps: int = 0
    full_logits_decode_readbacks: int = 0
    device_position_advances: int = 0
    device_token_feedbacks: int = 0
    synchronizations: int = 0
    sampling_trace_variant_precompiles: int = 0


class _SamplingArgs:
    def __init__(self, model: Phi35MiniForCausalLMTT) -> None:
        cfg = model.full_config
        self.vocab_size = cfg.vocab_size
        self.padded_vocab_size = cfg.padded_vocab_size
        self.max_batch_size = SAMPLING_BATCH_SIZE
        self.max_top_k = 32
        self.cluster_shape = tuple(model.mesh_device.shape)
        self.sampling_all_gather_axis = 1
        self.sampling_dp = 1
        self.num_devices = model.mesh_device.get_num_devices()
        self.is_galaxy = False
        self.model_config = {}
        self.use_topk_logprobs = True
        self.sub_core_grids = None
        self.sub_core_grid_topk = None
        self.pad_logits_to_power_of_2 = os.getenv("PHI35_SAMPLING_PAD_TO_POWER_OF_2", "0") == "1"


class Phi35MiniGenerator(Generator):
    """High-level and low-level generator contract for readiness checks."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        mesh_device,
        hf_model_id: str = MODEL_ID,
        revision: str | None = DEFAULT_REVISION,
        hf_snapshot: str | Path | None = None,
        num_layers: int | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        precision_config_path: str | Path | None = None,
        allocate_standalone_cache: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.mesh_device = mesh_device
        self.model = Phi35MiniForCausalLMTT.from_hf(
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            revision=revision,
            hf_snapshot=hf_snapshot,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            model_dir=self.model_dir,
            precision_config_path=precision_config_path,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, revision=revision, trust_remote_code=True)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.kv_cache = self.model.allocate_kv_cache() if allocate_standalone_cache else None
        self.page_table = (
            self.model.make_page_table(max_seq_len=self.model.full_config.max_seq_len)
            if allocate_standalone_cache
            else None
        )
        self.trace = DecodeTraceState()
        self.counters = TraceCounters()
        self.sampling = SamplingGenerator(
            args=_SamplingArgs(self.model),
            mesh_device=mesh_device,
            tt_ccl=None,
            enable_internal_trace=True,
        )
        self._sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=0.0, seed=None)
        self._reset_sampling_params(self._sampling_params)
        self._last_perf: dict[str, float] = {}
        self._vllm_page_table_host: torch.Tensor | None = None
        self._vllm_page_table_device: ttnn.Tensor | None = None
        self._vllm_expected_pos: int | None = None
        self._sampling_trace_variants_precompiled = False
        self._vllm_last_sampling_params: Any | None = None

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
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"Phi-3.5 generator expects tokens [1, seq], got {tuple(tokens.shape)}")
        prompt_len = int(prompt_lens[0])
        padded_tokens = _pad_tokens(tokens, self.model.full_config.block_size, int(self.pad_token_id))
        tt_tokens = self.model.tokens_to_device(padded_tokens)
        tt_page_table = self._coerce_page_table(page_table)
        tt_kv_cache = kv_cache if kv_cache is not None else self.kv_cache
        logits_tt = self.model.prefill_forward_ttnn(
            tt_tokens,
            page_table=tt_page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=[prompt_len],
            return_all_logits=return_all_logits,
        )
        logits = self.model.logits_to_torch(logits_tt)
        if return_all_logits:
            logits = logits[:, :, :prompt_len, :].reshape(1, prompt_len, self.model.full_config.vocab_size)
        else:
            logits = logits.reshape(1, 1, self.model.full_config.vocab_size)
        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor | None,
        kv_cache: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if tokens.ndim == 2:
            token_id = int(tokens[0, 0].item())
        else:
            token_id = int(tokens.reshape(-1)[0].item())
        pos = int(start_pos.reshape(-1)[0].item())
        tt_tokens = self._make_token_input_device(token_id)
        current_pos = self._make_current_pos_device(pos)
        logits_tt = self.model.decode_forward_from_ttnn_inputs(
            tt_tokens,
            current_pos,
            page_table=self._coerce_page_table(page_table),
            kv_cache=kv_cache if kv_cache is not None else self.kv_cache,
            rope_sequence_length=max(pos + 1, self.model.full_config.max_seq_len),
        )
        self.counters.full_logits_decode_readbacks += 1
        return self.model.logits_to_torch(logits_tt).reshape(1, self.model.full_config.vocab_size)

    def prefill_forward_token_out(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        prompt_lens: list[int] | torch.Tensor,
        sampling_params: Any,
        empty_slots: list[int] | None = None,
        prompt_tokens: torch.Tensor | None = None,
        enable_trace: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """vLLM prefill path that samples on device and reads back token ids only."""

        if sampling_params is None:
            raise ValueError("prefill_forward_token_out requires on-device sampling_params")
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"Phi-3.5 vLLM prefill supports batch 1, got {tuple(tokens.shape)}")

        prompt_len = int(torch.as_tensor(prompt_lens).reshape(-1)[0].item())
        if prompt_len <= 0:
            return torch.zeros((1,), dtype=torch.int64)

        empty_slots = [0] if empty_slots is None else [int(slot) for slot in empty_slots]
        self._apply_vllm_prefill_sampling_state(
            sampling_params=sampling_params,
            empty_slots=empty_slots,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else tokens,
        )

        padded_tokens = _pad_tokens(tokens, self.model.full_config.block_size, int(self.pad_token_id))
        tt_tokens = self.model.tokens_to_device(padded_tokens)
        tt_page_table, _ = self._coerce_page_table_vllm(page_table)
        raw_logits_tt = self.model.prefill_forward_ttnn(
            tt_tokens,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            prompt_lens=[prompt_len],
            return_all_logits=False,
        )
        logits_tt = self._pad_logits_for_sampling(raw_logits_tt)
        sampled = self.sampling.sample(logits_tt, enable_trace=enable_trace, skip_precompile=True)
        out = self._sampled_output_to_host(sampled, batch_size=1)
        self._deallocate_prefill_temporaries(tt_tokens, raw_logits_tt, logits_tt, sampled)
        self._vllm_expected_pos = None
        return out

    def decode_forward_token_out(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor | ttnn.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        sampling_params: Any,
        enable_trace: bool = True,
        read_from_device: bool = True,
        reset_batch: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        """vLLM decode path using traced model replay and canonical split sampling."""

        if sampling_params is None:
            raise ValueError("decode_forward_token_out requires on-device sampling_params")
        active_slots = _active_decode_slots(start_pos)
        if not active_slots:
            empty = torch.empty((0,), dtype=torch.int32)
            return empty if read_from_device else None
        if len(active_slots) != 1:
            raise ValueError(f"Phi-3.5 vLLM decode supports one active slot, got {active_slots}")

        slot = active_slots[0]
        token_id = int(tokens.reshape(tokens.shape[0], -1)[slot, 0].item())
        pos = int(start_pos.reshape(-1)[slot].item())
        self._apply_vllm_decode_sampling_state(
            sampling_params=sampling_params,
            active_slots=active_slots,
            start_pos=start_pos,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
        )

        tt_page_table, page_table_changed = self._coerce_page_table_vllm(page_table)
        rope_sequence_length = max(pos + 1, self.model.full_config.max_seq_len)
        current_sampling_mode_key = self._sampling_mode_key()
        needs_trace_capture = (
            self.trace.trace_id is None
            or self.trace.logits is None
            or self.trace.sampled is None
            or not self.trace.capture_sampling
        )
        sampling_mode_changed = (
            self.trace.trace_id is not None
            and self.trace.capture_sampling
            and self.trace.sampling_mode_key != current_sampling_mode_key
        )
        reset_inputs = (
            reset_batch
            or needs_trace_capture
            or sampling_mode_changed
            or self._vllm_expected_pos is None
            or self._vllm_expected_pos != pos
        )

        if not enable_trace:
            sampled = self._decode_next_token_eager_device(
                token_id=token_id,
                start_pos=pos,
                page_table=tt_page_table,
                kv_cache=kv_cache,
                rope_sequence_length=rope_sequence_length,
            )
            self._vllm_expected_pos = pos + 1
            if not read_from_device:
                return sampled
            return self._sampled_output_to_host(sampled, batch_size=1)

        if needs_trace_capture or sampling_mode_changed or self.trace.page_table is not tt_page_table:
            if self.trace.trace_id is not None:
                self._release_decode_trace()
            self._ensure_decode_trace(
                token_id=token_id,
                start_pos=pos,
                page_table=tt_page_table,
                kv_cache=kv_cache,
                rope_sequence_length=rope_sequence_length,
                capture_sampling=True,
            )
            reset_inputs = False
        elif self.trace.rope_sequence_length != rope_sequence_length:
            self.trace.rope_sequence_length = rope_sequence_length

        if reset_inputs:
            self._copy_token_to_trace_input(token_id, teacher_forcing=False)
            self._copy_position_to_trace_input(pos, setup=False)
        elif page_table_changed:
            # The persistent page-table tensor was updated in place; token and
            # position remain device-owned from the previous sampled-token path.
            pass

        sampled = self._decode_next_token_traced(readback=False, return_device=True)
        self._vllm_expected_pos = pos + 1
        if not read_from_device:
            return sampled
        return self._sampled_output_to_host(sampled, batch_size=1)

    def read_decode_output(self, tt_out: Any, async_read: bool = True) -> Any:
        if tt_out is None or isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out
        host_out = self._tt_output_cpu(tt_out, async_read=async_read)
        if async_read:
            return host_out, [ttnn.record_event(self.mesh_device, 0)]
        return host_out

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = True) -> Any:
        if isinstance(tt_out, tuple):
            tokens, log_probs = tt_out
            tokens_t = self._host_tt_to_torch(tokens).reshape(-1).to(torch.int32)
            log_probs_t = None
            if log_probs is not None:
                log_probs_t = self._host_tt_to_torch(log_probs)
                if not isinstance(log_probs_t, tuple):
                    log_probs_t = log_probs_t.reshape(-1)
            return tokens_t, log_probs_t
        if is_tokens:
            return self._host_tt_to_torch(tt_out).reshape(-1).to(torch.int32)
        return self._host_tt_to_torch(tt_out).reshape(1, 1, self.model.full_config.vocab_size)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        enable_trace: bool,
        can_sample_on_device: bool,
        **_: Any,
    ) -> None:
        if not can_sample_on_device:
            return
        tokens = torch.full((1, self.model.full_config.block_size), int(self.pad_token_id), dtype=torch.long)
        tokens[0, 0] = int(self.eos_token_id if self.eos_token_id is not None else self.pad_token_id)
        page_table = torch.zeros((1, 1), dtype=torch.int32)
        params = SamplingParams(temperature=0.0, top_k=1, top_p=0.0, seed=None)
        self.prefill_forward_token_out(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[1],
            sampling_params=params,
            enable_trace=False,
        )

    def warmup_model_decode(
        self,
        *,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        max_batch_size: int,
        num_blocks: int,
        enable_trace: bool,
        can_sample_on_device: bool,
        **_: Any,
    ) -> None:
        if not can_sample_on_device:
            return
        if max_batch_size != 1:
            raise ValueError(f"Phi-3.5 vLLM decode supports max_batch_size=1, got {max_batch_size}")
        tokens = torch.zeros((1, 1), dtype=torch.int32)
        start_pos = torch.zeros((1,), dtype=torch.int32)
        page_table = torch.zeros((1, max(1, int(num_blocks))), dtype=torch.int32)
        params = SamplingParams(temperature=0.0, top_k=1, top_p=0.0, seed=None)
        out = self.decode_forward_token_out(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=params,
            enable_trace=enable_trace,
            read_from_device=False,
            reset_batch=True,
        )
        if out is not None:
            self.read_decode_output(out, async_read=False)

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        sampling_params: SamplingParams | None = None,
        stop_on_eos: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if len(prompt_token_ids) > self.model.full_config.max_seq_len:
            raise ValueError(
                f"prompt length {len(prompt_token_ids)} exceeds max_seq_len={self.model.full_config.max_seq_len}"
            )
        if sampling_params is not None:
            self._sampling_params = sampling_params
            self._reset_sampling_params(sampling_params)

        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        start_s = time.perf_counter()
        prefill_logits = self.prefill_forward(
            prompt,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=False,
        )
        first_token_s = time.perf_counter()
        predicted = int(torch.argmax(prefill_logits[0, -1]).item())
        outputs = [predicted]
        feed_token = next_input(0, predicted) if next_input is not None else predicted

        if stop_on_eos and next_input is None and _is_eos(predicted, self.eos_token_id):
            self._record_perf(start_s, first_token_s, first_token_s, len(outputs))
            return outputs

        if max_new_tokens == 1:
            self._record_perf(start_s, first_token_s, first_token_s, len(outputs))
            return outputs

        if not enable_trace:
            last_s = self._generate_eager(
                outputs=outputs,
                feed_token=int(feed_token),
                start_pos=len(prompt_token_ids),
                max_new_tokens=max_new_tokens,
                next_input=next_input,
                stop_on_eos=stop_on_eos,
            )
            self._record_perf(start_s, first_token_s, last_s, len(outputs))
            return outputs

        self._ensure_decode_trace(
            token_id=int(feed_token),
            start_pos=len(prompt_token_ids),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            rope_sequence_length=max(self.model.full_config.max_seq_len, len(prompt_token_ids) + max_new_tokens),
        )

        last_decode_s = first_token_s
        for step in range(1, max_new_tokens):
            predicted = self._decode_next_token_traced(readback=True)
            last_decode_s = time.perf_counter()
            outputs.append(predicted)
            feed_token = next_input(step, predicted) if next_input is not None else predicted
            if stop_on_eos and next_input is None and _is_eos(predicted, self.eos_token_id):
                break
            if next_input is not None and step + 1 < max_new_tokens:
                self._copy_token_to_trace_input(int(feed_token), teacher_forcing=True)

        self._record_perf(start_s, first_token_s, last_decode_s, len(outputs))
        return outputs

    def benchmark_token_out_decode(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        sampling_params: SamplingParams | None = None,
        stop_on_eos: bool = False,
        warmup_decode_steps: int = 0,
    ) -> dict[str, Any]:
        """Run traced token-out decode without per-token sampled-token readback.

        This is the optimized steady-state benchmark path. It still uses prefill
        to seed the first decode token, then replays model and sampling traces
        with ``tt_out_tok`` feedback and synchronizes once after the decode loop.
        """
        if stop_on_eos:
            raise ValueError("no-readback benchmark cannot stop on EOS without reading sampled tokens")
        if max_new_tokens <= 0:
            return {}
        if not prompt_token_ids:
            raise ValueError("prompt_token_ids must be non-empty")
        if sampling_params is not None:
            self._sampling_params = sampling_params
            self._reset_sampling_params(sampling_params)

        prompt = torch.tensor([prompt_token_ids], dtype=torch.long)
        start_s = time.perf_counter()
        prefill_logits = self.prefill_forward(
            prompt,
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            return_all_logits=False,
        )
        first_token_s = time.perf_counter()
        feed_token = int(torch.argmax(prefill_logits[0, -1]).item())

        decode_tokens = max(max_new_tokens - 1, 0)
        measured_decode_s = 0.0
        if decode_tokens:
            self._ensure_decode_trace(
                token_id=feed_token,
                start_pos=len(prompt_token_ids),
                page_table=self.page_table,
                kv_cache=self.kv_cache,
                rope_sequence_length=max(self.model.full_config.max_seq_len, len(prompt_token_ids) + max_new_tokens),
            )
            for _ in range(max(0, int(warmup_decode_steps))):
                self._decode_next_token_traced(readback=False)
            if warmup_decode_steps:
                ttnn.synchronize_device(self.mesh_device)
                self.counters.synchronizations += 1

            decode_start_s = time.perf_counter()
            for _ in range(decode_tokens):
                self._decode_next_token_traced(readback=False)
            ttnn.synchronize_device(self.mesh_device)
            self.counters.synchronizations += 1
            measured_decode_s = max(time.perf_counter() - decode_start_s, 0.0)
        last_decode_s = time.perf_counter()
        self._record_perf(start_s, first_token_s, last_decode_s, max_new_tokens)
        result = self.last_perf()
        result["decode_tokens"] = float(decode_tokens)
        result["decode_elapsed_s"] = measured_decode_s
        result["decode_t/s/u"] = (decode_tokens / measured_decode_s) if measured_decode_s > 0 else 0.0
        result.update(
            {
                "readback": False,
                "trace_replay_blocking": False,
                "sampling_pad_to_power_of_2": self.sampling.tt_sampling.pad_to_power_of_2,
                "warmup_decode_steps": int(warmup_decode_steps),
                "counters": self.trace_counters(),
            }
        )
        return result

    def reset(self) -> None:
        if self.kv_cache is not None:
            for layer_cache in self.kv_cache:
                for cache_tensor in layer_cache:
                    ttnn.fill(cache_tensor, 0.0, output_tensor=cache_tensor)
        self._reset_sampling_params(self._sampling_params)
        self._vllm_page_table_host = None
        self._vllm_page_table_device = None
        self._vllm_expected_pos = None

    def teardown(self) -> None:
        if self.trace.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace.trace_id)
            self.trace.trace_id = None
        self.sampling.reset_trace()

    def trace_counters(self) -> dict[str, int]:
        return dict(self.counters.__dict__)

    def last_perf(self) -> dict[str, float]:
        return dict(self._last_perf)

    def _generate_eager(
        self,
        *,
        outputs: list[int],
        feed_token: int,
        start_pos: int,
        max_new_tokens: int,
        next_input: Optional[NextInputFn],
        stop_on_eos: bool,
    ) -> float:
        last_s = time.perf_counter()
        token = int(feed_token)
        for step in range(1, max_new_tokens):
            logits = self.decode_forward(
                torch.tensor([[token]], dtype=torch.long),
                torch.tensor([start_pos + step - 1], dtype=torch.int32),
                page_table=self.page_table,
                kv_cache=self.kv_cache,
            )
            predicted = int(torch.argmax(logits[0]).item())
            outputs.append(predicted)
            last_s = time.perf_counter()
            token = next_input(step, predicted) if next_input is not None else predicted
            if stop_on_eos and next_input is None and _is_eos(predicted, self.eos_token_id):
                break
        return last_s

    def _apply_vllm_prefill_sampling_state(
        self,
        *,
        sampling_params: Any,
        empty_slots: list[int],
        prompt_tokens: torch.Tensor | None,
    ) -> None:
        formatted = format_sampling_params(sampling_params, self.sampling.tt_sampling.max_batch_size)
        self.sampling.apply_prefill_state(
            sampling_params=formatted,
            prompt_tokens=prompt_tokens,
            empty_slots=empty_slots,
            replicate_seeds=False,
        )

    def _apply_vllm_decode_sampling_state(
        self,
        *,
        sampling_params: Any,
        active_slots: list[int],
        start_pos: torch.Tensor,
        reset_batch: bool,
        prompt_tokens: torch.Tensor | None,
        output_tokens: torch.Tensor | None,
        slot_remap: torch.Tensor | None,
    ) -> None:
        formatted = format_sampling_params(sampling_params, self.sampling.tt_sampling.max_batch_size)
        self._vllm_last_sampling_params = formatted
        if slot_remap is not None:
            self.sampling.seed_manager.apply_slot_remap(slot_remap.reshape(-1).tolist())
        self.sampling.reset_sampling_params(formatted)
        if reset_batch:
            self.sampling.reset_prompt_tokens(prompt_tokens)
            self.sampling.reset_output_state(output_tokens)
        reset_seed = self.sampling.seed_manager.reset_seed_from_slots_if_needed(formatted.seed, active_slots)
        if reset_batch or reset_seed:
            self.sampling.seed_manager.align_seed_counters_to_positions(formatted.seed, active_slots, start_pos)
        self.sampling.seed_manager.get_new_values(active_slots)

    def _coerce_page_table_vllm(self, page_table: torch.Tensor | ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
        if isinstance(page_table, ttnn.Tensor):
            changed = self._vllm_page_table_device is not page_table
            self._vllm_page_table_device = page_table
            self._vllm_page_table_host = None
            return page_table, changed

        host = page_table.to(torch.int32).contiguous()
        if self._vllm_page_table_device is None or self._vllm_page_table_host is None:
            self._vllm_page_table_host = host.clone()
            self._vllm_page_table_device = self._page_table_host_to_device(host)
            self.counters.page_table_setup_refreshes += 1
            return self._vllm_page_table_device, True

        if tuple(host.shape) != tuple(self._vllm_page_table_host.shape):
            self._vllm_page_table_host = host.clone()
            self._vllm_page_table_device = self._page_table_host_to_device(host)
            self.counters.page_table_changed_refreshes += 1
            return self._vllm_page_table_device, True

        if torch.equal(host, self._vllm_page_table_host):
            return self._vllm_page_table_device, False

        host_tt = ttnn.from_torch(
            host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host_tt, self._vllm_page_table_device)
        self._vllm_page_table_host = host.clone()
        self.counters.page_table_changed_refreshes += 1
        return self._vllm_page_table_device, True

    def _page_table_host_to_device(self, host: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _pad_logits_for_sampling(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        batch_dim = int(logits.shape[-2])
        if batch_dim >= SAMPLING_BATCH_SIZE:
            return logits
        return ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, SAMPLING_BATCH_SIZE - batch_dim), (0, 0)], value=0.0)

    def _decode_next_token_eager_device(
        self,
        *,
        token_id: int,
        start_pos: int,
        page_table: ttnn.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        rope_sequence_length: int,
    ) -> Any:
        tt_tokens = self._make_token_input_device(token_id)
        current_pos = self._make_current_pos_device(start_pos)
        logits_tt = self.model.decode_forward_from_ttnn_inputs(
            tt_tokens,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=rope_sequence_length,
            pad_for_sampling=True,
            advance_position=False,
        )
        return self.sampling.sample(logits_tt, enable_trace=False, tt_out_tok=tt_tokens)

    def _sampled_output_to_host(self, sampled: Any, *, batch_size: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        host = self.read_decode_output(sampled, async_read=False)
        processed = self.process_decode_output_host(host, is_tokens=True)
        if isinstance(processed, tuple):
            tokens, log_probs = processed
            tokens = tokens[:batch_size].to(torch.int64)
            if isinstance(log_probs, tuple):
                log_probs = tuple(part[:batch_size] for part in log_probs)
            elif log_probs is not None:
                log_probs = log_probs[:batch_size]
            return tokens, log_probs
        return processed[:batch_size].to(torch.int64)

    def _deallocate_prefill_temporaries(
        self,
        tt_tokens: ttnn.Tensor,
        raw_logits: ttnn.Tensor,
        logits: ttnn.Tensor,
        sampled: Any,
    ) -> None:
        if isinstance(sampled, tuple):
            token_tensor = sampled[0]
            if isinstance(token_tensor, ttnn.Tensor):
                ttnn.deallocate(token_tensor)
        elif isinstance(sampled, ttnn.Tensor):
            ttnn.deallocate(sampled)
        if logits is not raw_logits:
            ttnn.deallocate(logits)
        ttnn.deallocate(raw_logits)
        ttnn.deallocate(tt_tokens)

    def _tt_output_cpu(self, tt_out: Any, *, async_read: bool) -> Any:
        if isinstance(tt_out, tuple):
            return tuple(self._tt_output_cpu(item, async_read=async_read) if item is not None else None for item in tt_out)
        if isinstance(tt_out, list):
            return [self._tt_output_cpu(item, async_read=async_read) if item is not None else None for item in tt_out]
        if isinstance(tt_out, LogProbsResult):
            return tt_out.cpu(blocking=not async_read)
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        if isinstance(tt_out, ttnn.Tensor):
            return tt_out.cpu(blocking=not async_read, cq_id=0)
        return tt_out

    def _host_tt_to_torch(self, value: Any) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if value is None:
            return torch.empty((0,), dtype=torch.float32)
        if isinstance(value, LogProbsResult):
            topk_logprobs = value.topk_logprobs_host if value.topk_logprobs_host is not None else value.topk_logprobs
            topk_indices = value.topk_indices_host if value.topk_indices_host is not None else value.topk_indices
            return (
                self._logprobs_tensor_to_torch(topk_logprobs).float(),
                self._logprobs_tensor_to_torch(topk_indices).int(),
            )
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, ttnn.Tensor):
            try:
                return ttnn.to_torch(ttnn.get_device_tensors(value)[0])
            except RuntimeError:
                return ttnn.to_torch(value)
        if isinstance(value, list):
            if not value:
                return torch.empty((0,), dtype=torch.float32)
            return self._host_tt_to_torch(value[0])
        raise TypeError(f"Unsupported TT host output type {type(value)}")

    def _logprobs_tensor_to_torch(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, ttnn.Tensor):
            try:
                tensor = ttnn.to_torch(ttnn.get_device_tensors(value)[0])
            except RuntimeError:
                tensor = ttnn.to_torch(value)
        else:
            raise TypeError(f"Unsupported logprobs tensor type {type(value)}")
        if tensor.dim() >= 4:
            tensor = tensor[0, 0]
        return tensor.reshape(-1, tensor.shape[-1])

    def _ensure_decode_trace(
        self,
        *,
        token_id: int,
        start_pos: int,
        page_table: ttnn.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        rope_sequence_length: int,
        capture_sampling: bool = False,
    ) -> None:
        sampling_mode_key = self._sampling_mode_key() if capture_sampling else None
        if self.trace.trace_id is not None:
            if (
                self.trace.page_table is not page_table
                or self.trace.rope_sequence_length != rope_sequence_length
                or self.trace.capture_sampling != capture_sampling
                or (capture_sampling and self.trace.sampling_mode_key != sampling_mode_key)
            ):
                if self.trace.page_table is not page_table:
                    self.counters.page_table_changed_refreshes += 1
                self._release_decode_trace()
                self._ensure_decode_trace(
                    token_id=token_id,
                    start_pos=start_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    rope_sequence_length=rope_sequence_length,
                    capture_sampling=capture_sampling,
                )
                return
            self._copy_token_to_trace_input(token_id, teacher_forcing=False)
            self._copy_position_to_trace_input(start_pos, setup=True)
            self.trace.rope_sequence_length = rope_sequence_length
            return

        self.trace.token_input = self._make_token_input_device(token_id)
        self.trace.current_pos = self._make_current_pos_device(start_pos)
        self.trace.page_table = page_table
        self.trace.rope_sequence_length = rope_sequence_length
        self.counters.token_setup_refreshes += 1
        self.counters.position_setup_refreshes += 1
        self.counters.page_table_setup_refreshes += 1

        warm_logits = self.model.decode_forward_from_ttnn_inputs(
            self.trace.token_input,
            self.trace.current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=rope_sequence_length,
            pad_for_sampling=True,
            advance_position=True,
        )
        self.sampling.sample(warm_logits, enable_trace=False, tt_out_tok=self.trace.token_input)
        if not capture_sampling:
            self._precompile_sampling_trace_variants(warm_logits, tt_out_tok=self.trace.token_input)
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1
        ttnn.deallocate(warm_logits)
        self._copy_token_to_trace_input(token_id, teacher_forcing=False)
        self._copy_position_to_trace_input(start_pos, setup=True)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits = self.model.decode_forward_from_ttnn_inputs(
            self.trace.token_input,
            self.trace.current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=rope_sequence_length,
            pad_for_sampling=True,
            advance_position=True,
        )
        sampled = None
        if capture_sampling:
            sampled = self.sampling.sample(logits, enable_trace=False, tt_out_tok=self.trace.token_input)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self.counters.synchronizations += 1
        self._copy_position_to_trace_input(start_pos, setup=True)

        self.trace.trace_id = trace_id
        self.trace.logits = logits
        self.trace.sampled = sampled
        self.trace.sampling_mode_key = sampling_mode_key
        self.trace.capture_sampling = capture_sampling
        self.counters.model_trace_captures += 1

        if not capture_sampling:
            before_sampling_traces = sum(1 for slot in self.sampling._trace_states.values() if slot["id"] is not None)
            self.sampling.sample(logits, enable_trace=True, tt_out_tok=self.trace.token_input, skip_precompile=True)
            after_sampling_traces = sum(1 for slot in self.sampling._trace_states.values() if slot["id"] is not None)
            if after_sampling_traces > before_sampling_traces:
                self.counters.sampling_trace_captures += 1
            ttnn.synchronize_device(self.mesh_device)
            self.counters.synchronizations += 1

        self._copy_token_to_trace_input(token_id, teacher_forcing=False)

    def _release_decode_trace(self) -> None:
        if self.trace.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace.trace_id)
        self.sampling.reset_trace()
        self.trace = DecodeTraceState()

    def _decode_next_token_traced(self, *, readback: bool = True, return_device: bool = False) -> int | Any | None:
        if self.trace.trace_id is None or self.trace.token_input is None:
            raise RuntimeError("decode trace has not been captured")
        ttnn.execute_trace(self.mesh_device, self.trace.trace_id, cq_id=0, blocking=False)
        self.counters.model_trace_replays += 1
        self.counters.device_position_advances += 1
        if self.trace.capture_sampling:
            if self.trace.sampled is None:
                raise RuntimeError("token-out decode trace has no sampled output")
            sampled = self.trace.sampled
        else:
            if self.trace.logits is None:
                raise RuntimeError("split-sampling decode trace has no logits output")
            before_sampling_traces = sum(1 for slot in self.sampling._trace_states.values() if slot["id"] is not None)
            sampled = self.sampling.sample(
                self.trace.logits,
                enable_trace=True,
                tt_out_tok=self.trace.token_input,
                skip_precompile=True,
            )
            after_sampling_traces = sum(1 for slot in self.sampling._trace_states.values() if slot["id"] is not None)
            if after_sampling_traces > before_sampling_traces:
                self.counters.sampling_trace_captures += 1
        self.counters.device_token_feedbacks += 1
        if return_device:
            self.counters.no_readback_decode_steps += 1
            return sampled
        if not readback:
            self.counters.no_readback_decode_steps += 1
            return None
        return self._read_first_token(sampled[0] if isinstance(sampled, tuple) else sampled)

    def _sampling_mode_key(self) -> tuple[bool, bool, bool]:
        return (
            bool(self.sampling._penalties_active),
            bool(getattr(self.sampling, "_log_probs_active", False)),
            bool(self.sampling.tt_sampling.force_argmax_sampling),
        )

    def _precompile_sampling_trace_variants(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor) -> None:
        """Warm sampler program variants before a model trace is active.

        vLLM sampling tests can switch between plain token sampling, top-k
        logprobs, and penalty modes after the model decode trace has already
        been captured. Those modes use separate sampling trace keys. Compiling
        their op sequences here lets later trace captures use program-cache hits
        and avoids device-buffer allocation while the model trace is active.
        """

        if self._sampling_trace_variants_precompiled:
            return

        warm_sampler = SamplingGenerator(
            args=_SamplingArgs(self.model),
            mesh_device=self.mesh_device,
            tt_ccl=None,
            enable_internal_trace=False,
        )
        variants = (
            SamplingParams(temperature=0.0, top_k=1, top_p=0.0, seed=None, enable_log_probs=False, num_logprobs=0),
            SamplingParams(temperature=0.0, top_k=1, top_p=0.0, seed=None, enable_log_probs=True, num_logprobs=10),
            SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=0.0,
                presence_penalty=0.1,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=None,
                enable_log_probs=False,
                num_logprobs=0,
            ),
            SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=0.0,
                presence_penalty=0.1,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=None,
                enable_log_probs=True,
                num_logprobs=10,
            ),
        )
        dummy_tokens = torch.zeros((1, 1), dtype=torch.int64)
        for params in variants:
            formatted = format_sampling_params(params, warm_sampler.tt_sampling.max_batch_size)
            warm_sampler.reset_sampling_params(formatted)
            if warm_sampler._penalties_active:
                warm_sampler.reset_prompt_tokens(dummy_tokens)
                warm_sampler.reset_output_state(dummy_tokens)
            warm_sampler.sample(logits, enable_trace=False, tt_out_tok=tt_out_tok)
            self.counters.sampling_trace_variant_precompiles += 1
        self._sampling_trace_variants_precompiled = True

    def _reset_sampling_params(self, params: SamplingParams) -> None:
        formatted = format_sampling_params(params, self.sampling.tt_sampling.max_batch_size)
        self.sampling.reset_sampling_params(formatted)
        self.sampling.reset_prompt_tokens(torch.zeros((1, 1), dtype=torch.int64))
        self.sampling.reset_output_state(torch.zeros((1, 1), dtype=torch.int64))
        self.sampling.seed_manager.reset_seed(formatted.seed, list(range(self.sampling.tt_sampling.max_batch_size)))
        self.sampling.seed_manager.get_new_values(list(range(self.sampling.tt_sampling.max_batch_size)))

    def _coerce_page_table(self, page_table: torch.Tensor | ttnn.Tensor | None) -> ttnn.Tensor:
        if page_table is None:
            if self.page_table is None:
                raise ValueError("page_table is required when the generator does not own a standalone cache")
            return self.page_table
        if isinstance(page_table, ttnn.Tensor):
            return page_table
        return ttnn.from_torch(
            page_table.to(torch.int32).contiguous(),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _make_token_input_device(self, token_id: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            _token_buffer(token_id),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _make_current_pos_device(self, pos: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _copy_token_to_trace_input(self, token_id: int, *, teacher_forcing: bool) -> None:
        if self.trace.token_input is None:
            raise RuntimeError("decode trace token input is not initialized")
        host = ttnn.from_torch(
            _token_buffer(token_id),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.trace.token_input)
        if teacher_forcing:
            self.counters.token_teacher_forcing_refreshes += 1
        else:
            self.counters.token_setup_refreshes += 1

    def _copy_position_to_trace_input(self, pos: int, *, setup: bool) -> None:
        if self.trace.current_pos is None:
            raise RuntimeError("decode trace current_pos input is not initialized")
        host = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self.trace.current_pos)
        if setup:
            self.counters.position_setup_refreshes += 1
        else:
            self.counters.position_steady_state_refreshes += 1

    def _read_first_token(self, tt_tokens: ttnn.Tensor) -> int:
        device_tokens = ttnn.get_device_tensors(tt_tokens)
        host_tensor = device_tokens[0].cpu(blocking=True, cq_id=0)
        torch_tokens = ttnn.to_torch(host_tensor).reshape(-1)
        self.counters.sampled_token_readbacks += 1
        return int(torch_tokens[0].item())

    def _record_perf(self, start_s: float, first_token_s: float, last_decode_s: float, token_count: int) -> None:
        elapsed_s = max(time.perf_counter() - start_s, 0.0)
        ttft_s = max(first_token_s - start_s, 0.0)
        decode_tokens = max(token_count - 1, 0)
        decode_elapsed_s = max(last_decode_s - first_token_s, 0.0) if decode_tokens else 0.0
        self._last_perf = {
            "elapsed_s": elapsed_s,
            "ttft_ms": ttft_s * 1000.0,
            "decode_tokens": float(decode_tokens),
            "decode_elapsed_s": decode_elapsed_s,
            "decode_t/s/u": (decode_tokens / decode_elapsed_s) if decode_elapsed_s > 0 else 0.0,
            "e2e_t/s/u": (token_count / elapsed_s) if elapsed_s > 0 else 0.0,
        }


def build_generator(model_dir, mesh_device, **kwargs) -> Phi35MiniGenerator:
    return Phi35MiniGenerator(model_dir=model_dir, mesh_device=mesh_device, **kwargs)


def _pad_tokens(tokens: torch.Tensor, block_size: int, pad_token_id: int) -> torch.Tensor:
    seq_len = int(tokens.shape[-1])
    padded_len = max(block_size, math.ceil(seq_len / block_size) * block_size)
    if padded_len == seq_len:
        return tokens.to(torch.int64).contiguous()
    out = torch.full((tokens.shape[0], padded_len), pad_token_id, dtype=torch.int64)
    out[:, :seq_len] = tokens.to(torch.int64)
    return out


def _token_buffer(token_id: int) -> torch.Tensor:
    buf = torch.zeros((1, 1, 1, SAMPLING_BATCH_SIZE), dtype=torch.uint32)
    buf[0, 0, 0, 0] = int(token_id)
    return buf


def _is_eos(token_id: int, eos_token_id) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple, set)):
        return int(token_id) in {int(x) for x in eos_token_id}
    return int(token_id) == int(eos_token_id)


def _active_decode_slots(start_pos: torch.Tensor) -> list[int]:
    flat = torch.as_tensor(start_pos).reshape(-1)
    return [idx for idx, pos in enumerate(flat.tolist()) if int(pos) >= 0]
