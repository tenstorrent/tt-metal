# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        self.host_sampling_compat = host_sampling_compat
        self.kv_cache = self.model.init_paged_kv_cache()
        self.page_table = self.model.make_identity_page_table(batch_size=1)
        self.page_table_generation = 0
        self.sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.model.padded_vocab_size,
                mesh_device=mesh_device,
                max_batch_size=1,
                max_top_k=32,
                allow_force_argmax=False,
                pad_to_power_of_2=True,
                ag_topology=ttnn.Topology.Linear,
            )
        )
        self.force_argmax_sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.model.padded_vocab_size,
                mesh_device=mesh_device,
                max_batch_size=1,
                max_top_k=32,
                allow_force_argmax=True,
                pad_to_power_of_2=True,
                ag_topology=ttnn.Topology.Linear,
            )
        )
        self.sampler.load_device_buffers()
        self.force_argmax_sampler.load_device_buffers()
        self._sampling_trace_id: int | None = None
        self._sampling_trace_output = None
        self._sampling_params = self._make_sampling_params(batch_size=1, top_k=1, top_p=0.0, temperature=1.0)
        self._sampling_prewarm_output = ttnn.from_torch(
            torch.zeros((1, 1, 1, 1), dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
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
        self.page_table = self.model.make_identity_page_table(batch_size=1)
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

    def _capture_sampling_trace(
        self,
        logits: ttnn.Tensor,
        *,
        tt_out_tok: ttnn.Tensor,
        skip_precompile: bool = False,
        force_argmax: bool = True,
    ):
        sampler = self.force_argmax_sampler if force_argmax else self.sampler
        if not skip_precompile:
            if force_argmax:
                self._sample_force_argmax_to_output(logits, tt_out_tok=tt_out_tok)
            else:
                k, p, temp = self._sampling_params
                sampler.decode_forward(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)
            ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        if force_argmax:
            output = self._sample_force_argmax_to_output(logits, tt_out_tok=tt_out_tok)
        else:
            k, p, temp = self._sampling_params
            output = sampler.decode_forward(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(self.mesh_device)
        self._sampling_trace_id = trace_id
        self._sampling_trace_output = output
        return output

    def _sample_force_argmax_to_output(self, logits: ttnn.Tensor, *, tt_out_tok: ttnn.Tensor):
        sampled, log_probs = self.force_argmax_sampler.decode_forward(logits, tt_out_tok=None)
        sampled_for_copy = sampled
        if list(sampled.shape) != list(tt_out_tok.shape):
            sampled_for_copy = ttnn.reshape(sampled, list(tt_out_tok.shape))
        ttnn.copy(sampled_for_copy, tt_out_tok)
        return tt_out_tok, log_probs

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
            k, p, temp = self._sampling_params
            return self.sampler.decode_forward(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)
        if self._sampling_trace_id is None:
            return self._capture_sampling_trace(
                logits,
                tt_out_tok=tt_out_tok,
                skip_precompile=skip_trace_precompile,
                force_argmax=False,
            )
        ttnn.execute_trace(self.mesh_device, self._sampling_trace_id, cq_id=0, blocking=False)
        return self._sampling_trace_output

    def prepare_token_out_decode(
        self,
        *,
        first_input_token: int,
        start_pos: int,
        prompt_len: int,
    ) -> int:
        token_input = torch.tensor([[first_input_token]], dtype=torch.long)
        pos = torch.tensor([start_pos], dtype=torch.int32)
        self.model.reset_decode_trace_state(
            token_input=token_input,
            start_pos=pos,
            page_table=self.page_table,
            prompt_lens=[prompt_len],
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
        self.sample_logits_on_device(
            warm_logits,
            tt_out_tok=self._sampling_prewarm_output,
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
        token = int(ttnn.to_torch(ttnn.get_device_tensors(sampled)[0]).reshape(-1)[0].item())
        self.model.trace_state.counters["readbacks"] += 1
        return token

    def decode_next_token_on_device(self):
        logits = self.model.execute_decode_trace()
        return self.sample_logits_on_device(
            logits,
            tt_out_tok=self.model.trace_state.token_input,
            enable_trace=True,
            force_argmax=False,
        )

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
