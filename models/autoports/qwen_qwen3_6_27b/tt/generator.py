# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Readiness and serving generator for the TP4 Qwen3.6-27B full model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_REVISION
from models.autoports.qwen_qwen3_6_27b.tt.model import Qwen36Model
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.readiness_check.contract import Generator as ReadinessGenerator
from models.common.readiness_check.contract import NextInputFn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params


@dataclass
class SamplingArgs:
    vocab_size: int
    padded_vocab_size: int
    max_batch_size: int
    cluster_shape: tuple[int, int] = (1, 4)
    max_top_k: int = 32
    sampling_all_gather_axis: int = 1
    sampling_dp: int = 1
    use_topk_logprobs: bool = False
    force_argmax_active_rows: int | None = None
    pad_logits_to_power_of_2: bool = True
    local_topk_num_chunks: int = 2
    mask_invalid_vocab: bool = True
    model_config: dict = field(
        default_factory=lambda: {
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": True,
                "num_links": 1,
                "chunks_per_sync": 10,
                "num_workers_per_link": 1,
                "topology": ttnn.Topology.Linear,
            }
        }
    )


class Qwen36Generator(ReadinessGenerator):
    """Two-level generator with externally owned state and traced token-out."""

    MAX_PREFILL_TOKENS = 262144

    def __init__(
        self,
        *,
        model: Qwen36Model,
        mesh_device,
        tokenizer,
        host_sampling_compatibility=False,
        force_argmax_greedy=False,
        pad_sampling_logits_to_power_of_2=True,
    ):
        self.model, self.mesh_device, self.tokenizer = model, mesh_device, tokenizer
        self.host_sampling_compatibility = bool(host_sampling_compatibility)
        self.kv_cache = model.kv_cache
        self.page_table_host = model.allocate_page_table()
        self._page_table = self._upload(self.page_table_host, dtype=ttnn.int32)
        sampling_args = SamplingArgs(
            model.vocab_size,
            model.padded_vocab_size,
            model.batch,
            force_argmax_active_rows=model.batch,
            pad_logits_to_power_of_2=bool(pad_sampling_logits_to_power_of_2),
        )
        sampling_args.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = bool(force_argmax_greedy)
        self.sampling = SamplingGenerator(
            args=sampling_args,
            mesh_device=mesh_device,
            tt_ccl=get_tt_ccl(mesh_device),
        )
        greedy = SamplingParams(temperature=1.0, top_k=1, top_p=0.0)
        self.sampling.reset_sampling_params(format_sampling_params(greedy, 32))
        self._decode_trace_id = None
        self._trace_token = None
        self._trace_position = None
        self._trace_active_mask = None
        self._trace_active_state_mask = None
        self._trace_page_table = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_cache_backups = None
        self._compat_trace_id = None
        self._compat_token = None
        self._compat_position = None
        self._compat_logits = None
        self.trace_counters = {
            "replays": 0,
            "token_host_refreshes": 0,
            "position_host_refreshes": 0,
            "page_table_refreshes": 0,
            "readbacks": 0,
        }
        self._slots_requiring_prefill = set()

    def _upload(self, value, *, dtype=ttnn.uint32):
        torch_dtype = (
            torch.bfloat16 if dtype == ttnn.bfloat16 else (torch.int32 if dtype == ttnn.int32 else torch.uint32)
        )
        value = value.to(torch_dtype).contiguous()
        return ttnn.from_torch(
            value,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _to_host_logits(self, logits):
        self.trace_counters["readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))[
            ..., : self.model.vocab_size
        ]

    def _check_state(self, page_table, kv_cache):
        if kv_cache is not None and kv_cache is not self.kv_cache:
            self.model.bind_kv_cache(kv_cache)
            self.kv_cache = kv_cache
        return self._page_table if page_table is None else page_table

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: List[int],
        return_all_logits: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        page_table = self._check_state(page_table, kv_cache)
        batch, physical_len = tokens.shape
        if batch != self.model.batch or len(prompt_lens) != batch:
            raise ValueError("tokens and prompt_lens must match the generator's fixed slot count")
        supported_prefill = min(self.model.max_context, self.MAX_PREFILL_TOKENS)
        if physical_len < 1 or physical_len > supported_prefill:
            raise ValueError(f"physical prompt extent is outside supported prefill [1, {supported_prefill}]")
        if any(length < 0 or length > physical_len for length in prompt_lens):
            raise ValueError("logical prompt length is outside the supported context")
        if not any(prompt_lens):
            raise ValueError("prefill requires at least one active prompt")
        if return_all_logits and physical_len > self.model.PREFILL_STACK_CHUNK_SIZE:
            raise ValueError(
                "return_all_logits is supported only through "
                f"{self.model.PREFILL_STACK_CHUNK_SIZE} tokens; long streaming prefill returns terminal prompt logits"
            )
        # Public lengths remain logical. Padding is masked by per-row positions;
        # returned logits are sliced back to the requested logical extent.
        token_tt = self._upload(tokens, dtype=ttnn.uint32)
        positions = torch.arange(physical_len, dtype=torch.int64).to(torch.uint32).repeat(batch, 1)
        position_tt = self._upload(positions, dtype=ttnn.uint32)
        # Linear attention scans in 64-token chunks. Metadata is chunk-local:
        # a row whose prompt ended in an earlier chunk gets a zero-length mask
        # and selectors that preserve the incoming four-token conv state.
        sequence_mask_tt, selector_tensors = [], []
        prompt_lens_tensor = torch.tensor(prompt_lens)
        for start in range(0, physical_len, 64):
            chunk_len = min(64, physical_len - start)
            active_len = torch.clamp(prompt_lens_tensor - start, min=0, max=chunk_len)
            mask = (torch.arange(chunk_len).reshape(1, chunk_len) < active_len.reshape(batch, 1)).to(torch.bfloat16)
            sequence_mask_tt.append(self._upload(mask, dtype=ttnn.bfloat16))
            chunk_selectors = []
            for lane in range(4):
                selector = torch.zeros((batch, chunk_len + 4), dtype=torch.bfloat16)
                selector[torch.arange(batch), active_len + lane] = 1
                chunk_selectors.append(self._upload(selector, dtype=ttnn.bfloat16))
            selector_tensors.append(chunk_selectors)
        cache_page_table = page_table
        temporary_cache_page_table = None
        if any(length == 0 for length in prompt_lens):
            if page_table is not self._page_table:
                raise ValueError("inactive-slot prefill requires the generator-owned page table")
            effective = self.page_table_host.clone()
            for slot, length in enumerate(prompt_lens):
                if length == 0:
                    effective[slot].fill_(-1)
            temporary_cache_page_table = self._upload(effective, dtype=ttnn.int32)
            cache_page_table = temporary_cache_page_table
        host = None
        logits = None
        try:
            logits = self.model.prefill_forward(
                token_ids=token_tt,
                page_table=page_table,
                current_positions=position_tt,
                sequence_mask=sequence_mask_tt,
                conv_state_selectors=selector_tensors,
                logit_positions=None if return_all_logits else prompt_lens,
                cache_page_table=cache_page_table,
            )
            host = self._to_host_logits(logits)
        finally:
            # Request metadata is consumed asynchronously by every decoder
            # layer.  Fence before clearing aliases and explicitly releasing
            # the O(ceil(S/64)) uploaded masks/selectors; otherwise a maximum
            # context request leaves 15,040 device tensors model-owned.
            ttnn.synchronize_device(self.mesh_device)
            self.model.clear_prefill_request_state()
            for tensor in sequence_mask_tt:
                ttnn.deallocate(tensor)
            for chunk in selector_tensors:
                for tensor in chunk:
                    ttnn.deallocate(tensor)
            if logits is not None:
                ttnn.deallocate(logits)
            ttnn.deallocate(token_tt)
            ttnn.deallocate(position_tt)
            if temporary_cache_page_table is not None:
                ttnn.deallocate(temporary_cache_page_table)
        self._slots_requiring_prefill.difference_update(slot for slot, length in enumerate(prompt_lens) if length > 0)
        if return_all_logits:
            return host.reshape(batch, physical_len, -1)
        return host.reshape(batch, 1, -1)

    def decode_forward(
        self, tokens: torch.Tensor, start_pos: torch.Tensor, *, page_table, kv_cache, active_mask=None, **kwargs: Any
    ):
        page_table = self._check_state(page_table, kv_cache)
        requested_active = (
            torch.ones(self.model.batch, dtype=torch.bool)
            if active_mask is None
            else torch.as_tensor(active_mask).bool()
        )
        blocked = sorted(slot for slot in self._slots_requiring_prefill if requested_active[slot])
        if blocked:
            raise RuntimeError(f"reset slots require prefill before decode: {blocked}")
        token_tt = self._upload(tokens, dtype=ttnn.uint32)
        position_tt = self._upload(start_pos, dtype=ttnn.uint32)
        if active_mask is None:
            active_mask = torch.ones(self.model.batch, dtype=torch.bfloat16)
        active_mask_tt = self._upload(torch.as_tensor(active_mask), dtype=ttnn.bfloat16)
        logits = None
        try:
            logits = self.model.decode_forward(
                token_ids=token_tt, page_table=page_table, current_positions=position_tt, active_mask=active_mask_tt
            )
            return self._to_host_logits(logits).reshape(self.model.batch, -1)
        finally:
            # Eager low-level decode owns these request tensors.  The host
            # logits boundary completes their consumers; do not retain them
            # as if they were the persistent split-trace inputs.
            ttnn.synchronize_device(self.mesh_device)
            if logits is not None:
                ttnn.deallocate(logits)
            ttnn.deallocate(token_tt)
            ttnn.deallocate(position_tt)
            ttnn.deallocate(active_mask_tt)

    def _capture_token_out_trace(self, first_token, start_pos, active_mask=None, page_table=None):
        tokens = torch.zeros((1, 1, 1, self.sampling.tt_sampling.max_batch_size), dtype=torch.uint32)
        first_tokens = torch.as_tensor(first_token, dtype=torch.uint32).reshape(-1)
        if first_tokens.numel() == 1:
            first_tokens = first_tokens.repeat(self.model.batch)
        if first_tokens.numel() != self.model.batch:
            raise ValueError("first_token must be scalar or have one value per fixed slot")
        tokens[0, 0, 0, : self.model.batch] = first_tokens
        position_values = torch.as_tensor(start_pos, dtype=torch.uint32).reshape(-1)
        if position_values.numel() == 1:
            position_values = position_values.repeat(self.model.batch)
        if position_values.numel() != self.model.batch:
            raise ValueError("start_pos must be scalar or have one value per fixed slot")
        positions = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
        positions[0, 0, 0, : self.model.batch] = position_values
        active = (
            torch.ones(self.model.batch, dtype=torch.uint32)
            if active_mask is None
            else torch.as_tensor(active_mask, dtype=torch.uint32)
        )
        if active.numel() != self.model.batch:
            raise ValueError("active_mask must have one value per fixed slot")
        self._trace_token = self._upload(tokens, dtype=ttnn.uint32)
        self._trace_position = ttnn.from_torch(
            positions,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        active_positions = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
        active_positions[0, 0, 0, : self.model.batch] = active
        self._trace_active_mask = ttnn.from_torch(
            active_positions,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._trace_active_state_mask = self._upload(active, dtype=ttnn.bfloat16)
        self._trace_page_table = self._page_table if page_table is None else page_table

        # Warmup and capture execute the stateful decoder graph. Preserve all
        # request state first, and keep backups allocated for the lifetime of
        # the trace so no post-capture deallocation can disturb traced buffer
        # addresses. Optimized high-level generation is batch 1, where this
        # temporary capacity is part of the measured trace envelope.
        self._trace_cache_backups = [[ttnn.clone(tensor) for tensor in pair] for pair in self.kv_cache]
        # Cache restoration occurs after both traces are live. Warm the exact
        # backup-to-cache copy variants now so that restore cannot compile or
        # allocate ordinary buffers while trace addresses must remain stable.
        for pair, backups in zip(self.kv_cache, self._trace_cache_backups):
            for tensor, backup in zip(pair, backups):
                ttnn.copy(backup, tensor)
        ttnn.synchronize_device(self.mesh_device)

        # Exact warmup includes model, device position advance, and sampler.
        logits = self.model.decode_forward(
            token_ids=self._trace_token,
            page_table=self._trace_page_table,
            current_positions=self._trace_position,
            active_mask=self._trace_active_state_mask,
        )
        ttnn.add(self._trace_position, self._trace_active_mask, output_tensor=self._trace_position)
        logits = self._sampling_logits(logits)
        self.sampling.sample(logits, enable_trace=False, tt_out_tok=self._trace_token)
        ttnn.synchronize_device(self.mesh_device)

        self._decode_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_logits = self.model.decode_forward(
            token_ids=self._trace_token,
            page_table=self._trace_page_table,
            current_positions=self._trace_position,
            active_mask=self._trace_active_state_mask,
        )
        self._trace_logits = self._sampling_logits(self._trace_logits)
        ttnn.add(self._trace_position, self._trace_active_mask, output_tensor=self._trace_position)
        ttnn.end_trace_capture(self.mesh_device, self._decode_trace_id, cq_id=0)
        # Sampling is deliberately a separate trace and writes directly into
        # the persistent token consumed by the next model replay.
        # The exact sampler path was already compiled above, before the model
        # trace was captured.  Re-running SamplingGenerator's default
        # precompile here would allocate/deallocate intermediates while a live
        # model trace holds device addresses.  Capture directly so temporary
        # buffers are owned by the sampling trace allocator.
        self._trace_sampled = self.sampling.capture_trace(
            self._trace_logits, tt_out_tok=self._trace_token, skip_precompile=True
        )
        for pair, backups in zip(self.kv_cache, self._trace_cache_backups):
            for tensor, backup in zip(pair, backups):
                ttnn.copy(backup, tensor)

    def _sampling_logits(self, logits):
        """Return sampler-ready logits with the proven fixed-slot row shape."""
        return logits

    def _read_sampled_token(self):
        self.trace_counters["readbacks"] += 1
        value = ttnn.to_torch(ttnn.get_device_tensors(self._trace_token)[0])
        return int(value.reshape(-1)[0])

    def setup_token_out_decode(
        self,
        tokens,
        positions,
        *,
        page_table=None,
        kv_cache=None,
        active_mask=None,
        sampling_params: SamplingParams | None = None,
    ):
        """Public serving boundary for persistent split-trace token-out decode.

        State is explicit at request setup; subsequent steps preserve tokens,
        positions, caches and an unchanged page table entirely on device.
        """
        active_values = (
            torch.ones(self.model.batch, dtype=torch.bool)
            if active_mask is None
            else torch.as_tensor(active_mask).bool()
        )
        blocked = sorted(slot for slot in self._slots_requiring_prefill if active_values[slot])
        if blocked:
            raise RuntimeError(f"reset slots require prefill before decode: {blocked}")
        self._release_traces()
        resolved_page_table = self._check_state(page_table, kv_cache)
        if isinstance(page_table, torch.Tensor):
            self.refresh_page_table(page_table)
            resolved_page_table = self._page_table
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=1.0, top_k=1, top_p=0.0)
        self.sampling.reset_sampling_params(format_sampling_params(sampling_params, 32))
        self._capture_token_out_trace(tokens, positions, active_mask=active_mask, page_table=resolved_page_table)
        self._seed_token_out_trace(tokens, positions)
        return {
            "token": self._trace_token,
            "position": self._trace_position,
            "page_table": self._trace_page_table,
            "kv_cache": self.kv_cache,
            "active_mask": self._trace_active_mask,
        }

    def token_out_decode_step(self, *, page_table=None, readback: bool = False):
        """Replay model plus common sampler traces without a logits boundary."""
        if self._decode_trace_id is None:
            raise RuntimeError("call setup_token_out_decode before token_out_decode_step")
        if page_table is not None:
            if not isinstance(page_table, torch.Tensor):
                raise TypeError("changed page tables must be supplied as host torch tensors")
            if self._trace_page_table is not self._page_table:
                raise RuntimeError(
                    "changed page-table refresh is unsupported for an externally owned captured device tensor; "
                    "recapture with a host page table or omit page_table for unchanged ownership"
                )
            self.refresh_page_table(page_table)
        ttnn.execute_trace(self.mesh_device, self._decode_trace_id, cq_id=0, blocking=False)
        self.sampling.sample(self._trace_logits, enable_trace=True, tt_out_tok=self._trace_token)
        self.trace_counters["replays"] += 1
        if not readback:
            return self._trace_token
        self.trace_counters["readbacks"] += 1
        values = ttnn.to_torch(ttnn.get_device_tensors(self._trace_token)[0]).reshape(-1)
        return [int(value) for value in values[: self.model.batch]]

    def _seed_token_out_trace(self, tokens, positions):
        """Seed stable trace inputs after real prefill, without rebuilding them."""
        token_values = torch.as_tensor(tokens, dtype=torch.uint32).reshape(-1)
        if token_values.numel() == 1:
            token_values = token_values.repeat(self.model.batch)
        position_values = torch.as_tensor(positions, dtype=torch.uint32).reshape(-1)
        if position_values.numel() == 1:
            position_values = position_values.repeat(self.model.batch)
        token_host = torch.zeros((1, 1, 1, self._trace_token.shape[-1]), dtype=torch.uint32)
        token_host[0, 0, 0, : self.model.batch] = token_values
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(token_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), self._trace_token
        )
        position_host = torch.zeros((1, 1, 1, 32), dtype=torch.uint32)
        position_host[0, 0, 0, : self.model.batch] = position_values
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(position_host, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT), self._trace_position
        )

    def _capture_compatibility_trace(self, token: int, position: int):
        """Capture traced logits-only decode for explicit host-sampling tests."""
        self._compat_token = self._upload(torch.tensor([[token]], dtype=torch.uint32), dtype=ttnn.uint32)
        self._compat_position = self._upload(torch.tensor([position], dtype=torch.uint32), dtype=ttnn.uint32)
        # Exact warm compile before capture. Preserve every state tensor so the
        # warm call is not a hidden extra teacher-forcing step, especially for
        # linear-attention recurrent/conv state.
        cache_backups = [[ttnn.clone(tensor) for tensor in pair] for pair in self.kv_cache]
        self.model.decode_forward(
            token_ids=self._compat_token, page_table=self._page_table, current_positions=self._compat_position
        )
        ttnn.synchronize_device(self.mesh_device)
        for pair, backups in zip(self.kv_cache, cache_backups):
            for tensor, backup in zip(pair, backups):
                ttnn.copy(backup, tensor)
                ttnn.deallocate(backup)
        self._compat_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._compat_logits = self.model.decode_forward(
            token_ids=self._compat_token, page_table=self._page_table, current_positions=self._compat_position
        )
        ttnn.end_trace_capture(self.mesh_device, self._compat_trace_id, cq_id=0)

    def _compatibility_decode(self, token: int, position: int):
        if self._compat_trace_id is None:
            self._capture_compatibility_trace(token, position)
        else:
            token_host = ttnn.from_torch(
                torch.tensor([[token]], dtype=torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            pos_host = ttnn.from_torch(
                torch.tensor([position], dtype=torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(token_host, self._compat_token)
            ttnn.copy_host_to_device_tensor(pos_host, self._compat_position)
            self.trace_counters["token_host_refreshes"] += 1
            self.trace_counters["position_host_refreshes"] += 1
        ttnn.execute_trace(self.mesh_device, self._compat_trace_id, cq_id=0, blocking=True)
        self.trace_counters["replays"] += 1
        return self._to_host_logits(self._compat_logits).reshape(self.model.batch, -1)

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        host_sampling_compatibility: bool | None = None,
        **kwargs: Any,
    ) -> List[int]:
        if not enable_trace:
            raise ValueError("Qwen3.6 readiness decode requires enable_trace=True")
        if self.model.batch != 1:
            raise ValueError("high-level generate is the batch-1 API; use low-level fixed slots for mixed batches")
        host_mode = (
            self.host_sampling_compatibility if host_sampling_compatibility is None else host_sampling_compatibility
        )
        self.reset()
        tokens = torch.tensor([prompt_token_ids], dtype=torch.long)
        prefill_logits = self.prefill_forward(
            tokens, page_table=self._page_table, kv_cache=self.kv_cache, prompt_lens=[len(prompt_token_ids)]
        )
        predicted = int(torch.argmax(prefill_logits[0, 0]).item())
        output = [predicted]
        if max_new_tokens <= 1:
            return output[:max_new_tokens]
        feed = next_input(0, predicted) if next_input else predicted

        if host_mode:
            # Explicit compatibility boundary for tests that require host-side
            # teacher forcing/sampling; never used for optimized token-out perf.
            for step in range(1, max_new_tokens):
                logits = self._compatibility_decode(feed, len(prompt_token_ids) + step - 1)
                predicted = int(torch.argmax(logits[0]).item())
                output.append(predicted)
                feed = next_input(step, predicted) if next_input else predicted
            return output

        if next_input is not None:
            raise ValueError(
                "teacher forcing requires host_sampling_compatibility=True; optimized autoregressive mode is device-owned"
            )
        self._capture_token_out_trace(feed, len(prompt_token_ids))
        self._seed_token_out_trace(feed, len(prompt_token_ids))
        for _ in range(1, max_new_tokens):
            self.token_out_decode_step(readback=False)
            output.append(self._read_sampled_token())
        return output

    def refresh_page_table(self, page_table: torch.Tensor):
        if torch.equal(page_table, self.page_table_host):
            return
        host = ttnn.from_torch(page_table.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self._page_table)
        self.page_table_host = page_table.clone()
        self.trace_counters["page_table_refreshes"] += 1

    def reset(self) -> None:
        self._release_traces()
        self.model.reset_cache()
        self.trace_counters.update({key: 0 for key in self.trace_counters})
        self._slots_requiring_prefill.clear()

    def reset_slots(self, slots) -> None:
        """Invalidate selected request slots without disturbing live peers."""
        slots = sorted({int(slot) for slot in slots})
        self._release_traces()
        self.model.reset_slots(slots)
        self._slots_requiring_prefill.update(slots)

    def _release_traces(self):
        """Release model and sampler traces before rebinding request state."""
        # token_out_decode_step deliberately submits replay nonblocking.  A
        # request may reset, reconfigure, or tear down immediately afterward;
        # fence outstanding model/sampler work before releasing trace buffers
        # or deallocating their persistent cache snapshots.
        ttnn.synchronize_device(self.mesh_device)
        if self._decode_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._decode_trace_id)
        if self._compat_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self._compat_trace_id)
        self.sampling.reset_trace()
        self._decode_trace_id = self._compat_trace_id = None

        # These tensors are allocated before capture and remain ordinary DRAM
        # allocations; releasing a trace drops captured programs/intermediates
        # but does not release their persistent inputs.  Deallocate only the
        # generator-owned tensors after both model and sampler traces have
        # released every alias.  Identity deduplication makes this safe if a
        # future compatibility path deliberately shares a persistent input.
        owned_names = (
            "_trace_token",
            "_trace_position",
            "_trace_active_mask",
            "_trace_active_state_mask",
            "_compat_token",
            "_compat_position",
        )
        deallocated = set()
        for name in owned_names:
            tensor = getattr(self, name)
            if tensor is not None and id(tensor) not in deallocated:
                ttnn.deallocate(tensor)
                deallocated.add(id(tensor))
            setattr(self, name, None)

        # The page table is generator- or caller-owned, while logits and
        # sampled outputs are trace-owned.  Clear aliases without freeing them
        # as ordinary tensors.
        self._trace_page_table = None
        self._trace_logits = self._trace_sampled = None
        if self._trace_cache_backups is not None:
            for pair in self._trace_cache_backups:
                for tensor in pair:
                    ttnn.deallocate(tensor)
        self._trace_cache_backups = None
        self._compat_logits = None

    def teardown(self) -> None:
        self._release_traces()


def build_generator(model_dir, mesh_device, **kwargs):
    snapshot = Path(
        kwargs.pop("snapshot", Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION)
    )
    host_sampling_compatibility = kwargs.pop("host_sampling_compatibility", False)
    force_argmax_greedy = kwargs.pop("force_argmax_greedy", False)
    pad_sampling_logits_to_power_of_2 = kwargs.pop("pad_sampling_logits_to_power_of_2", True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = Qwen36Model.from_pretrained(mesh_device=mesh_device, snapshot=snapshot, **kwargs)
    return Qwen36Generator(
        model=model,
        mesh_device=mesh_device,
        tokenizer=tokenizer,
        host_sampling_compatibility=host_sampling_compatibility,
        force_argmax_greedy=force_argmax_greedy,
        pad_sampling_logits_to_power_of_2=pad_sampling_logits_to_power_of_2,
    )


__all__ = ["Qwen36Generator", "build_generator"]
