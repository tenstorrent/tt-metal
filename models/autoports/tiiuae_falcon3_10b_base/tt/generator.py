# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the optimized TP4 Falcon3-10B model."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import ttnn
from models.autoports.tiiuae_falcon3_10b_base.tt.model import (
    DEFAULT_PREFILL_CHUNK_SIZE,
    HF_MODEL_ID,
    HF_REVISION,
    Falcon3Model,
)
from models.common.readiness_check.contract import Generator, NextInputFn
from models.tt_transformers.tt.common import copy_host_to_device


def _round_up(value: int, multiple: int) -> int:
    return multiple * math.ceil(value / multiple)


def _first_device_to_torch(tensor) -> torch.Tensor:
    shards = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(shards[0] if shards else tensor)


class Falcon3Generator(Generator):
    """Caller-owned low-level caches plus traced device-feedback generation."""

    def __init__(self, model: Falcon3Model, tokenizer, *, prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE):
        if prefill_chunk_size < 128 or prefill_chunk_size % model.page_block_size:
            raise ValueError("prefill_chunk_size must be a page-aligned multiple of at least 128")
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.batch = model.max_batch_size
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.pages_per_user = math.ceil(model.max_cache_len / model.page_block_size)
        self.num_blocks = self.batch * self.pages_per_user

        self._kv_cache = None
        self._page_table_host: torch.Tensor | None = None
        self._trace_model_id = None
        self._trace_sampling_id = None
        self._trace_inputs = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_kv_cache = None
        self._trace_page_table_snapshot = None
        self._trace_active_batch = None
        self._decode_warm_key = None
        self._sampling_params = None
        self._sampling_snapshot = None
        self._sampling_stochastic = False
        self.trace_stats = {
            "captures": 0,
            "replays": 0,
            "releases": 0,
            "decode_warmups": 0,
            "token_host_copies": 0,
            "token_device_copies": 0,
            "position_host_copies": 0,
            "rotary_position_host_copies": 0,
            "page_table_host_copies": 0,
            "sampling_param_host_copies": 0,
            "caller_token_readbacks": 0,
            "explicit_synchronizations": 0,
            "resets": 0,
            "rng_checkpoints": 0,
            "rng_restores": 0,
            "ccl_capture_epoch_resets": 0,
        }
        self._allocate_persistent_inputs()

    def _ensure_kv_cache(self):
        if self._kv_cache is None:
            self._kv_cache = self.model.allocate_kv_cache(paged=True, num_blocks=self.num_blocks)
        return self._kv_cache

    def _page_table_to_torch(self, page_table) -> torch.Tensor:
        if isinstance(page_table, torch.Tensor):
            host = page_table.detach().cpu().to(torch.int32)
        elif isinstance(page_table, ttnn.Tensor):
            host = _first_device_to_torch(page_table).to(torch.int32)
        else:
            raise TypeError("page_table must be a torch or TTNN tensor")
        if host.ndim != 2:
            raise ValueError(f"page_table must be rank two, got {tuple(host.shape)}")
        return host

    def _normalise_page_table(self, page_table, active_batch: int) -> torch.Tensor:
        host = self._page_table_to_torch(page_table)
        if host.shape[0] < active_batch or host.shape[0] > self.batch:
            raise ValueError("page table does not match the configured/active batch")
        if host.shape[1] < self.pages_per_user:
            host = torch.nn.functional.pad(host, (0, self.pages_per_user - host.shape[1]), value=-1)
        elif host.shape[1] > self.pages_per_user:
            # The common readiness runner may provide a wider generic table;
            # Falcon3 needs 1,024 32-token columns at the advertised 32K
            # context.
            host = host[:, : self.pages_per_user]
        if host.shape[0] < self.batch:
            host = torch.nn.functional.pad(host, (0, 0, 0, self.batch - host.shape[0]), value=-1)
        return host.contiguous()

    def _sdpa_rounded_page_count(self, token_count: int) -> int:
        """Return the physical mappings read by dynamic paged decode SDPA.

        The decode kernel rounds a sequence of at most eight 32-token tiles to
        the next power of two.  Longer sequences use eight-tile chunks.  It
        reads the whole rounded window before causal masking, so every rounded
        tail page must have a valid mapping even though it contains no live
        token yet.
        """
        if token_count < 1 or token_count > self.model.max_cache_len:
            raise ValueError("SDPA token count is outside the supported context")
        logical_pages = math.ceil(token_count / self.model.page_block_size)
        if logical_pages <= 8:
            return 1 << (logical_pages - 1).bit_length()
        return _round_up(logical_pages, 8)

    def _make_page_table(self, lengths: list[int]) -> torch.Tensor:
        if len(lengths) > self.batch:
            raise ValueError(f"{len(lengths)} prompts exceed configured batch {self.batch}")
        table = torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32)
        next_block = 0
        for user, length in enumerate(lengths):
            blocks = self._sdpa_rounded_page_count(int(length))
            if blocks > self.pages_per_user or next_block + blocks > self.num_blocks:
                raise ValueError("paged KV-cache capacity is insufficient for the requested prompts")
            table[user, :blocks] = torch.arange(next_block, next_block + blocks, dtype=torch.int32)
            next_block += blocks
        return table

    def _replicated_host_tensor(self, host: torch.Tensor, *, dtype):
        return ttnn.from_torch(
            host,
            device=None,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _replicated_device_tensor(self, host: torch.Tensor, *, dtype):
        return ttnn.from_torch(
            host,
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _allocate_persistent_inputs(self) -> None:
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        # Materialize the sampler's mutable seed buffer before any trace is
        # captured.  The checkpoint is a stable device allocation used to make
        # compile warm-up/capture observationally invisible to request RNG.
        self.model.sampler.load_device_buffers()
        self._sampling_seed_checkpoint = self._replicated_device_tensor(
            torch.zeros(32, dtype=torch.int32), dtype=ttnn.uint32
        )
        self._prefill_page_table = self._replicated_device_tensor(
            torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32), dtype=ttnn.int32
        )
        self._prefill_token_buffers = {
            length: self._replicated_device_tensor(
                torch.zeros((self.batch, length), dtype=torch.int32), dtype=ttnn.uint32
            )
            for length in range(128, self.prefill_chunk_size + 1, 128)
        }
        self._prefill_chunk_tables = {
            pages: self._replicated_device_tensor(
                torch.full((self.batch, pages), -1, dtype=torch.int32), dtype=ttnn.int32
            )
            for pages in range(1, self.prefill_chunk_size // self.model.page_block_size + 1)
        }
        self._prefill_sampled = ttnn.from_torch(
            torch.zeros((1, 1, 1, 32), dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        self._decode_trace_input_pool = (
            ttnn.from_torch(
                torch.zeros((1, 1, 1, 32), dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            self._replicated_device_tensor(torch.full((self.batch,), -1, dtype=torch.int32), dtype=ttnn.int32),
            self._replicated_device_tensor(torch.zeros((self.batch,), dtype=torch.int32), dtype=ttnn.uint32),
            self._replicated_device_tensor(
                torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32), dtype=ttnn.int32
            ),
        )

    def _copy_host(self, host: torch.Tensor, device, *, dtype) -> None:
        ttnn.copy_host_to_device_tensor(self._replicated_host_tensor(host, dtype=dtype), device)

    def _stage_prefill_chunk(self, tokens: torch.Tensor, page_host: torch.Tensor, *, start: int, length: int):
        active_batch, logical_width = tokens.shape
        token_host = torch.zeros((self.batch, length), dtype=torch.int32)
        source_end = min(logical_width, start + length)
        if source_end > start:
            token_host[:active_batch, : source_end - start] = tokens[:, start:source_end].to(torch.int32)
        token_device = self._prefill_token_buffers[length]
        self._copy_host(token_host, token_device, dtype=ttnn.uint32)
        chunk_table = None
        if start:
            first_page = start // self.model.page_block_size
            page_count = math.ceil(length / self.model.page_block_size)
            chunk_host = page_host[:, first_page : first_page + page_count].contiguous()
            chunk_table = self._prefill_chunk_tables[page_count]
            self._copy_host(chunk_host, chunk_table, dtype=ttnn.int32)
        return token_device, chunk_table

    def _synchronize(self) -> None:
        ttnn.synchronize_device(self.mesh_device)
        self.trace_stats["explicit_synchronizations"] += 1

    def _release_decode_traces_before_allocating_prefill(self) -> None:
        """Enter a safe allocator epoch without discarding staging buffers.

        TT-Metal records the addresses used by a captured trace and marks all
        later device allocations unsafe until every live trace is released.
        Prefill is intentionally eager and allocates transient outputs, so a
        decode trace from the preceding request cannot remain installed across
        this boundary.  Quiesce any nonblocking replay before releasing it;
        program-cache entries and the persistent input pool remain resident.
        """
        if self._trace_model_id is None and self._trace_sampling_id is None:
            return
        self._synchronize()
        self._release_decode_traces()

    def _sampled_to_torch(self, sampled) -> torch.Tensor:
        self.trace_stats["caller_token_readbacks"] += 1
        return _first_device_to_torch(sampled).reshape(-1)[: self.batch].to(torch.long)

    def _local_logits_to_torch(self, local_logits, *, valid_rows: int) -> torch.Tensor:
        return self.model.gather_logits_to_torch(local_logits, valid_rows=valid_rows)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache: Any,
        prompt_lens: list[int],
        return_all_logits: bool = False,
        sampling_mode: str = "host",
        **kwargs: Any,
    ):
        """Prefill arbitrary logical lengths with generator-owned padding/chunks."""
        if sampling_mode not in {"host", "device"}:
            raise ValueError("sampling_mode must be 'host' or 'device'")
        if sampling_mode == "device" and return_all_logits:
            raise ValueError("return_all_logits is incompatible with device sampling")
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [batch,seq], got {tuple(tokens.shape)}")
        active_batch, logical_width = (int(tokens.shape[0]), int(tokens.shape[1]))
        if active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active batch must be in [1,{self.batch}]")
        if len(prompt_lens) != active_batch or any(length < 1 or length > logical_width for length in prompt_lens):
            raise ValueError("prompt_lens must contain one valid logical length per input row")
        if max(prompt_lens) > self.model.max_cache_len:
            raise ValueError("prompt exceeds the supported context")

        self._release_decode_traces_before_allocating_prefill()
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        page_host = self._normalise_page_table(page_table, active_batch)
        self._copy_host(page_host, self._prefill_page_table, dtype=ttnn.int32)
        padded_width = max(128, _round_up(max(prompt_lens), 128))
        padded_prompt_lens = list(prompt_lens) + [0] * (self.batch - active_batch)
        selected = [None] * active_batch
        all_logits = []

        for start in range(0, padded_width, self.prefill_chunk_size):
            chunk_len = min(self.prefill_chunk_size, padded_width - start)
            token_device, chunk_table = self._stage_prefill_chunk(
                tokens,
                page_host,
                start=start,
                length=chunk_len,
            )
            hidden = self.model.prefill_hidden(
                token_device,
                page_table=self._prefill_page_table,
                kv_cache=caches,
                prompt_lens=padded_prompt_lens,
                chunk_start_idx=start,
                chunk_page_table=chunk_table,
            )
            if return_all_logits:
                local_groups = self.model.prefill_hidden_local_logits(hidden)
                flat_groups = []
                remaining = self.batch * chunk_len
                for local in local_groups:
                    rows = min(32, remaining)
                    flat_groups.append(self._local_logits_to_torch(local, valid_rows=rows)[0, 0])
                    remaining -= rows
                chunk_host = torch.cat(flat_groups, dim=0).reshape(self.batch, chunk_len, self.model.vocab_size)
                all_logits.append(chunk_host[:active_batch])
            else:
                for user, prompt_len in enumerate(prompt_lens):
                    if start < prompt_len <= start + chunk_len:
                        selected[user] = self.model.select_prefill_token_hidden(
                            hidden,
                            user,
                            prompt_len - start - 1,
                        )
            ttnn.deallocate(hidden, True)

        self._page_table_host = page_host
        if return_all_logits:
            return torch.cat(all_logits, dim=1)[:, :logical_width]
        if any(row is None for row in selected):
            raise RuntimeError("prefill failed to retain every final prompt row")
        local_logits = self.model.prefill_selected_hidden_logits(selected, fixed_sampling_rows=True)
        if sampling_mode == "device":
            sampled = self._sample_device(local_logits, tt_out_tok=self._prefill_sampled)
            return sampled
        host = self._local_logits_to_torch(local_logits, valid_rows=active_batch)[0, 0, :active_batch]
        return host.unsqueeze(1)

    def _prefill_single_local_logits(self, prompt: list[int], page_host: torch.Tensor, kv_cache):
        """Evidence helper: prefill one prompt and retain sampler-ready local logits."""
        prompt_len = len(prompt)
        if prompt_len < 1 or prompt_len > self.model.max_cache_len:
            raise ValueError("prompt is outside the supported context")
        self._release_decode_traces_before_allocating_prefill()
        page_host = self._normalise_page_table(page_host, 1)
        self._copy_host(page_host, self._prefill_page_table, dtype=ttnn.int32)
        tokens = torch.tensor([prompt], dtype=torch.long)
        padded_width = max(128, _round_up(prompt_len, 128))
        selected = None
        prompt_lens = [prompt_len] + [0] * (self.batch - 1)
        for start in range(0, padded_width, self.prefill_chunk_size):
            chunk_len = min(self.prefill_chunk_size, padded_width - start)
            token_device, chunk_table = self._stage_prefill_chunk(
                tokens,
                page_host,
                start=start,
                length=chunk_len,
            )
            hidden = self.model.prefill_hidden(
                token_device,
                page_table=self._prefill_page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                chunk_start_idx=start,
                chunk_page_table=chunk_table,
            )
            if start < prompt_len <= start + chunk_len:
                selected = self.model.select_prefill_token_hidden(hidden, 0, prompt_len - start - 1)
            ttnn.deallocate(hidden, True)
        if selected is None:
            raise RuntimeError("prefill did not retain its final row")
        return self.model.prefill_selected_hidden_logits([selected], fixed_sampling_rows=True)

    def _prepare_decode_host_inputs(self, tokens: torch.Tensor, positions: torch.Tensor, page_table: torch.Tensor):
        tokens = tokens.reshape(-1).to(torch.int64)
        positions = positions.reshape(-1).to(torch.int64)
        if tokens.numel() > self.batch or positions.numel() > self.batch:
            raise ValueError("decode batch exceeds the configured fixed slots")
        padded_tokens = torch.zeros(32, dtype=torch.int32)
        padded_tokens[: tokens.numel()] = tokens.to(torch.int32)
        padded_positions = torch.full((self.batch,), -1, dtype=torch.int32)
        padded_positions[: positions.numel()] = positions.to(torch.int32)
        rotary_positions = torch.clamp(padded_positions, min=0)
        return (
            self._replicated_host_tensor(padded_tokens.reshape(1, 1, 1, 32), dtype=ttnn.uint32),
            self._replicated_host_tensor(padded_positions, dtype=ttnn.int32),
            self._replicated_host_tensor(rotary_positions, dtype=ttnn.uint32),
            self._replicated_host_tensor(page_table, dtype=ttnn.int32),
        )

    def _validate_page_coverage(self, page_table: torch.Tensor, positions: torch.Tensor, active_batch: int) -> None:
        assigned: set[int] = set()
        for slot, position in enumerate(positions.reshape(-1).tolist()[:active_batch]):
            if position < 0:
                continue
            rounded_pages = self._sdpa_rounded_page_count(int(position) + 1)
            if rounded_pages > page_table.shape[1]:
                raise ValueError(f"slot {slot} page table is too narrow for decode position {position}")
            physical_pages = [int(value) for value in page_table[slot, :rounded_pages].tolist()]
            if any(value < 0 or value >= self.num_blocks for value in physical_pages):
                raise ValueError(
                    f"slot {slot} lacks valid physical pages for the rounded SDPA read at position {position}"
                )
            if len(set(physical_pages)) != len(physical_pages) or assigned.intersection(physical_pages):
                raise ValueError("active page-table rows must map disjoint physical cache pages")
            assigned.update(physical_pages)

    def _ensure_sampling_params(self):
        if self._sampling_params is not None:
            return self._sampling_params
        self._sampling_params = (
            self._replicated_device_tensor(torch.ones(32, dtype=torch.int32), dtype=ttnn.uint32),
            self._replicated_device_tensor(torch.zeros(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
            self._replicated_device_tensor(torch.ones(32, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
        )
        self._sampling_snapshot = (tuple([1] * 32), tuple([0.0] * 32), tuple([1.0] * 32))
        return self._sampling_params

    @staticmethod
    def _expand_sampling_value(value, *, active_batch: int, inactive_value, name: str):
        active = [value] * active_batch if isinstance(value, (int, float)) else list(value)
        if len(active) != active_batch:
            raise ValueError(f"{name} must be scalar or contain {active_batch} values")
        return active + [inactive_value] * (32 - active_batch)

    def set_sampling_params(self, *, top_k=1, top_p=0.0, temperature=1.0, active_batch: int = 1) -> None:
        if active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active_batch must be in [1,{self.batch}]")
        k = [
            int(value)
            for value in self._expand_sampling_value(top_k, active_batch=active_batch, inactive_value=1, name="top_k")
        ]
        p = [
            float(value)
            for value in self._expand_sampling_value(top_p, active_batch=active_batch, inactive_value=0.0, name="top_p")
        ]
        public_temp = [
            float(value)
            for value in self._expand_sampling_value(
                temperature, active_batch=active_batch, inactive_value=0.0, name="temperature"
            )
        ]
        if any(value < 1 or value > 32 for value in k[:active_batch]):
            raise ValueError("top_k must be in [1,32]")
        if any(value < 0.0 or value > 1.0 for value in p[:active_batch]):
            raise ValueError("top_p must be in [0,1]")
        if any(value < 0.0 for value in public_temp[:active_batch]):
            raise ValueError("temperature must be non-negative")
        device_temp = []
        for slot, value in enumerate(public_temp):
            if value == 0.0:
                k[slot], p[slot], value = 1, 0.0, 1.0
            device_temp.append(1.0 / value)
        stochastic = any(value > 1 for value in k[:active_batch])
        if stochastic != self._sampling_stochastic and self._trace_model_id is not None:
            self._release_decode_traces()
            self._decode_warm_key = None
        self._sampling_stochastic = stochastic
        device_params = self._ensure_sampling_params()
        snapshot = (tuple(k), tuple(p), tuple(device_temp))
        if snapshot == self._sampling_snapshot:
            return
        host_params = (
            self._replicated_host_tensor(torch.tensor(k, dtype=torch.int32), dtype=ttnn.uint32),
            self._replicated_host_tensor(torch.tensor(p, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
            self._replicated_host_tensor(torch.tensor(device_temp, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
        )
        copy_host_to_device(host_params, device_params)
        self._sampling_snapshot = snapshot
        self.trace_stats["sampling_param_host_copies"] += 3

    def _sample_device(self, logits, *, tt_out_tok=None):
        k, p, temp = self._ensure_sampling_params()
        if self._sampling_stochastic:
            sampled = self.model.sample_stochastic_split(
                logits,
                k=k,
                p=p,
                temp=temp,
                tt_out_tok=tt_out_tok,
            )
            ttnn.plus_one(self.model.sampler._seeds)
            return sampled
        return self.model.sample_greedy_split(logits, tt_out_tok=tt_out_tok)

    def _checkpoint_sampling_rng(self) -> None:
        """Save mutable sampler seeds without a host read or new allocation."""
        ttnn.copy(self.model.sampler._seeds, self._sampling_seed_checkpoint)
        self.trace_stats["rng_checkpoints"] += 1

    def _restore_sampling_rng(self) -> None:
        """Restore sampler seeds after non-request warm-up/capture execution."""
        ttnn.copy(self._sampling_seed_checkpoint, self.model.sampler._seeds)
        self.trace_stats["rng_restores"] += 1

    def _restore_trace_inputs(self, host_inputs, *, include_page_table: bool, token_device=None) -> None:
        count = 4 if include_page_table else 3
        if token_device is None:
            copy_host_to_device(host_inputs[:count], self._trace_inputs[:count])
            self.trace_stats["token_host_copies"] += 1
        else:
            ttnn.copy(token_device, self._trace_inputs[0])
            copy_host_to_device(host_inputs[1:count], self._trace_inputs[1:count])
            self.trace_stats["token_device_copies"] += 1
        self.trace_stats["position_host_copies"] += 1
        self.trace_stats["rotary_position_host_copies"] += 1
        if include_page_table:
            self.trace_stats["page_table_host_copies"] += 1

    def _warm_decode_graphs(self, host_inputs, kv_cache, *, active_batch: int, initial_token_device=None) -> None:
        self._trace_inputs = self._decode_trace_input_pool
        self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
        token, current_pos, rotary_pos, page_table = self._trace_inputs
        self.model.sampler.load_device_buffers()
        warm_logits = self.model.decode_forward_from_ttnn_inputs(
            token,
            current_pos,
            rotary_position=rotary_pos,
            page_table=page_table,
            kv_cache=kv_cache,
        )
        self._sample_device(warm_logits, tt_out_tok=token)
        self._synchronize()
        self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
        self._synchronize()
        self._decode_warm_key = (id(kv_cache), active_batch, self._sampling_stochastic)
        self.trace_stats["decode_warmups"] += 1

    def _capture_decode_traces(
        self,
        host_inputs,
        kv_cache,
        *,
        active_batch: int,
        initial_token_device=None,
    ) -> None:
        # all_reduce_async advances its persistent global semaphores.  A trace
        # release does not reset them, so recapturing from the advanced values
        # makes otherwise identical requests numerically dependent on the prior
        # trace epoch.  Restore the same two shared buffers/semaphores once per
        # capture; the selected persistent BF16 CCL remains unchanged in every
        # token replay and no work is added to the token loop.
        self.model.reset_decode_ccl_state()
        self._synchronize()
        self.trace_stats["ccl_capture_epoch_resets"] += 1
        self._trace_inputs = self._decode_trace_input_pool
        model_trace_id = None
        sampling_trace_id = None
        model_capture_open = False
        sampling_capture_open = False
        rng_checkpointed = self._sampling_stochastic
        if rng_checkpointed:
            self._checkpoint_sampling_rng()
        try:
            warm_key = (id(kv_cache), active_batch, self._sampling_stochastic)
            if self._decode_warm_key != warm_key:
                self._warm_decode_graphs(
                    host_inputs,
                    kv_cache,
                    active_batch=active_batch,
                    initial_token_device=initial_token_device,
                )
            self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
            self._synchronize()
            token, current_pos, rotary_pos, page_table = self._trace_inputs

            self.mesh_device.set_program_cache_misses_allowed(False)
            try:
                model_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                model_capture_open = True
                logits = self.model.decode_forward_from_ttnn_inputs(
                    token,
                    current_pos,
                    rotary_position=rotary_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                )
                ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
                model_capture_open = False
                self._synchronize()

                sampling_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                sampling_capture_open = True
                sampled = self._sample_device(logits, tt_out_tok=token)
                ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
                sampling_capture_open = False
                self._synchronize()
            finally:
                self.mesh_device.set_program_cache_misses_allowed(True)
        except Exception:
            if sampling_capture_open:
                try:
                    ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
                except Exception:
                    pass
            if model_capture_open:
                try:
                    ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
                except Exception:
                    pass
            for trace_id in (sampling_trace_id, model_trace_id):
                if trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, trace_id)
                    except Exception:
                        pass
            raise
        finally:
            if rng_checkpointed:
                self._restore_sampling_rng()

        self._trace_model_id = model_trace_id
        self._trace_sampling_id = sampling_trace_id
        self._trace_logits = logits
        self._trace_sampled = sampled
        self._trace_kv_cache = kv_cache
        self._trace_page_table_snapshot = self._page_table_to_torch(host_inputs[3]).clone()
        self._trace_active_batch = active_batch
        self.trace_stats["captures"] += 1
        self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
        self._synchronize()

    def _refresh_trace_state(self, host_inputs, kv_cache, *, active_batch: int, initial_token_device=None) -> None:
        new_page_table = self._page_table_to_torch(host_inputs[3])
        page_shape_changed = (
            self._trace_page_table_snapshot is not None
            and new_page_table.shape != self._trace_page_table_snapshot.shape
        )
        if self._trace_model_id is not None and (
            kv_cache is not self._trace_kv_cache or active_batch != self._trace_active_batch or page_shape_changed
        ):
            self._release_decode_traces()
        if self._trace_model_id is None:
            self._capture_decode_traces(
                host_inputs,
                kv_cache,
                active_batch=active_batch,
                initial_token_device=initial_token_device,
            )
            return
        if initial_token_device is None:
            copy_host_to_device(host_inputs[:3], self._trace_inputs[:3])
            self.trace_stats["token_host_copies"] += 1
        else:
            ttnn.copy(initial_token_device, self._trace_inputs[0])
            copy_host_to_device(host_inputs[1:3], self._trace_inputs[1:3])
            self.trace_stats["token_device_copies"] += 1
        self.trace_stats["position_host_copies"] += 1
        self.trace_stats["rotary_position_host_copies"] += 1
        if not torch.equal(new_page_table, self._trace_page_table_snapshot):
            ttnn.copy_host_to_device_tensor(host_inputs[3], self._trace_inputs[3])
            self._trace_page_table_snapshot = new_page_table.clone()
            self.trace_stats["page_table_host_copies"] += 1

    def _refresh_persistent_page_table(self, page_table, kv_cache, *, active_batch: int) -> None:
        if self._trace_model_id is None:
            raise RuntimeError("decode trace is not initialized")
        if kv_cache is not self._trace_kv_cache:
            raise RuntimeError("KV-cache identity changed; initialize a new trace")
        if active_batch != self._trace_active_batch:
            raise RuntimeError("fixed active slots changed; initialize a new trace")
        if page_table is None:
            return
        new_page_table = self._normalise_page_table(page_table, active_batch)
        if torch.equal(new_page_table, self._trace_page_table_snapshot):
            return
        host = self._replicated_host_tensor(new_page_table, dtype=ttnn.int32)
        ttnn.copy_host_to_device_tensor(host, self._trace_inputs[3])
        self._trace_page_table_snapshot = new_page_table.clone()
        self.trace_stats["page_table_host_copies"] += 1

    def _replay_split_sampling(self):
        ttnn.execute_trace(self.mesh_device, self._trace_model_id, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh_device, self._trace_sampling_id, cq_id=0, blocking=False)
        self.trace_stats["replays"] += 1
        return self._trace_sampled

    def _copy_forced_tokens(self, tokens: torch.Tensor) -> None:
        values = tokens.reshape(-1).to(torch.int64)
        if values.numel() != self._trace_active_batch:
            raise ValueError(f"expected {self._trace_active_batch} forced tokens, got {values.numel()}")
        host = torch.zeros(32, dtype=torch.int32)
        host[: values.numel()] = values.to(torch.int32)
        ttnn.copy_host_to_device_tensor(
            self._replicated_host_tensor(host.reshape(1, 1, 1, 32), dtype=ttnn.uint32),
            self._trace_inputs[0],
        )
        self.trace_stats["token_host_copies"] += 1

    def _release_decode_traces(self) -> None:
        released = self._trace_model_id is not None or self._trace_sampling_id is not None
        for trace_id in (self._trace_model_id, self._trace_sampling_id):
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
        if released:
            self.trace_stats["releases"] += 1
        self._trace_model_id = None
        self._trace_sampling_id = None
        self._trace_inputs = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_kv_cache = None
        self._trace_page_table_snapshot = None
        self._trace_active_batch = None
        # A warm-up belongs to the allocator/trace epoch whose addresses and
        # persistent CCL state it prepared.  Once either cooperating trace is
        # released, the next capture must take the same warm-up path as the
        # initial request.  Retaining this key made reset/re-capture skip that
        # path and produced request-order-dependent batch-32 results at page
        # boundaries.
        self._decode_warm_key = None

    def decode_forward(
        self,
        tokens: torch.Tensor | None,
        start_pos: torch.Tensor | None,
        *,
        page_table,
        kv_cache: Any,
        sampling_mode: str = "host",
        enable_trace: bool = False,
        active_batch: int | None = None,
        **kwargs: Any,
    ):
        """Decode once, or replay persistent token/position/page-table state."""
        if sampling_mode not in {"host", "device"}:
            raise ValueError("sampling_mode must be 'host' or 'device'")
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        inferred_batch = self._trace_active_batch if tokens is None else int(tokens.numel())
        active_batch = inferred_batch if active_batch is None else int(active_batch)
        if active_batch is None or active_batch < 1 or active_batch > self.batch:
            raise ValueError(f"active_batch must be in [1,{self.batch}]")
        if tokens is not None and tokens.numel() != active_batch:
            raise ValueError("tokens do not match active_batch")
        if start_pos is not None and start_pos.numel() != active_batch:
            raise ValueError("start_pos does not match active_batch")

        if enable_trace and sampling_mode == "device":
            if start_pos is not None:
                if page_table is None:
                    raise ValueError("initial trace state requires positions and page_table")
                page_host = self._normalise_page_table(page_table, active_batch)
                self._validate_page_coverage(page_host, start_pos, active_batch)
                initial_token_device = self._prefill_sampled if tokens is None else None
                host_tokens = torch.zeros(active_batch, dtype=torch.long) if tokens is None else tokens
                host_inputs = self._prepare_decode_host_inputs(host_tokens, start_pos, page_host)
                self._refresh_trace_state(
                    host_inputs,
                    caches,
                    active_batch=active_batch,
                    initial_token_device=initial_token_device,
                )
            else:
                self._refresh_persistent_page_table(page_table, caches, active_batch=active_batch)
                if tokens is not None:
                    self._copy_forced_tokens(tokens)
            return self._replay_split_sampling()

        if tokens is None or start_pos is None or page_table is None:
            raise ValueError("eager/host decode requires tokens, start_pos, and page_table")
        page_host = self._normalise_page_table(page_table, active_batch)
        self._validate_page_coverage(page_host, start_pos, active_batch)
        host_inputs = self._prepare_decode_host_inputs(tokens, start_pos, page_host)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)
        logits = self.model.decode_forward_from_ttnn_inputs(
            device_inputs[0],
            device_inputs[1],
            rotary_position=device_inputs[2],
            page_table=device_inputs[3],
            kv_cache=caches,
        )
        if sampling_mode == "device":
            return self._sample_device(logits)
        return self._local_logits_to_torch(logits, valid_rows=active_batch)[0, 0, :active_batch]

    def _generate_host_compat(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn],
    ) -> list[int]:
        """Explicit compatibility mode for tests which require host sampling."""
        # Eager host compatibility may allocate temporary device buffers.  TTNN
        # forbids allocations while a replayable trace is live because replay
        # addresses would no longer be allocator-safe.
        self._release_decode_traces()
        self._decode_warm_key = None
        kv_cache = self._ensure_kv_cache()
        horizon = len(prompt_token_ids) + max_new_tokens - 1
        page_host = self._make_page_table([horizon])
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=page_host,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode="host",
        )
        predicted = int(logits[0, 0].argmax().item())
        outputs = []
        for step in range(max_new_tokens):
            outputs.append(predicted)
            next_token = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            decoded = self.decode_forward(
                torch.tensor([[next_token]]),
                torch.tensor([len(prompt_token_ids) + step]),
                page_table=page_host,
                kv_cache=kv_cache,
                sampling_mode="host",
                enable_trace=False,
            )
            predicted = int(decoded[0].argmax().item())
        return outputs

    def generate(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        enable_trace: bool = True,
        sampling_mode: str = "device",
        stop_on_eos: bool = False,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        seed=None,
        **kwargs: Any,
    ) -> list[int]:
        """Generate through canonical model-trace + split-sampler feedback."""
        if not prompt_token_ids or max_new_tokens < 1:
            return []
        horizon = len(prompt_token_ids) + max_new_tokens - 1
        if horizon > self.model.max_cache_len:
            raise ValueError("prompt plus requested output exceeds the supported context")
        if sampling_mode == "host":
            return self._generate_host_compat(
                prompt_token_ids,
                max_new_tokens,
                next_input=next_input,
            )
        if sampling_mode != "device":
            raise ValueError("sampling_mode must be 'device' or 'host'")
        if not enable_trace and max_new_tokens > 1:
            raise ValueError("the optimized token-out path requires enable_trace=True")
        self.set_sampling_params(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            active_batch=1,
        )
        if seed is not None and self._sampling_stochastic:
            self.model.sampler.load_device_buffers()
            seed_values = [(int(seed) + slot * 0x9E3779B9) % (2**32 - 1) for slot in range(32)]
            self._copy_host(
                torch.tensor(seed_values, dtype=torch.int64).to(torch.uint32),
                self.model.sampler._seeds,
                dtype=ttnn.uint32,
            )

        kv_cache = self._ensure_kv_cache()
        page_host = self._make_page_table([horizon])
        sampled = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=page_host,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode="device",
        )
        predicted = int(self._sampled_to_torch(sampled)[0].item())
        outputs = []

        for step in range(max_new_tokens):
            outputs.append(predicted)
            forced_or_predicted = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            if stop_on_eos and next_input is None and predicted == self.tokenizer.eos_token_id:
                break
            initial_decode = step == 0
            sampled = self.decode_forward(
                (torch.tensor([[forced_or_predicted]]) if next_input is not None else None),
                torch.tensor([len(prompt_token_ids)]) if initial_decode else None,
                page_table=page_host if initial_decode else None,
                kv_cache=kv_cache,
                sampling_mode="device",
                enable_trace=True,
                active_batch=1,
            )
            predicted = int(self._sampled_to_torch(sampled)[0].item())
        self._page_table_host = page_host
        return outputs

    def reset(self) -> None:
        """Release request traces, then scrub persistent request state in place."""
        # Cache fill/reset operations may allocate internal device temporaries.
        # Metal forbids such allocations while any trace is live because a
        # later replay can corrupt them.  Release both cooperating traces before
        # the first reset op; stable input/cache addresses and the program cache
        # are retained, so the next request performs one safe recapture rather
        # than rebuilding anything per token.
        if self._trace_model_id is not None or self._trace_sampling_id is not None:
            self._synchronize()
            self._release_decode_traces()
        caches_to_clear = []
        if self._kv_cache is not None:
            caches_to_clear.append(self._kv_cache)
        if self._trace_kv_cache is not None and all(self._trace_kv_cache is not cache for cache in caches_to_clear):
            caches_to_clear.append(self._trace_kv_cache)
        for cache in caches_to_clear:
            # reset_kv_cache uses fill(..., output_tensor=cache): cache tensor
            # identities and DRAM addresses remain stable for trace replay.
            self.model.reset_kv_cache(cache)

        empty_page_table = torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32)
        self._copy_host(empty_page_table, self._prefill_page_table, dtype=ttnn.int32)
        for length, buffer in self._prefill_token_buffers.items():
            self._copy_host(torch.zeros((self.batch, length), dtype=torch.int32), buffer, dtype=ttnn.uint32)
        for pages, buffer in self._prefill_chunk_tables.items():
            self._copy_host(torch.full((self.batch, pages), -1, dtype=torch.int32), buffer, dtype=ttnn.int32)
        self._copy_host(torch.zeros((1, 1, 1, 32), dtype=torch.int32), self._prefill_sampled, dtype=ttnn.uint32)
        token, current_pos, rotary_pos, page_table = self._decode_trace_input_pool
        self._copy_host(torch.zeros((1, 1, 1, 32), dtype=torch.int32), token, dtype=ttnn.uint32)
        self._copy_host(torch.full((self.batch,), -1, dtype=torch.int32), current_pos, dtype=ttnn.int32)
        self._copy_host(torch.zeros((self.batch,), dtype=torch.int32), rotary_pos, dtype=ttnn.uint32)
        self._copy_host(empty_page_table, page_table, dtype=ttnn.int32)
        self._copy_host(
            torch.arange(32, dtype=torch.int64).to(torch.uint32), self.model.sampler._seeds, dtype=ttnn.uint32
        )
        self._copy_host(torch.zeros(32, dtype=torch.int32), self._sampling_seed_checkpoint, dtype=ttnn.uint32)
        self._page_table_host = None
        self.trace_stats["resets"] += 1
        self._synchronize()

    def teardown(self) -> None:
        self._release_decode_traces()
        if self._kv_cache is not None:
            for layer_cache in self._kv_cache:
                for tensor in layer_cache:
                    tensor.deallocate(True)
            self._kv_cache = None


def _resolve_snapshot(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot = hf_home / "hub" / "models--tiiuae--Falcon3-10B-Base" / "snapshots" / HF_REVISION
    if snapshot.is_dir():
        required = [
            snapshot / "config.json",
            snapshot / "tokenizer.json",
            snapshot / "model.safetensors.index.json",
            *[snapshot / f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)],
        ]
        if all(path.exists() for path in required):
            return snapshot
    return Path(snapshot_download(HF_MODEL_ID, revision=HF_REVISION, local_files_only=True))


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator:
    """Readiness discovery factory; intentionally builds no vLLM state."""
    snapshot = _resolve_snapshot(kwargs.pop("model_path", os.getenv("FALCON3_10B_MODEL_PATH")))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    max_context_len = int(kwargs.pop("max_context_len", 32768))
    override_num_layers = kwargs.pop("override_num_layers", None)
    num_layers = 40 if override_num_layers is None else int(override_num_layers)
    prefill_chunk_size = int(kwargs.pop("prefill_chunk_size", DEFAULT_PREFILL_CHUNK_SIZE))
    weight_cache_path = kwargs.pop("weight_cache_path", Path(model_dir) / "weight_cache")
    if kwargs:
        raise TypeError(f"unsupported build_generator kwargs: {sorted(kwargs)}")
    model = Falcon3Model.from_checkpoint(
        snapshot,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_cache_len=max_context_len,
        num_layers=num_layers,
        weight_cache_path=weight_cache_path,
    )
    return Falcon3Generator(model, tokenizer, prefill_chunk_size=prefill_chunk_size)


__all__ = ["Falcon3Generator", "build_generator"]
