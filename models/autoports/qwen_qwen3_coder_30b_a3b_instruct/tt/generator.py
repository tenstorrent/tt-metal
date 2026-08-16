# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Metal-readiness generator for the 4-die Qwen3-Coder-30B-A3B-Instruct model.

Two API levels, as the readiness contract requires:

* **low level** -- ``prefill_forward`` / ``decode_forward``. The caller owns the
  KV cache, the page table, the per-user prompt lengths and the per-user decode
  positions, and threads them through each call. Mixed-length prompts, fixed
  request slots and inactive rows (position ``-1``) are all expressible here.
  This is the surface a serving adapter drives.
* **high level** -- ``generate``. Owns the cache and page table and loops
  deterministically over the low-level calls.

The measured token-out path is **entirely on device**: the model trace produces
sampler-ready per-die logits, a second trace runs the split sampler, the sampled
token is written straight into the persistent decode token input through
``tt_out_tok``, and the trace advances both position tensors itself with
``ttnn.plus_one``. Between two steady-state tokens the host does exactly one
thing -- replay two traces -- plus whatever readback the caller asked for.
``sampling_mode="host"`` is the explicit compatibility mode for tests that need
host sampling and is never used to produce a performance number.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import (
    HF_MODEL_ID,
    HF_REVISION,
    MAX_CONTEXT,
    NUM_LAYERS,
    Qwen3CoderModel,
)
from models.common.readiness_check.contract import Generator, NextInputFn

#: ``ttnn.sampling``/``nlp_create_qkv_heads_decode`` both work in 32-slot units.
SAMPLING_SLOTS = 32


def _first_device_to_torch(tensor) -> torch.Tensor:
    shards = ttnn.get_device_tensors(tensor)
    return ttnn.to_torch(shards[0] if shards else tensor)


class Qwen3CoderGenerator(Generator):
    """Caller-owned cache/page-table state plus traced on-device token feedback."""

    def __init__(self, model: Qwen3CoderModel, tokenizer):
        self.model = model
        self.mesh_device = model.mesh_device
        self.tokenizer = tokenizer
        self.batch = model.max_batch_size
        self.page_block_size = model.page_block_size
        self.pages_per_user = math.ceil(model.max_cache_len / self.page_block_size)
        self.num_blocks = self.batch * self.pages_per_user

        self._kv_cache: list | None = None
        self._trace_model_id = None
        self._trace_sampling_id = None
        self._trace_inputs = None
        self._trace_logits = None
        self._trace_sampled = None
        self._trace_kv_cache = None
        self._trace_page_table_snapshot: torch.Tensor | None = None
        self._trace_active_batch = None
        #: Highest rotary position the *next* trace replay will gather at. The
        #: trace advances ``rotary_position`` on device with ``ttnn.plus_one``
        #: and nothing on device clamps it, so this host-side mirror is the only
        #: thing that can tell a replay it is about to index past the cos/sin
        #: table. See ``decode_forward``.
        self._trace_rotary_position: int | None = None
        self._decode_warm_key = None
        self._sampling_params = None
        self._sampling_snapshot = None
        self._sampling_stochastic = False

        #: Steady-state host-work counters. Everything except ``replays`` and
        #: ``caller_token_readbacks`` must stay flat while tokens are produced.
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
        }
        self._allocate_persistent_inputs()

    # -- persistent device state ---------------------------------------------

    def _replicated_host_tensor(self, host: torch.Tensor, *, dtype):
        return ttnn.from_torch(
            host.contiguous(),
            device=None,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _replicated_device_tensor(self, host: torch.Tensor, *, dtype):
        return ttnn.from_torch(
            host.contiguous(),
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _copy_host(self, host: torch.Tensor, device, *, dtype) -> None:
        ttnn.copy_host_to_device_tensor(self._replicated_host_tensor(host, dtype=dtype), device)

    def _allocate_persistent_inputs(self) -> None:
        """Allocate every stable decode input **before** any trace is captured."""
        self._prefill_page_table = self._replicated_device_tensor(
            torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32), dtype=ttnn.int32
        )
        self._prefill_sampled = self._replicated_device_tensor(
            torch.zeros((1, 1, 1, SAMPLING_SLOTS), dtype=torch.int32), dtype=ttnn.uint32
        )
        self._decode_trace_input_pool = (
            # token: [1,1,1,32] uint32, the tensor ``tt_out_tok`` writes into
            self._replicated_device_tensor(
                torch.zeros((1, 1, 1, SAMPLING_SLOTS), dtype=torch.int32), dtype=ttnn.uint32
            ),
            # current_pos: [batch] int32, consumed by paged_update_cache and SDPA
            self._replicated_device_tensor(torch.full((self.batch,), -1, dtype=torch.int32), dtype=ttnn.int32),
            # rotary_position: [1, batch] uint32, the cos/sin gather index
            self._replicated_device_tensor(torch.zeros((1, self.batch), dtype=torch.int32), dtype=ttnn.uint32),
            # page_table: [batch, pages_per_user] int32
            self._replicated_device_tensor(
                torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32), dtype=ttnn.int32
            ),
        )

    def _ensure_kv_cache(self):
        if self._kv_cache is None:
            self._kv_cache = self.model.allocate_kv_cache(num_blocks=self.num_blocks)
        return self._kv_cache

    def _synchronize(self) -> None:
        ttnn.synchronize_device(self.mesh_device)
        self.trace_stats["explicit_synchronizations"] += 1

    # -- page tables ----------------------------------------------------------

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
            host = host[:, : self.pages_per_user]
        if host.shape[0] < self.batch:
            host = torch.nn.functional.pad(host, (0, 0, 0, self.batch - host.shape[0]), value=-1)
        return host.contiguous()

    def _sdpa_rounded_page_count(self, token_count: int) -> int:
        """Physical pages the paged decode SDPA kernel actually reads.

        The kernel rounds a short sequence up to a power-of-two tile count and a
        long one up to a multiple of eight, and it reads the whole rounded
        window before causal masking. Every rounded tail page therefore needs a
        valid mapping even though it holds no live token yet -- allocating only
        ``ceil(len/block)`` pages produces top-k misses that cliff at exactly
        those boundaries and look like dtype drift.
        """
        if token_count < 1 or token_count > self.model.max_cache_len:
            raise ValueError("SDPA token count is outside the supported context")
        logical_pages = math.ceil(token_count / self.page_block_size)
        if logical_pages <= 8:
            return 1 << (logical_pages - 1).bit_length()
        return 8 * math.ceil(logical_pages / 8)

    def make_page_table(self, lengths: Sequence[int]) -> torch.Tensor:
        """A disjoint physical-block assignment covering each user's horizon."""
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

    def _validate_page_coverage(self, page_table: torch.Tensor, positions: torch.Tensor, active_batch: int) -> None:
        assigned: set[int] = set()
        for slot, position in enumerate(positions.reshape(-1).tolist()[:active_batch]):
            if position < 0:  # inactive row
                continue
            logical_pages = math.ceil((int(position) + 1) / self.page_block_size)
            rounded_pages = self._sdpa_rounded_page_count(int(position) + 1)
            if rounded_pages > page_table.shape[1]:
                raise ValueError(f"slot {slot} page table is too narrow for decode position {position}")
            physical = [int(v) for v in page_table[slot, :rounded_pages].tolist()]
            if any(v < 0 or v >= self.num_blocks for v in physical):
                raise ValueError(f"slot {slot} lacks valid physical pages for the rounded SDPA read at {position}")
            live = physical[:logical_pages]
            if len(set(live)) != len(live) or assigned.intersection(live):
                raise ValueError("active page-table rows must map disjoint physical cache pages")
            assigned.update(live)

    # -- prefill --------------------------------------------------------------

    def _release_decode_traces_before_allocating(self) -> None:
        """Prefill is eager and allocates; a live trace makes that unsafe."""
        if self._trace_model_id is None and self._trace_sampling_id is None:
            return
        self._synchronize()
        self._release_decode_traces()

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache: Any,
        prompt_lens: Sequence[int],
        return_all_logits: bool = False,
        sampling_mode: str = "host",
        **kwargs: Any,
    ):
        """Prefill arbitrary logical lengths, one user at a time into the cache.

        ``tokens`` is ``[active_batch, width]`` and ``prompt_lens`` gives each
        row's **real** length. Rows may differ in length; each user is prefilled
        at exactly its own logical length, so nothing is padded to a chunk, tile
        or page boundary at the model boundary and no mask is needed. The
        returned logits are sliced back to the logical prompt length.
        """
        if sampling_mode not in {"host", "device"}:
            raise ValueError("sampling_mode must be 'host' or 'device'")
        if sampling_mode == "device" and return_all_logits:
            raise ValueError("return_all_logits is incompatible with device sampling")
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [batch,seq], got {tuple(tokens.shape)}")
        active_batch, logical_width = int(tokens.shape[0]), int(tokens.shape[1])
        if not 1 <= active_batch <= self.batch:
            raise ValueError(f"active batch must be in [1,{self.batch}]")
        if len(prompt_lens) != active_batch or any(not 1 <= int(n) <= logical_width for n in prompt_lens):
            raise ValueError("prompt_lens must contain one valid logical length per input row")
        if max(prompt_lens) > self.model.max_cache_len:
            raise ValueError("prompt exceeds the supported context")

        self._release_decode_traces_before_allocating()
        self.model.ensure_rope_capacity(max(prompt_lens))
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        page_host = self._normalise_page_table(page_table, active_batch)
        self._copy_host(page_host, self._prefill_page_table, dtype=ttnn.int32)
        self.model.bind_page_table(caches, self._prefill_page_table)

        per_user_logits: list[torch.Tensor] = []
        selected_rows = []
        for user in range(active_batch):
            prompt_len = int(prompt_lens[user])
            token_device = ttnn.from_torch(
                tokens[user : user + 1, :prompt_len].to(torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            hidden = self.model.prefill_hidden(token_device, kv_cache=caches, user_id=user)
            ttnn.deallocate(token_device, True)
            if return_all_logits:
                normed = self.model.prefill_norm(hidden)
                ttnn.deallocate(hidden, True)
                local = self.model.local_logits(normed)
                ttnn.deallocate(normed, True)
                host = self.model.gather_logits_to_torch(local)[0, 0, :prompt_len, :]
                ttnn.deallocate(local, True)
                per_user_logits.append(
                    torch.nn.functional.pad(host, (0, 0, 0, logical_width - prompt_len)).unsqueeze(0)
                )
            else:
                selected_rows.append(self.model.select_prefill_rows(hidden, [prompt_len - 1]))
                ttnn.deallocate(hidden, True)

        if return_all_logits:
            return torch.cat(per_user_logits, dim=0)[:, :logical_width]

        selected = (
            selected_rows[0]
            if len(selected_rows) == 1
            else ttnn.concat(selected_rows, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        normed = self.model.prefill_norm(selected)
        if selected is not selected_rows[0] or len(selected_rows) > 1:
            for row in selected_rows:
                ttnn.deallocate(row, True)
        else:
            ttnn.deallocate(selected, True)
        if sampling_mode == "device":
            padded = self._pad_rows_to_sampling_slots(normed, active_batch)
            local = self.model.local_logits(padded)
            ttnn.deallocate(padded, True)
            sampled = self._sample_device(local, tt_out_tok=self._prefill_sampled)
            ttnn.deallocate(local, True)
            return sampled
        local = self.model.local_logits(normed)
        ttnn.deallocate(normed, True)
        host = self.model.gather_logits_to_torch(local, valid_rows=active_batch)[0, 0]
        ttnn.deallocate(local, True)
        return host.unsqueeze(1)

    def _pad_rows_to_sampling_slots(self, normed, active_batch: int):
        """``ttnn.sampling`` works in 32 fixed slots; pad the selected rows up."""
        rows = int(normed.shape[-2])
        if rows >= SAMPLING_SLOTS:
            return normed
        padded = ttnn.pad(
            normed,
            [(0, 0), (0, 0), (0, SAMPLING_SLOTS - rows), (0, 0)],
            value=0.0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(normed, True)
        return padded

    # -- sampling parameters --------------------------------------------------

    def _ensure_sampling_params(self):
        if self._sampling_params is None:
            self._sampling_params = (
                self._replicated_device_tensor(torch.ones(SAMPLING_SLOTS, dtype=torch.int32), dtype=ttnn.uint32),
                self._replicated_device_tensor(torch.zeros(SAMPLING_SLOTS, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
                self._replicated_device_tensor(torch.ones(SAMPLING_SLOTS, dtype=torch.bfloat16), dtype=ttnn.bfloat16),
            )
            self._sampling_snapshot = ((1,) * SAMPLING_SLOTS, (0.0,) * SAMPLING_SLOTS, (1.0,) * SAMPLING_SLOTS)
        return self._sampling_params

    @staticmethod
    def _expand(value, *, active_batch: int, inactive, name: str):
        active = [value] * active_batch if isinstance(value, (int, float)) else list(value)
        if len(active) != active_batch:
            raise ValueError(f"{name} must be scalar or contain {active_batch} values")
        return active + [inactive] * (SAMPLING_SLOTS - active_batch)

    def set_sampling_params(self, *, top_k=1, top_p=0.0, temperature=1.0, active_batch: int = 1) -> None:
        """Set per-slot ``(k, p, temperature)``. ``k=1`` is exactly greedy."""
        if not 1 <= active_batch <= self.batch:
            raise ValueError(f"active_batch must be in [1,{self.batch}]")
        k = [int(v) for v in self._expand(top_k, active_batch=active_batch, inactive=1, name="top_k")]
        p = [float(v) for v in self._expand(top_p, active_batch=active_batch, inactive=0.0, name="top_p")]
        temp = [
            float(v) for v in self._expand(temperature, active_batch=active_batch, inactive=0.0, name="temperature")
        ]
        if any(not 0.0 <= v <= 1.0 for v in p[:active_batch]):
            raise ValueError("top_p must be in [0,1]")
        if any(v < 0.0 for v in temp[:active_batch]):
            raise ValueError("temperature must be non-negative")
        for slot in range(active_batch):
            # A serving stack spells greedy as temperature=0 / top_k=0.
            if temp[slot] == 0.0 and k[slot] == 0:
                k[slot] = 1
        if any(not 1 <= v <= SAMPLING_SLOTS for v in k[:active_batch]):
            raise ValueError("top_k must be in [1,32] (0 accepted only with temperature=0)")
        device_temp = []
        for slot, value in enumerate(temp):
            if value == 0.0:
                k[slot], p[slot], value = 1, 0.0, 1.0
            device_temp.append(1.0 / value)
        stochastic = any(v > 1 for v in k[:active_batch]) or any(v > 0.0 for v in p[:active_batch])
        if stochastic != self._sampling_stochastic and self._trace_model_id is not None:
            self._release_decode_traces()
        self._sampling_stochastic = stochastic
        params = self._ensure_sampling_params()
        snapshot = (tuple(k), tuple(p), tuple(device_temp))
        if snapshot == self._sampling_snapshot:
            return
        for host, device, dtype in (
            (torch.tensor(k, dtype=torch.int32), params[0], ttnn.uint32),
            (torch.tensor(p, dtype=torch.bfloat16), params[1], ttnn.bfloat16),
            (torch.tensor(device_temp, dtype=torch.bfloat16), params[2], ttnn.bfloat16),
        ):
            self._copy_host(host, device, dtype=dtype)
            self.trace_stats["sampling_param_host_copies"] += 1
        self._sampling_snapshot = snapshot

    def _sample_device(self, logits, *, tt_out_tok=None):
        """Greedy takes the argmax strategy, anything sampled takes the split one.

        Both are ``Sampling1D``, both traced, both write ``tt_out_tok``. The
        split is by *request*, not by convenience: greedy is exactly top-1, and
        at this vocabulary the argmax strategy computes it 3.3x faster
        (``doc/full_model/README.md``, "The sampler comparison"). Changing
        between the two releases the decode traces, which ``set_sampling_params``
        already does when ``_sampling_stochastic`` flips.
        """
        k, p, temp = self._ensure_sampling_params()
        if not self._sampling_stochastic:
            return self.model.sample_greedy_argmax(logits, tt_out_tok=tt_out_tok)
        return self.model.sample_split(logits, k=k, p=p, temp=temp, tt_out_tok=tt_out_tok)

    def _sampled_to_torch(self, sampled) -> torch.Tensor:
        self.trace_stats["caller_token_readbacks"] += 1
        return _first_device_to_torch(sampled).reshape(-1)[: self.batch].to(torch.long)

    # -- decode trace ---------------------------------------------------------

    def _prepare_decode_host_inputs(self, tokens: torch.Tensor, positions: torch.Tensor, page_table: torch.Tensor):
        tokens = tokens.reshape(-1).to(torch.int64)
        positions = positions.reshape(-1).to(torch.int64)
        if tokens.numel() > self.batch or positions.numel() > self.batch:
            raise ValueError("decode batch exceeds the configured fixed slots")
        padded_tokens = torch.zeros(SAMPLING_SLOTS, dtype=torch.int32)
        padded_tokens[: tokens.numel()] = tokens.to(torch.int32)
        padded_positions = torch.full((self.batch,), -1, dtype=torch.int32)
        padded_positions[: positions.numel()] = positions.to(torch.int32)
        rotary = torch.clamp(padded_positions, min=0).reshape(1, self.batch)
        return (
            self._replicated_host_tensor(padded_tokens.reshape(1, 1, 1, SAMPLING_SLOTS), dtype=ttnn.uint32),
            self._replicated_host_tensor(padded_positions, dtype=ttnn.int32),
            self._replicated_host_tensor(rotary, dtype=ttnn.uint32),
            self._replicated_host_tensor(page_table, dtype=ttnn.int32),
        )

    def _restore_trace_inputs(self, host_inputs, *, include_page_table: bool, token_device=None) -> None:
        count = 4 if include_page_table else 3
        start = 0
        if token_device is not None:
            ttnn.copy(token_device, self._trace_inputs[0])
            self.trace_stats["token_device_copies"] += 1
            start = 1
        for index in range(start, count):
            ttnn.copy_host_to_device_tensor(host_inputs[index], self._trace_inputs[index])
        if token_device is None:
            self.trace_stats["token_host_copies"] += 1
        self.trace_stats["position_host_copies"] += 1
        self.trace_stats["rotary_position_host_copies"] += 1
        if include_page_table:
            self.trace_stats["page_table_host_copies"] += 1

    def _warm_decode_graphs(self, host_inputs, kv_cache, *, active_batch: int, initial_token_device=None) -> None:
        """Compile every program once eagerly.

        Non-negotiable rather than merely tidy: ``_decode_ccl_buffers``
        allocates the two persistent collective buffers on the first call at a
        shape, and ``ttnn.from_torch`` inside ``begin_trace_capture`` raises and
        leaves the capture open -- a hung mesh (stage-04 ``work_log.md`` §6).
        """
        self._trace_inputs = self._decode_trace_input_pool
        self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
        token, current_pos, rotary_pos, page_table = self._trace_inputs
        self.model.bind_page_table(kv_cache, page_table)
        logits = self.model.decode_forward_from_ttnn_inputs(
            # ``advance_position=True`` here as well as in the capture: every op
            # the traced graph contains must already be in the program cache,
            # and that includes the two ``ttnn.plus_one`` calls. The positions
            # this leaves behind are overwritten by the restore below.
            token,
            current_pos,
            rotary_position=rotary_pos,
            kv_cache=kv_cache,
            advance_position=True,
        )
        self._sample_device(logits, tt_out_tok=token)
        ttnn.deallocate(logits, True)
        self._synchronize()
        self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
        self._synchronize()
        self._decode_warm_key = (id(kv_cache), active_batch, self._sampling_stochastic)
        self.trace_stats["decode_warmups"] += 1

    def _capture_decode_traces(self, host_inputs, kv_cache, *, active_batch: int, initial_token_device=None) -> None:
        self._trace_inputs = self._decode_trace_input_pool
        model_trace_id = sampling_trace_id = None
        model_open = sampling_open = False
        try:
            warm_key = (id(kv_cache), active_batch, self._sampling_stochastic)
            if self._decode_warm_key != warm_key:
                self._warm_decode_graphs(
                    host_inputs, kv_cache, active_batch=active_batch, initial_token_device=initial_token_device
                )
            self._restore_trace_inputs(host_inputs, include_page_table=True, token_device=initial_token_device)
            self._synchronize()
            token, current_pos, rotary_pos, page_table = self._trace_inputs
            self.model.bind_page_table(kv_cache, page_table)

            model_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            model_open = True
            logits = self.model.decode_forward_from_ttnn_inputs(
                token, current_pos, rotary_position=rotary_pos, kv_cache=kv_cache, advance_position=True
            )
            ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
            model_open = False
            self._synchronize()

            sampling_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            sampling_open = True
            sampled = self._sample_device(logits, tt_out_tok=token)
            ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
            sampling_open = False
            self._synchronize()
        except Exception:
            if sampling_open:
                ttnn.end_trace_capture(self.mesh_device, sampling_trace_id, cq_id=0)
            if model_open:
                ttnn.end_trace_capture(self.mesh_device, model_trace_id, cq_id=0)
            for trace_id in (sampling_trace_id, model_trace_id):
                if trace_id is not None:
                    try:
                        ttnn.release_trace(self.mesh_device, trace_id)
                    except Exception:
                        pass
            raise

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
        shape_changed = (
            self._trace_page_table_snapshot is not None
            and new_page_table.shape != self._trace_page_table_snapshot.shape
        )
        if self._trace_model_id is not None and (
            kv_cache is not self._trace_kv_cache or active_batch != self._trace_active_batch or shape_changed
        ):
            self._release_decode_traces()
        if self._trace_model_id is None:
            self._capture_decode_traces(
                host_inputs, kv_cache, active_batch=active_batch, initial_token_device=initial_token_device
            )
            return
        self._restore_trace_inputs(host_inputs, include_page_table=False, token_device=initial_token_device)
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
            return  # unchanged page table costs zero host copies
        ttnn.copy_host_to_device_tensor(
            self._replicated_host_tensor(new_page_table, dtype=ttnn.int32), self._trace_inputs[3]
        )
        self._trace_page_table_snapshot = new_page_table.clone()
        self.trace_stats["page_table_host_copies"] += 1

    def _copy_forced_tokens(self, tokens: torch.Tensor) -> None:
        """Teacher forcing: overwrite the fed-back token, keep everything else."""
        values = tokens.reshape(-1).to(torch.int64)
        if values.numel() != self._trace_active_batch:
            raise ValueError(f"expected {self._trace_active_batch} forced tokens, got {values.numel()}")
        host = torch.zeros(SAMPLING_SLOTS, dtype=torch.int32)
        host[: values.numel()] = values.to(torch.int32)
        ttnn.copy_host_to_device_tensor(
            self._replicated_host_tensor(host.reshape(1, 1, 1, SAMPLING_SLOTS), dtype=ttnn.uint32),
            self._trace_inputs[0],
        )
        self.trace_stats["token_host_copies"] += 1

    def _replay_split_sampling(self):
        ttnn.execute_trace(self.mesh_device, self._trace_model_id, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh_device, self._trace_sampling_id, cq_id=0, blocking=False)
        self.trace_stats["replays"] += 1
        return self._trace_sampled

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
        self._trace_rotary_position = None
        self._decode_warm_key = None

    def _ensure_decode_rope_capacity(self, required_len: int) -> None:
        """Grow the cos/sin tables for decode, releasing traces if they move.

        ``Qwen3CoderModel.ensure_rope_capacity`` reallocates ``cos_table`` and
        ``sin_table`` when it grows them, and a captured trace holds the *old*
        tensor identities -- replaying it afterwards would gather from freed
        DRAM. ``prefill_forward`` and ``generate`` are safe because both release
        the decode traces before they call it; the low-level ``decode_forward``
        has no such release, so it does one here and only when the tables
        actually moved.
        """
        if required_len > self.model.max_cache_len:
            raise ValueError(f"decode horizon {required_len} exceeds the supported context {self.model.max_cache_len}")
        if self.model.ensure_rope_capacity(required_len):
            self._release_decode_traces()

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
        decode_horizon: int | None = None,
        **kwargs: Any,
    ):
        """One decode step.

        With ``enable_trace=True, sampling_mode="device"`` this is the delivered
        path: pass ``start_pos``/``page_table`` on the first step to install the
        trace, then call with ``tokens=None, start_pos=None, page_table=None``
        and the traces replay over persistent state -- the sampled token from
        step *N* is already the token input of step *N+1*, and both position
        tensors were advanced on device inside the model trace.

        **Rotary capacity.** The cos/sin tables are sized lazily, and the traced
        loop advances ``rotary_position`` with ``ttnn.plus_one`` with nothing on
        device to clamp it, so a replay past the table length would gather
        out of range and silently rotate at the wrong position. Pass
        ``decode_horizon`` -- the highest position this trace will ever decode
        at, i.e. ``prompt_len + max_new_tokens - 1`` -- on the installing call
        and the tables are grown once, up front, to cover the whole run. Without
        it the tables are sized only for ``start_pos`` and a replay that would
        step past them raises instead of returning a wrong answer.
        ``generate`` sizes for its own horizon and never hits either path.
        """
        if sampling_mode not in {"host", "device"}:
            raise ValueError("sampling_mode must be 'host' or 'device'")
        caches = self._ensure_kv_cache() if kv_cache is None else kv_cache
        inferred = self._trace_active_batch if tokens is None else int(tokens.numel())
        active_batch = inferred if active_batch is None else int(active_batch)
        if active_batch is None or not 1 <= active_batch <= self.batch:
            raise ValueError(f"active_batch must be in [1,{self.batch}]")
        if tokens is not None and tokens.numel() != active_batch:
            raise ValueError("tokens do not match active_batch")
        if start_pos is not None and start_pos.numel() != active_batch:
            raise ValueError("start_pos does not match active_batch")

        if enable_trace and sampling_mode == "device":
            if start_pos is not None:
                if page_table is None:
                    raise ValueError("initial trace state requires positions and page_table")
                highest = int(start_pos.reshape(-1).max().item())
                horizon = highest + 1 if decode_horizon is None else int(decode_horizon)
                if horizon < highest + 1:
                    raise ValueError("decode_horizon is below the requested start_pos")
                self._ensure_decode_rope_capacity(horizon)
                page_host = self._normalise_page_table(page_table, active_batch)
                self._validate_page_coverage(page_host, start_pos, active_batch)
                initial_token_device = self._prefill_sampled if tokens is None else None
                host_tokens = torch.zeros(active_batch, dtype=torch.long) if tokens is None else tokens
                host_inputs = self._prepare_decode_host_inputs(host_tokens, start_pos, page_host)
                self._refresh_trace_state(
                    host_inputs, caches, active_batch=active_batch, initial_token_device=initial_token_device
                )
                # The installing call also replays once, at ``highest``.
                self._trace_rotary_position = highest
            else:
                self._refresh_persistent_page_table(page_table, caches, active_batch=active_batch)
                if tokens is not None:
                    self._copy_forced_tokens(tokens)
                if self._trace_rotary_position is not None:
                    # ``ttnn.plus_one`` already moved the device tensor on the
                    # previous replay; this replay gathers at that position.
                    self._trace_rotary_position += 1
                    if self._trace_rotary_position >= self.model.rope_cache_len:
                        raise RuntimeError(
                            f"decode position {self._trace_rotary_position} is past the rotary table "
                            f"({self.model.rope_cache_len} entries). ``ttnn.embedding`` would gather out of "
                            "range inside the replayed trace and rotate at a wrong position without "
                            "raising. Re-install the trace with decode_horizon= set to the highest "
                            "position this run will reach."
                        )
            return self._replay_split_sampling()

        if tokens is None or start_pos is None or page_table is None:
            raise ValueError("eager/host decode requires tokens, start_pos and page_table")
        self._release_decode_traces_before_allocating()
        # Eager decode gathers cos/sin at ``start_pos`` too, and holds no trace
        # by this point, so the tables can simply be grown to fit.
        self._ensure_decode_rope_capacity(int(start_pos.reshape(-1).max().item()) + 1)
        page_host = self._normalise_page_table(page_table, active_batch)
        self._validate_page_coverage(page_host, start_pos, active_batch)
        host_inputs = self._prepare_decode_host_inputs(tokens, start_pos, page_host)
        device_inputs = [
            ttnn.to_device(tensor, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for tensor in host_inputs
        ]
        self.model.bind_page_table(caches, device_inputs[3])
        logits = self.model.decode_forward_from_ttnn_inputs(
            device_inputs[0],
            device_inputs[1],
            rotary_position=device_inputs[2],
            kv_cache=caches,
            advance_position=False,
        )
        if sampling_mode == "device":
            sampled = self._sample_device(logits)
            ttnn.deallocate(logits, True)
            return sampled
        host = self.model.gather_logits_to_torch(logits, valid_rows=active_batch)[0, 0]
        ttnn.deallocate(logits, True)
        return host

    # -- high level -----------------------------------------------------------

    def _generate_host_compat(
        self, prompt_token_ids: list[int], max_new_tokens: int, *, next_input: Optional[NextInputFn]
    ) -> list[int]:
        """Explicit host-sampling compatibility mode. Never a measured path."""
        self._release_decode_traces()
        kv_cache = self._ensure_kv_cache()
        horizon = len(prompt_token_ids) + max_new_tokens - 1
        page_host = self.make_page_table([horizon])
        logits = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=page_host,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode="host",
        )
        predicted = int(logits[0, 0].argmax().item())
        outputs: list[int] = []
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
        **kwargs: Any,
    ) -> list[int]:
        """Prefill, then loop the traced split-sampling decode path."""
        if not prompt_token_ids or max_new_tokens < 1:
            return []
        horizon = len(prompt_token_ids) + max_new_tokens - 1
        if horizon > self.model.max_cache_len:
            raise ValueError("prompt plus requested output exceeds the supported context")
        self._release_decode_traces_before_allocating()
        self.model.ensure_rope_capacity(horizon)
        if sampling_mode == "host":
            return self._generate_host_compat(prompt_token_ids, max_new_tokens, next_input=next_input)
        if sampling_mode != "device":
            raise ValueError("sampling_mode must be 'device' or 'host'")
        if not enable_trace and max_new_tokens > 1:
            raise ValueError("the optimized token-out path requires enable_trace=True")
        self.set_sampling_params(top_k=top_k, top_p=top_p, temperature=temperature, active_batch=1)

        kv_cache = self._ensure_kv_cache()
        page_host = self.make_page_table([horizon])
        sampled = self.prefill_forward(
            torch.tensor([prompt_token_ids]),
            page_table=page_host,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_token_ids)],
            sampling_mode="device",
        )
        predicted = int(self._sampled_to_torch(sampled)[0].item())
        outputs: list[int] = []
        for step in range(max_new_tokens):
            outputs.append(predicted)
            forced = next_input(step, predicted) if next_input is not None else predicted
            if step + 1 == max_new_tokens:
                break
            if stop_on_eos and next_input is None and predicted == self.tokenizer.eos_token_id:
                break
            initial = step == 0
            sampled = self.decode_forward(
                (torch.tensor([[forced]]) if next_input is not None else None),
                torch.tensor([len(prompt_token_ids)]) if initial else None,
                page_table=page_host if initial else None,
                kv_cache=kv_cache,
                sampling_mode="device",
                enable_trace=True,
                active_batch=1,
                decode_horizon=horizon,
            )
            predicted = int(self._sampled_to_torch(sampled)[0].item())
        return outputs

    def reset(self) -> None:
        """Wipe per-prompt state, keeping weights, buffers and program cache."""
        if self._trace_model_id is not None or self._trace_sampling_id is not None:
            self._synchronize()
            self._release_decode_traces()
        if self._kv_cache is not None:
            self.model.reset_kv_cache(self._kv_cache)
        empty = torch.full((self.batch, self.pages_per_user), -1, dtype=torch.int32)
        self._copy_host(empty, self._prefill_page_table, dtype=ttnn.int32)
        self._copy_host(
            torch.zeros((1, 1, 1, SAMPLING_SLOTS), dtype=torch.int32), self._prefill_sampled, dtype=ttnn.uint32
        )
        token, current_pos, rotary_pos, page_table = self._decode_trace_input_pool
        self._copy_host(torch.zeros((1, 1, 1, SAMPLING_SLOTS), dtype=torch.int32), token, dtype=ttnn.uint32)
        self._copy_host(torch.full((self.batch,), -1, dtype=torch.int32), current_pos, dtype=ttnn.int32)
        self._copy_host(torch.zeros((1, self.batch), dtype=torch.int32), rotary_pos, dtype=ttnn.uint32)
        self._copy_host(empty, page_table, dtype=ttnn.int32)
        self._trace_rotary_position = None
        self.trace_stats["resets"] += 1
        self._synchronize()

    def teardown(self) -> None:
        self._release_decode_traces()
        if self._kv_cache is not None:
            for cache in self._kv_cache:
                ttnn.deallocate(cache.k, True)
                ttnn.deallocate(cache.v, True)
            self._kv_cache = None


def _resolve_snapshot(model_path: str | Path | None = None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshot = hf_home / "hub" / "models--Qwen--Qwen3-Coder-30B-A3B-Instruct" / "snapshots" / HF_REVISION
    if (snapshot / "model.safetensors.index.json").is_file():
        return snapshot
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_MODEL_ID, revision=HF_REVISION))


def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator:
    """Readiness discovery factory. See ``models/common/readiness_check/contract.py``.

    **This is the construction path the precision config has to reach.** The
    readiness runners, the qualitative suite and (later) vLLM all arrive here
    and none of them can pass a Python object, so ``precision`` is accepted as a
    kwarg *and* read from ``QWEN3_PRECISION_CONFIG`` in the environment as a
    path to a ``selected_precision_config.json``. Unset -- which is every run to
    date -- means ``DEFAULT_PRECISION``, i.e. the shipped policy, so this
    default is the one the stage-07 goal asks for rather than a JSON field that
    hard-coded model code ignores.
    """
    snapshot = _resolve_snapshot(kwargs.pop("model_path", os.getenv("QWEN3_CODER_30B_MODEL_PATH")))
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    max_batch_size = int(kwargs.pop("max_batch_size", 1))
    max_context_len = int(kwargs.pop("max_context_len", MAX_CONTEXT))
    override_num_layers = kwargs.pop("override_num_layers", None)
    num_layers = NUM_LAYERS if override_num_layers is None else int(override_num_layers)
    page_block_size = int(kwargs.pop("page_block_size", 32))
    rope_cache_len = int(kwargs.pop("rope_cache_len", 8192))
    precision = kwargs.pop("precision", os.getenv("QWEN3_PRECISION_CONFIG") or None)
    if kwargs:
        raise TypeError(f"unsupported build_generator kwargs: {sorted(kwargs)}")
    model = Qwen3CoderModel.from_checkpoint(
        snapshot,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_cache_len=max_context_len,
        num_layers=num_layers,
        page_block_size=page_block_size,
        rope_cache_len=rope_cache_len,
        precision=precision,
    )
    return Qwen3CoderGenerator(model, tokenizer)


__all__ = ["Qwen3CoderGenerator", "build_generator"]
