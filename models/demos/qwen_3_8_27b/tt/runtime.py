# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill runtime: ``compile`` / ``make_chunk_input`` / ``prefill_chunk``.

Shape of the API and the chunk-range contract from minimax_m3/tt/tt_prefill_runtime.py:

  0 <= slot_id < num_users
  actual_start % chunk_size == 0                      (block-cyclic period of the KV table)
  actual_start < actual_end <= actual_start + chunk_size
  actual_start + chunk_size <= cache capacity, actual_end <= spec max_seq_len
  actual_end % 32 == 0                                (ring SDPA KV-pad rotation is tile aligned)

Added for the hybrid model: the Gated-DeltaNet layers carry a recurrent state, so a user's chunks
must arrive in order — ``actual_start`` must equal that user's current length (``actual_start == 0``
restarts the user and zeroes its GDN state). An out-of-contract chunk fails loudly.

``prefill_one_shot`` runs a whole prompt in a single forward (the KV table period is then the
one-shot length); it is the P1 path and the verifier's ``PREFILL_CHUNKED=0`` mode.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.qwen_3_8_27b.tt.context import PrefillCtx


class TtPrefillRuntime:
    def __init__(self, model, *, chunk_size: int, max_seq_len: int, capacity: int, num_users: int = 1):
        sp = model.mc.sp
        assert chunk_size % (32 * sp) == 0, f"chunk_size {chunk_size} must be a multiple of 32*sp"
        assert max_seq_len % (32 * sp) == 0, f"max_seq_len {max_seq_len} must be a multiple of 32*sp"
        assert capacity >= max_seq_len and capacity % chunk_size == 0
        self.model = model
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.capacity = capacity
        self.num_users = num_users
        self.user_len = [0] * num_users

    def allocate_caches(self):
        return self.model.allocate_caches(self.capacity, num_users=self.num_users)

    # ---- inputs ----
    def make_chunk_input(self, token_ids) -> ttnn.Tensor:
        """Exactly ``chunk_size`` token ids (pad the tail) -> SP-sharded device tokens."""
        t = torch.as_tensor(token_ids).flatten()
        assert (
            t.numel() == self.chunk_size
        ), f"chunk input must be exactly chunk_size={self.chunk_size} tokens, got {t.numel()}"
        return self.model.embedding.make_tokens(t)

    # ---- compile ----
    def compile(self, caches):
        """Build every program once: a first chunk (live-KV ring SDPA) and a later chunk (cache-read
        ring SDPA). State is reset afterwards; the KV rows written are overwritten by real prefill."""
        dummy = torch.zeros(self.chunk_size, dtype=torch.int64)
        for start in (0, self.chunk_size):
            if start + self.chunk_size > self.capacity:
                break
            out = self.prefill_chunk(self.make_chunk_input(dummy), caches, 0, start, start + self.chunk_size)
            ttnn.deallocate(out)
        ttnn.synchronize_device(self.model.mc.mesh_device)
        self.reset_user(caches, 0)
        logger.info("prefill runtime compiled")

    def reset_user(self, caches, slot_id: int):
        caches.reset_gdn(slot_id)
        self.user_len[slot_id] = 0

    # ---- run ----
    def prefill_chunk(self, input_tensor, caches, slot_id: int, actual_start: int, actual_end: int):
        """One chunk: returns the final-normed hidden ``[1, 1, chunk/sp, H]`` (SP-sharded)."""
        assert 0 <= slot_id < self.num_users, f"slot_id {slot_id} out of range [0, {self.num_users})"
        assert actual_start % self.chunk_size == 0, f"actual_start {actual_start} not a multiple of chunk_size"
        assert (
            actual_start < actual_end <= actual_start + self.chunk_size
        ), f"chunk range [{actual_start}, {actual_end}) outside [start, start + {self.chunk_size}]"
        assert actual_start + self.chunk_size <= self.capacity, "chunk runs past the KV cache capacity"
        assert actual_end <= self.max_seq_len, f"actual_end {actual_end} > max_seq_len {self.max_seq_len}"
        assert actual_end % 32 == 0, "actual_end must be tile aligned (ring SDPA KV-pad rotation)"
        if actual_start == 0:
            self.reset_user(caches, slot_id)
        assert (
            actual_start == self.user_len[slot_id]
        ), f"GDN state is sequential: slot {slot_id} is at {self.user_len[slot_id]}, chunk starts at {actual_start}"
        ctx = PrefillCtx(
            caches=caches, user_id=slot_id, start=actual_start, valid_end=actual_end, tokens=self.chunk_size
        )
        out = self.model.forward(input_tensor, ctx)
        # a padded (short) chunk ends the sequence: no continuation is possible after it
        self.user_len[slot_id] = actual_end if actual_end == actual_start + self.chunk_size else -1
        return out

    def prefill_one_shot(self, token_ids, caches, slot_id: int = 0):
        """The whole prompt in one forward. Its length must be a multiple of 32*sp (no padding here)."""
        t = torch.as_tensor(token_ids).flatten()
        T = t.numel()
        assert T % (32 * self.model.mc.sp) == 0, f"one-shot length {T} must be a multiple of 32*sp"
        assert T <= self.max_seq_len and self.capacity % T == 0, "one-shot length must divide the cache capacity"
        self.reset_user(caches, slot_id)
        ctx = PrefillCtx(caches=caches, user_id=slot_id, start=0, valid_end=T, tokens=T)
        out = self.model.forward(self.model.embedding.make_tokens(t), ctx)
        self.user_len[slot_id] = -1
        return out
