# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefix caching for the hybrid (GDN + attention) fast path.

vLLM's automatic prefix caching keeps full-attention KV pages for block-aligned prompt prefixes and starts a request
at ``start_pos = cached_tokens``. Attention layers then just read the shared pages; the GDN layers instead need their
recurrent + conv state at exactly that position. This store keeps device snapshots of one slot's GDN state (all 30
layers) keyed by a chained hash of the 32-token blocks of the prefix (same block granularity as the KV pages):

- After prefilling a request, the state at the last 32-aligned position <= the prompt end is snapshotted (the
  prefill is split there when the end is not aligned). In agent loops the next turn's prompt extends the previous
  prompt (the chat template drops earlier thinking), so vLLM's hit lands exactly on such a snapshot.
- On a hit (``start > 0`` for a new request) the longest snapshotted prefix <= start is loaded into the slot and the
  prefill resumes from there (re-prefilling [snapshot, start) if needed); with no snapshot it prefills from 0. Either
  way the result is the same computation as a full prefill; only the work skipped differs.

Device memory per snapshot: 30 x (8x128x128 fp32 + 32x2048 bf16) per chip ~= 19.6 MB; capacity QWEN_PREFIX_SNAPSHOTS.
All pools are allocated before decode-trace capture (trace safety) and the save/load programs are warmed then too.
"""

import hashlib
from collections import OrderedDict

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
BLOCK = 32


def block_hashes(tokens, upto):
    """{prefix_len: digest} for every 32-aligned prefix length <= upto (chained, so a digest names the whole prefix)."""
    data = torch.as_tensor(tokens[:upto]).to(torch.int32).numpy().tobytes()
    out, h = {}, b""
    for k in range(upto // BLOCK):
        h = hashlib.blake2b(h + data[k * BLOCK * 4 : (k + 1) * BLOCK * 4], digest_size=16).digest()
        out[(k + 1) * BLOCK] = h
    return out


class GDNSnapshotStore:
    def __init__(self, generator, capacity):
        from .fast_gdn import VH, D

        self.gen = generator
        self.mesh = generator.mesh
        self.capacity = capacity
        self.layers = sorted(generator.fstate)
        self.rec, self.conv = {}, {}
        for li in self.layers:
            self.rec[li] = ttnn.zeros(
                (capacity, VH, D, D), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=DRAM
            )
            self.conv[li] = ttnn.zeros(
                (capacity, 1, 32, 2048),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
        # MTP drafter (QWEN_SPEC): its pending hidden row (final hidden of the snapshot's last position)
        self.mtp_rows = (
            ttnn.zeros(
                (capacity, 1, 32, 2048),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
            )
            if getattr(generator, "mtp", None) is not None
            else None
        )
        self.row_mask = []
        for slot in range(32):
            m = torch.zeros(1, 1, 32, 1)
            m[0, 0, slot, 0] = 1
            self.row_mask.append(
                ttnn.from_torch(
                    m,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=DRAM,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
            )
        self.entries = OrderedDict()  # digest -> pool index (LRU order)
        self.free = list(range(capacity))
        self.stats = dict(saves=0, hits=0, partial_hits=0, misses=0, tokens_skipped=0)

    # ------------------------------------------------------------------ device copies
    def _save(self, slot, index):
        from .fast_gdn import FastGDN

        for li in self.layers:
            state = self.gen.fstate[li]
            view, i = FastGDN.slot_view(state, slot)
            ttnn.fill_cache(self.rec[li], view[i : i + 1], index)
            rows = [ttnn.to_layout(c[:, :, slot : slot + 1, :], ttnn.ROW_MAJOR_LAYOUT) for c in state["conv"]]
            block = ttnn.pad(ttnn.concat(rows, dim=2), [(0, 0), (0, 0), (0, 32 - len(rows)), (0, 0)], 0.0)
            ttnn.fill_cache(self.conv[li], ttnn.to_layout(block, ttnn.TILE_LAYOUT), index)
        if self.mtp_rows is not None:
            row = ttnn.to_layout(self.gen.mtp.pending[:, :, slot : slot + 1, :], ttnn.ROW_MAJOR_LAYOUT)
            row = ttnn.pad(row, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
            ttnn.fill_cache(self.mtp_rows, ttnn.to_layout(row, ttnn.TILE_LAYOUT), index)

    def _load(self, slot, index):
        from .fast_gdn import FastGDN

        consts = self.gen.prefill_executor.consts
        for li in self.layers:
            state = self.gen.fstate[li]
            view, i = FastGDN.slot_view(state, slot)
            ttnn.fill_cache(view, self.rec[li][index : index + 1], i)
            block = self.conv[li][index : index + 1]
            for k, c in enumerate(state["conv"]):
                self._put_row(c, block[:, :, k : k + 1, :], slot)
        if self.mtp_rows is not None:
            self._put_row(self.gen.mtp.pending, self.mtp_rows[index : index + 1][:, :, 0:1, :], slot)

    def _put_row(self, dst, row, slot):
        """dst[..., slot, :] = row (exact select, one broadcast where + copy instead of the 5-op one-hot mix)."""
        ttnn.copy(ttnn.where(self.row_mask[slot], row, dst), dst)

    def warm(self, slot=0):
        """Build the save/load programs (call before decode-trace capture)."""
        index = self.free[0]
        self._save(slot, index)
        self._load(slot, index)

    # ------------------------------------------------------------------ API
    def save(self, slot, digest):
        if digest in self.entries:
            self.entries.move_to_end(digest)
            return
        if not self.free:
            _, index = self.entries.popitem(last=False)
            self.free.append(index)
        index = self.free.pop()
        self._save(slot, index)
        self.entries[digest] = index
        self.stats["saves"] += 1

    def restore(self, slot, tokens, start):
        """Load the longest snapshotted prefix <= start into `slot`; return its length (0 = none)."""
        hashes = block_hashes(tokens, start)
        for length in sorted(hashes, reverse=True):
            index = self.entries.get(hashes[length])
            if index is not None:
                self.entries.move_to_end(hashes[length])
                self._load(slot, index)
                self.stats["hits" if length == start else "partial_hits"] += 1
                self.stats["tokens_skipped"] += length
                return length
        self.stats["misses"] += 1
        return 0
