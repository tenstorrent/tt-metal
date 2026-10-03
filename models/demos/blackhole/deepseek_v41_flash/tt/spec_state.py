# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Position tables for speculative VERIFICATION blocks: every token row (user u, block index j) has its own position
``pos_tok[u*n + j] = base[u] + j`` and everything an attention layer needs is gathered from tables indexed by that position
(like ``DSV41StepState``), with two differences:

  * the compressed layers' ring has ``R = 128 + RING_MARGIN`` slots (slot = pos % R). A speculative block writes positions
    base..base+k before any of its queries read, so the ring must keep the 128 most recent positions of the EARLIEST query of the
    block while the block's later positions are already written: R >= 128 + k. The mask of token q only depends on q:
    slot s is valid iff (q - s) mod R <= 127 and q - ((q - s) mod R) >= 0 (the 128 most recent positions <= q).
  * ratio-2 layers write the group latent to its slot only from the ODD (group-completing) position; even positions skip the write (index -1), so two rows of one block never write the same slot.
"""


import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, NH, WINDOW

RING_MARGIN = 64  # ring = 192 slots: k <= 5 needs 128 + 5, 192 keeps R + max_comp a multiple of 64


class SpecStepState:
    def __init__(self, attn, max_pos=256):
        self.md, self.T = attn.mesh_device, attn.T  # T = token rows per device (users_per_row * n)
        self.n = attn.n
        self.compressed = hasattr(attn, "ratio")
        self.ratio = getattr(attn, "ratio", 0)
        self.R = getattr(attn, "R", 0)
        self.max_comp = getattr(attn, "max_comp", 0)
        P = self.max_pos = max_pos
        assert P <= 256
        pos = torch.arange(P)
        rep = ttnn.ReplicateTensorToMesh(self.md)
        up = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16).contiguous(),
            device=self.md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        # row masks of the n block indices (token row r has block index r % n): paged_update_cache RMWs whole tiles, so several rows of a
        # block writing adjacent positions in ONE call lose updates (probe: tests/test_spec_probe.py); the writes are issued per block index
        # with the other rows' index set to -1 (= skip)
        self.jmask = [
            ttnn.from_torch(
                (torch.arange(self.T) % self.n == j).float().reshape(1, 1, 1, self.T),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for j in range(self.n)
        ]
        c, s = attn._rope_inputs(pos)
        self.t = {"C": up(c), "S": up(s), "nS": up(-s)}
        if self.compressed:
            r, R, mc = self.ratio, self.R, self.max_comp
            g = (pos + 1 - r).clamp(min=0)
            self.t.update({"Cg": up(c[g]), "Sg": up(s[g])})
            mask = torch.full((P, R + mc), -1e9)
            for q in range(P):
                for sl in range(R):
                    d = (q - sl) % R
                    if d <= WINDOW - 1 and q - d >= 0:
                        mask[q, sl] = 0.0
                mask[q, R : R + (q + 1) // r] = 0.0
            self.t["mask"] = up(mask)
            ones = torch.ones(1, 32)
            skip = (
                -1
            )  # even positions of ratio-2 layers write no latent (idx -1 = skip; a write to the last slot of the page corrupted reads, see notes)
            slot = torch.where(
                (pos + 1) % r == 0, R + pos // r, torch.full_like(pos, skip)
            )  # only group-completing positions write their group slot
            self.t["ring"] = up((pos % R).reshape(P, 1) * ones)
            self.t["comp_hi"] = up((slot // 2).reshape(P, 1) * ones)
            self.t["comp_lo"] = up((slot % 2).reshape(P, 1) * ones)

    def _rows(self, name, idx, shape):
        e = ttnn.embedding(idx, self.t[name], layout=ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(e, shape), ttnn.TILE_LAYOUT)

    def _i32_rows(self, e):
        v = ttnn.to_layout(ttnn.slice(e, [0, 0, 0], [self.T, 1, 1]), ttnn.TILE_LAYOUT)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(v, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [self.T])

    def _per_j(self, idx):
        """int32 [T] row-major index -> list over block index j of int32 [T] row-major tensors: idx where the row has block index j, else -1."""
        T = self.T
        f = ttnn.typecast(ttnn.to_layout(ttnn.reshape(idx, [1, 1, 1, T]), ttnn.TILE_LAYOUT), ttnn.float32)
        f1 = ttnn.add(f, 1.0)
        out = []
        for j in range(self.n):
            v = ttnn.subtract(ttnn.multiply(f1, self.jmask[j]), 1.0)
            out.append(ttnn.reshape(ttnn.to_layout(ttnn.typecast(v, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [T]))
        return out

    def build(self, pos):
        """pos: int32 [T] row-major device tensor (token rows of this mesh row) -> the ``st`` dict of the spec attention layers."""
        T, W = self.T, HEAD_DIM
        idx = ttnn.typecast(ttnn.reshape(pos, [T, 1]), ttnn.uint32)
        st = {"pos": pos, "pos_j": self._per_j(pos)}
        st["Ch"] = self._rows("C", idx, [1, T, 1, W])
        st["Sh"] = self._rows("S", idx, [1, T, 1, W])
        st["nSh"] = self._rows("nS", idx, [1, T, 1, W])
        if self.compressed:
            st["Cg"] = self._rows("Cg", idx, [1, 1, T, W])
            st["Sg"] = self._rows("Sg", idx, [1, 1, T, W])
            m = self._rows("mask", idx, [T, 1, 1, self.R + self.max_comp])
            st["mask"] = ttnn.repeat(m, [1, 1, NH, 1])
            ring = ttnn.embedding(idx, self.t["ring"], layout=ttnn.ROW_MAJOR_LAYOUT)
            st["pos_ring"] = self._i32_rows(ring)
            hi = ttnn.embedding(idx, self.t["comp_hi"], layout=ttnn.ROW_MAJOR_LAYOUT)
            lo = ttnn.embedding(idx, self.t["comp_lo"], layout=ttnn.ROW_MAJOR_LAYOUT)
            comp = ttnn.add(
                ttnn.multiply(ttnn.to_layout(hi, ttnn.TILE_LAYOUT), 2.0), ttnn.to_layout(lo, ttnn.TILE_LAYOUT)
            )
            st["comp_idx"] = self._i32_rows(ttnn.to_layout(comp, ttnn.ROW_MAJOR_LAYOUT))
            st["pos_ring_j"] = self._per_j(st["pos_ring"])
            st["comp_idx_j"] = self._per_j(st["comp_idx"])
        return st
