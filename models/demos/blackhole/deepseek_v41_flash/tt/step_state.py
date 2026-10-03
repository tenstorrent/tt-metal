# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Position-dependent inputs of the attention layers, derived ON THE DEVICE from the position of each user.

Everything an attention layer needs per step (RoPE rows, the attention mask, the ring-cache slot, the compressed-cache slot,
the RoPE rows of a freshly pooled latent) is a function of the position only. The tables of all positions are computed once
at init and kept in device DRAM; per step the host uploads just the positions ([T] int32 per mesh row) and ``build`` gathers
the rows with ``ttnn.embedding`` (bf16 tables only: values > 256 are split in two exact halves). Layers of one kind (window /
ratio 2 / ratio 1) share one ``DSV41StepState`` and its output dict.
"""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import HEAD_DIM, NH, WINDOW


class DSV41StepState:
    def __init__(self, attn, max_pos=None):
        """``attn``: a representative DSV41Attention / DSV41CompressedAttention of the group (RoPE tables, ratio, sizes)."""
        self.md, self.T = attn.mesh_device, attn.T
        self.rows, self.cols = attn.rows, attn.cols
        self.compressed = hasattr(attn, "ratio")
        self.ratio = getattr(attn, "ratio", 0)
        self.max_comp = getattr(attn, "max_comp", 0)
        P = max_pos or attn.cos_tab.shape[0]
        self.max_pos = P
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
        assert P <= 256 + 1, "integer tables are bf16: positions must stay <= 256 to be exact"
        c, s = attn._rope_inputs(pos)  # [P, 512] full-width tables (C = 1 / S = 0 off the rotated dims)
        self.t = {"C": up(c), "S": up(s), "nS": up(-s)}
        if self.compressed:
            r = self.ratio
            g = (pos + 1 - r).clamp(min=0)  # RoPE position of a freshly pooled latent
            cg, sg = c[g], s[g]
            self.t.update({"Cg": up(cg), "Sg": up(sg)})
            mask = torch.zeros(P, WINDOW + self.max_comp)
            for p in range(P):
                mask[p, p + 1 : WINDOW] = -1e9  # ring slots not filled yet
                mask[p, WINDOW + (p + 1) // r :] = -1e9  # compressed slots not complete yet
            self.t["mask"] = up(mask)
            ones = torch.ones(1, 32)
            slot = WINDOW + pos // r  # compressed-cache slot of the group in progress, up to 383: two exact bf16 halves
            self.t["ring"] = up((pos % WINDOW).reshape(P, 1) * ones)
            self.t["comp_hi"] = up((slot // 2).reshape(P, 1) * ones)
            self.t["comp_lo"] = up((slot % 2).reshape(P, 1) * ones)

    def _rows(self, name, idx, shape, tile=True):
        """Gather table rows for the T positions of this device -> reshaped (row-major) then tiled."""
        e = ttnn.embedding(idx, self.t[name], layout=ttnn.ROW_MAJOR_LAYOUT)  # [T,1,W]
        e = ttnn.reshape(e, shape)
        return ttnn.to_layout(e, ttnn.TILE_LAYOUT) if tile else e

    def _i32_rows(self, e):
        """[T,1,32] bf16 rows (value in every column) -> int32 [T] row-major."""
        v = ttnn.to_layout(ttnn.slice(e, [0, 0, 0], [self.T, 1, 1]), ttnn.TILE_LAYOUT)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(v, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), [self.T])

    def build(self, pos):
        """pos: int32 [T] row-major device tensor (this mesh row's users) -> the ``st`` dict the attention layers read."""
        T, W = self.T, HEAD_DIM
        idx = ttnn.typecast(ttnn.reshape(pos, [T, 1]), ttnn.uint32)
        st = {"pos": pos}
        st["Ch"] = self._rows("C", idx, [1, T, 1, W])
        st["Sh"] = self._rows("S", idx, [1, T, 1, W])
        st["nSh"] = self._rows("nS", idx, [1, T, 1, W])
        if self.compressed:
            st["Cg"] = self._rows("Cg", idx, [1, 1, T, W])
            st["Sg"] = self._rows("Sg", idx, [1, 1, T, W])
            m = self._rows("mask", idx, [T, 1, 1, WINDOW + self.max_comp])  # tile [T,1,1,256]
            st["mask"] = ttnn.repeat(m, [1, 1, NH, 1])
            ring = ttnn.embedding(idx, self.t["ring"], layout=ttnn.ROW_MAJOR_LAYOUT)
            st["pos_ring"] = self._i32_rows(ring)
            hi = ttnn.embedding(idx, self.t["comp_hi"], layout=ttnn.ROW_MAJOR_LAYOUT)
            lo = ttnn.embedding(idx, self.t["comp_lo"], layout=ttnn.ROW_MAJOR_LAYOUT)
            comp = ttnn.add(
                ttnn.multiply(ttnn.to_layout(hi, ttnn.TILE_LAYOUT), 2.0), ttnn.to_layout(lo, ttnn.TILE_LAYOUT)
            )
            st["comp_idx"] = self._i32_rows(ttnn.to_layout(comp, ttnn.ROW_MAJOR_LAYOUT))
        return st
