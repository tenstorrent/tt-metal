# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 mHC residual (attn_residual / ffn_residual, reference/xing_ref.py:hc_residual) on the 4x2 mesh.

    out stream i = post[:, i] * y + sum_j comb[:, i, j] * x_j      (HF: post * out + matmul(comb, residual))

hc = [pre 4 | post 4 | comb 16 row-major], so post_i is column 4 + i and comb[i, j] is column 8 + 4 i + j. This is
comb, not glm53's comb^T (glm53 tt/residual.py reads column 2n + k n + m for output m).

Per chip (r, c): x [1, 1, S/4, 4 x 1792] fp32 streams (tt/layout.py, stream-major), hc [1, 1, S/4, 24] fp32
(replicated over axis 1), y [1, 1, S/4, 1792] (attn_out, column split from the o_proj reduce_scatter) ->
[1, 1, S/4, 4 x 1792] fp32. Every chip mixes its own rows and hidden columns: no collective, no weight.

XING_RESIDUAL_MIX selects the path:
- fused (default): ttnn.bringup.mhc_post (bring-up fork mhc_post_ttnn) in one program, with comb_transposed=False
  (Xing's comb, not the op's default comb^T). post / comb are sliced from hc (columns 4..7 / 8..23, row-major comb,
  which is the op's comb[j*n + i] for output j). fp32 mix in DEST on the SFPU, like addcmul.
- addcmul: 4 x (multiply + 4 x addcmul), all fp32 on the SFPU, so the fp32 streams keep full precision (the P.2
  baseline, 90 ms per chunk).
- matmul: glm53's block-diagonal Mix (load-time 0/1 selector with the Xing comb index, diagonal mask) as one batched
  HiFi4 fp32-DEST matmul. The FPU reads the fp32 streams as tf32, so this rounds the residual each layer; kept for
  comparison only.
"""

from __future__ import annotations

import os

import torch

import ttnn

from .layout import HC

TILE = 32
MIX_MODES = ("fused", "addcmul", "matmul")


def residual_mix_mode() -> str:
    mode = os.environ.get("XING_RESIDUAL_MIX", "fused")
    assert mode in MIX_MODES, f"XING_RESIDUAL_MIX={mode!r}, want one of {MIX_MODES}"
    return mode


def _f32(t):
    return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)


class TtHcResidual:
    def __init__(self, mesh=None, n: int = HC, mode: str | None = None):
        self.n = n
        self.mode = mode or residual_mix_mode()
        self.k_hc = (2 + n) * n
        if self.mode == "matmul":
            assert mesh is not None, "the matmul residual mix builds its constants on the mesh at load time"
            w = (n + 1) * TILE
            sel = torch.zeros(self.k_hc, n * w)  # sel[hc column of coef(m, k), m * w + k * 32 + t'] = 1
            for m in range(n):
                for k in range(n + 1):
                    col = n + m if k == n else 2 * n + m * n + k  # post_m | comb[m, k]
                    sel[col, m * w + k * TILE : m * w + (k + 1) * TILE] = 1.0
            rep = ttnn.ReplicateTensorToMesh(mesh)
            put = lambda t: ttnn.from_torch(  # noqa: E731
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            self.sel = put(sel.reshape(1, 1, self.k_hc, n * w))
            self.diag = put(torch.eye(TILE).repeat(1, n * (n + 1)).reshape(1, 1, TILE, n * w))
            self.ckc = ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

    def __call__(self, x: ttnn.Tensor, hc: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        if self.mode == "fused":
            return self._fused_mix(x, hc, y)
        if self.mode == "matmul" and x.shape[-2] % TILE == 0:
            return self._matmul_mix(x, hc, y)
        return self._addcmul_mix(x, hc, y)

    def _fused_mix(self, x, hc, y):
        n = self.n
        s4 = x.shape[-2]
        dram = ttnn.DRAM_MEMORY_CONFIG
        h32 = _f32(hc)
        post = ttnn.slice(h32, [0, 0, 0, n], [1, 1, s4, 2 * n], memory_config=dram)
        comb = ttnn.slice(h32, [0, 0, 0, 2 * n], [1, 1, s4, 2 * n + n * n], memory_config=dram)
        if h32 is not hc:
            ttnn.deallocate(h32)
        out = ttnn.bringup.mhc_post(y, x, post, comb, comb_transposed=False)
        ttnn.deallocate(post)
        ttnn.deallocate(comb)
        return out

    def _addcmul_mix(self, x, hc, y):
        n = self.n
        s4, w = x.shape[-2], x.shape[-1] // n
        dram = ttnn.DRAM_MEMORY_CONFIG
        xs = []
        for j in range(n):
            s = ttnn.slice(x, [0, 0, 0, j * w], [1, 1, s4, (j + 1) * w], memory_config=dram)
            s32 = _f32(s)
            if s32 is not s:
                ttnn.deallocate(s)
            xs.append(s32)
        y32 = _f32(y)
        h32 = _f32(hc)
        col = lambda k: ttnn.slice(h32, [0, 0, 0, k], [1, 1, s4, k + 1], memory_config=dram)  # noqa: E731
        out = []
        for i in range(n):
            p = col(n + i)
            o = ttnn.multiply(y32, p, dtype=ttnn.float32, memory_config=dram)
            ttnn.deallocate(p)
            for j in range(n):
                c = col(2 * n + i * n + j)
                o2 = ttnn.addcmul(o, xs[j], c, memory_config=dram)
                ttnn.deallocate(o)
                ttnn.deallocate(c)
                o = o2
            out.append(o)
        for t in xs:
            ttnn.deallocate(t)
        if y32 is not y:
            ttnn.deallocate(y32)
        if h32 is not hc:
            ttnn.deallocate(h32)
        z = ttnn.concat(out, dim=-1, memory_config=dram)
        for t in out:
            ttnn.deallocate(t)
        return z

    def _matmul_mix(self, x, hc, y):
        n = self.n
        S, H = x.shape[-2], x.shape[-1] // n
        B, w = S // TILE, (n + 1) * TILE
        dram = ttnn.DRAM_MEMORY_CONFIG
        h = _f32(hc)
        hk = h if h.shape[-1] == self.k_hc else ttnn.slice(h, [0, 0, 0, 0], [1, 1, S, self.k_hc])
        r = ttnn.matmul(hk, self.sel, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        for t in {id(hk): hk, id(h): h}.values():
            if t is not hc:
                ttnn.deallocate(t)
        rm = ttnn.multiply(ttnn.reshape(r, [1, B, TILE, n * w]), self.diag, memory_config=dram)
        ttnn.deallocate(r)
        blocks = [ttnn.slice(rm, [0, 0, 0, m * w], [1, B, TILE, (m + 1) * w]) for m in range(n)]
        ttnn.deallocate(rm)
        mix = ttnn.concat(blocks, dim=2, memory_config=dram)
        for t in blocks:
            ttnn.deallocate(t)
        parts = [
            ttnn.reshape(ttnn.slice(x, [0, 0, 0, j * H], [1, 1, S, (j + 1) * H]), [1, B, TILE, H]) for j in range(n)
        ]
        y32 = _f32(y)
        parts.append(ttnn.reshape(y32, [1, B, TILE, H]))
        xs = ttnn.concat(parts, dim=2, memory_config=dram)
        for p in parts[:-1]:
            ttnn.deallocate(p)
        if y32 is not y:
            ttnn.deallocate(y32)
        o = ttnn.matmul(mix, xs, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(mix)
        ttnn.deallocate(xs)
        out = [
            ttnn.reshape(ttnn.slice(o, [0, 0, m * TILE, 0], [1, B, (m + 1) * TILE, H]), [1, 1, S, H]) for m in range(n)
        ]
        ttnn.deallocate(o)
        z = ttnn.concat(out, dim=-1, memory_config=dram)
        for t in out:
            ttnn.deallocate(t)
        return z


def build_residual(cfg, mesh=None, mode: str | None = None) -> TtHcResidual:
    return TtHcResidual(mesh, n=cfg.hc_mult, mode=mode)
