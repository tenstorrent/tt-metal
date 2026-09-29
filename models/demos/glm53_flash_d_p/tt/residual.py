# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""mHC residual (attn_residual / ffn_residual) on device, replicated, no CCL, no weights.

out_m = post[:, m] * y + sum_n comb[:, n, m] * x_n (glm_ref.hc_residual: post * y + comb^T @ x).
x [1, 1, S, n*H] (streams packed along the last dim, the reference's token-major [S * n, H]), hc [1, 1, S, (2+n)*n]
fp32 = [pre n | post n | comb n*n row-major], y [1, 1, S, H]. Accumulates in fp32, rounds to bf16 once.

GLM_RESIDUAL_MIX selects the path:
- matmul (default): per 32-token tile b, X'[b] = [x_0; ..; x_{n-1}; y] rows (stream, token) [(n+1)*32, H] and
  out[b] = Mix[b] @ X'[b], Mix[b][(m, t), (k, t')] = delta(t, t') * coef_km(token 32b + t) (coef = comb[:, k, m],
  post[:, m] for k = n): one matmul batched over the S/32 token tiles, HiFi4, fp32 DEST, one bf16 rounding.
  The FPU reads the fp32 coefficients as tf32 (x and y are bf16, exact).
  Mix is built on the device from hc with a load-time 0/1 selector matmul and a load-time diagonal mask.
- addcmul: DeepSeek's TtMHCWrap.hc_post pattern, 16 fp32 multiply / addcmul ops on the stream slices.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import _streams
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate

TILE = 32
MIX_MODES = ("matmul", "addcmul")


def residual_mix_mode() -> str:
    mode = os.environ.get("GLM_RESIDUAL_MIX", "matmul")
    assert mode in MIX_MODES, f"GLM_RESIDUAL_MIX={mode!r}, want one of {MIX_MODES}"
    return mode


def _f32(t):
    return t if t.dtype == ttnn.float32 else ttnn.typecast(t, ttnn.float32)


class TtHcResidual(LightweightModule):
    def __init__(self, mesh=None, n=4, out_dtype=ttnn.bfloat16, mode: str | None = None):
        self.n, self.out_dtype = n, out_dtype
        self.mode = mode or residual_mix_mode()
        if self.mode == "matmul":
            assert mesh is not None, "the matmul residual mix builds its constants on the mesh at load time"
            k_hc, w = (2 + n) * n, (n + 1) * TILE
            sel = torch.zeros(k_hc, n * w)  # sel[hc column of coef_km, m * w + k * 32 + t'] = 1
            for m in range(n):
                for k in range(n + 1):
                    col = n + m if k == n else 2 * n + k * n + m
                    sel[col, m * w + k * TILE : m * w + (k + 1) * TILE] = 1.0
            self.sel = replicate(mesh, sel.reshape(1, 1, k_hc, n * w), dtype=ttnn.float32)
            diag = torch.eye(TILE).repeat(1, n * (n + 1))  # delta(t, t') in every (m, k) block
            self.diag = replicate(mesh, diag.reshape(1, 1, TILE, n * w), dtype=ttnn.float32)
            self.k_hc = k_hc
            self.ckc = hifi4_config(fp32_acc=True)

    def __call__(self, x, hc, y):
        """x [1, 1, S, n*H], hc [1, 1, S, K >= (2+n)*n] fp32, y [1, 1, S, H] -> [1, 1, S, n*H] out_dtype."""
        if self.mode == "matmul" and x.shape[-2] % TILE == 0:
            return self._matmul_mix(x, hc, y)
        return self._addcmul_mix(x, hc, y)

    def _matmul_mix(self, x, hc, y):
        n = self.n
        S, H = x.shape[-2], y.shape[-1]
        B, w = S // TILE, (n + 1) * TILE
        # Mix[b] [(m, t), (k, t')]: select the coefficient columns (exact 0/1 matmul), mask to the diagonal.
        h = hc if hc.shape[-1] == self.k_hc else ttnn.slice(hc, [0, 0, 0, 0], [1, 1, S, self.k_hc])
        r = ttnn.matmul(h, self.sel, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        if h is not hc:
            ttnn.deallocate(h)
        rm = ttnn.multiply(ttnn.reshape(r, [1, B, TILE, n * w]), self.diag)
        ttnn.deallocate(r)
        blocks = [ttnn.slice(rm, [0, 0, 0, m * w], [1, B, TILE, (m + 1) * w]) for m in range(n)]
        ttnn.deallocate(rm)
        mix = ttnn.concat(blocks, dim=2)
        for t in blocks:
            ttnn.deallocate(t)
        # X'[b] = [x_0; ..; x_{n-1}; y] (rows: stream, token), tile-row copies only.
        parts = [ttnn.reshape(p, [1, B, TILE, H]) for p in _streams(x, n)]
        parts.append(ttnn.reshape(y, [1, B, TILE, H]))
        xs = ttnn.concat(parts, dim=2)
        for p in parts[:-1]:
            ttnn.deallocate(p)
        o = ttnn.matmul(mix, xs, dtype=self.out_dtype, compute_kernel_config=self.ckc)  # [1, B, (m, t), H]
        ttnn.deallocate(mix)
        ttnn.deallocate(xs)
        out = [
            ttnn.reshape(ttnn.slice(o, [0, 0, m * TILE, 0], [1, B, (m + 1) * TILE, H]), [1, 1, S, H]) for m in range(n)
        ]
        ttnn.deallocate(o)
        z = ttnn.concat(out, dim=-1)
        for t in out:
            ttnn.deallocate(t)
        return z

    def _addcmul_mix(self, x, hc, y):
        n = self.n
        S = x.shape[-2]
        res = _streams(x, n)
        res32 = [_f32(r) for r in res]
        if res32[0] is not res[0]:
            for r in res:
                ttnn.deallocate(r)
        y32 = _f32(y)
        h32 = _f32(hc)
        col = lambda k: ttnn.slice(h32, [0, 0, 0, k], [1, 1, S, k + 1])  # noqa: E731
        out = []
        for m in range(n):
            p = col(n + m)
            o = ttnn.multiply(p, y32)
            ttnn.deallocate(p)
            for i in range(n):
                c = col(2 * n + i * n + m)
                o2 = ttnn.addcmul(o, res32[i], c)
                ttnn.deallocate(o)
                ttnn.deallocate(c)
                o = o2
            if self.out_dtype != ttnn.float32:
                o2 = ttnn.typecast(o, self.out_dtype)
                ttnn.deallocate(o)
                o = o2
            out.append(o)
        for t in res32:
            ttnn.deallocate(t)
        if y32 is not y:
            ttnn.deallocate(y32)
        if h32 is not hc:
            ttnn.deallocate(h32)
        z = ttnn.concat(out, dim=-1)
        for t in out:
            ttnn.deallocate(t)
        return z


def build_residual(cfg, mesh=None) -> TtHcResidual:
    return TtHcResidual(mesh, n=cfg.hc_mult)
