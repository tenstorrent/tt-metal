# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 mHC residual (attn_residual / ffn_residual; reference/ds41_ref.py:hc_post, inference Block.hc_post)
on the SP x TP mesh, fp32:

    out stream k = post[:, k] * y + sum_j comb[:, j, k] * x_j          (j = input stream, k = output stream)

mixes = [pre 4 | post 4 | comb 16 row-major (j, k)] (tt/mhc.py:TtHcMixes), so post_k is column 4 + k and comb[j, k]
is column 8 + 4 j + k. That is ttnn.bringup.mhc_post's DEFAULT (comb applied transposed: X'_k = post_k F +
sum_j comb[j n + k] X_j), unlike xing40 (comb_transposed=False).

Per chip (r, c): x [1, 1, S/SP, 4 x H/TP] fp32 streams (tt/layout.py, stream-major), mixes [1, 1, S/SP, 24] fp32
(row-split, replicated over TP), y [1, 1, S/SP, H/TP] fp32 (attn_out: rows over SP, hidden columns over TP) ->
[1, 1, S/SP, 4 x H/TP] fp32. Every chip mixes its own rows and hidden columns: no collective, no weight, no host work.

RESIDUAL_MIX (tt/settings.py) selects the path; both are fp32 SFPU math (no FPU reads, K-30 does not apply):
- fused (default): ttnn.bringup.mhc_post in one program (fp32 weighted sum in SFPU registers).
- addcmul: per output stream multiply + 4 x addcmul (the op chain, kept for comparison).
"""

from __future__ import annotations

import ttnn

from . import host_alloc
from .ccl import dealloc
from .common import DRAM, cast
from .layout import HC
from .settings import settings

MIX_MODES = ("fused", "addcmul")


def _fp32(y: ttnn.Tensor) -> ttnn.Tensor:
    """y as fp32 (exact): y itself when it already is; the caller frees what it got only when it is not y."""
    return y if y.dtype == ttnn.float32 else ttnn.typecast(y, ttnn.float32, memory_config=DRAM)


class TtHcResidual:
    """Call with (x streams [1, 1, R, n x w] fp32, mixes [1, 1, R, >= (2 + n) n] fp32, y [1, 1, R, w] fp32); returns
    the new streams [1, 1, R, n x w] fp32. Consumes no input."""

    def __init__(self, n: int = HC, mode: str | None = None):
        self.n = n
        self.mode = mode or settings.get("RESIDUAL_MIX")
        assert self.mode in MIX_MODES, (self.mode, MIX_MODES)
        # GLUE_FOLD=on (P.19): mhc_post reads post / comb from the packed mixes (post_col=n, comb_col=2n: no slices)
        # and its stats pass takes a bf16 y as is (no typecast to fp32)
        self.fold = settings.get("GLUE_FOLD") == "on"
        # FORK_OUT_BUFFERS=on (P.16, tt/host_alloc.reserve_forks): fused()'s stats row / partials scratch sites
        self.out_sites = (None, None)
        self._scratch_shapes: dict = {}

    def __call__(self, x: ttnn.Tensor, mix: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
        n = self.n
        # y bf16: the attention's bf16 reduce-scatter result under ATTN_GLUE=on (P.15); mhc_post reads it as is
        # x bf16: BF16_ACTIVATIONS=on (P.20), the new streams written bf16 by mhc_post (one rounding of the fp32 mix)
        assert x.dtype in (ttnn.float32, ttnn.bfloat16) and mix.dtype == ttnn.float32, (x.dtype, mix.dtype)
        assert y.dtype in (ttnn.float32, ttnn.bfloat16), y.dtype
        rows, w = x.shape[-2], y.shape[-1]
        assert x.shape[-1] == n * w and mix.shape[-2] == rows and y.shape[-2] == rows, (x.shape, mix.shape, y.shape)
        assert mix.shape[-1] >= (2 + n) * n, mix.shape
        if self.mode == "fused":
            return self._fused(x, mix, y, rows)
        assert x.dtype == ttnn.float32, "RESIDUAL_MIX=addcmul takes fp32 streams (BF16_ACTIVATIONS=off)"
        yf = _fp32(y)
        out = self._addcmul(x, mix, yf, rows, w)
        if yf is not y:
            dealloc(yf)
        return out

    def fused(self, x: ttnn.Tensor, mix: ttnn.Tensor, y: ttnn.Tensor, hc=None):
        """MHC_FUSED=on (tt/model.py): the new streams, and, when ``hc`` (the tt/mhc.py:TtHcMixes that reads them
        next) is given, that step's partial row [1, 1, R, 32] (x' @ fn^T + sum x'^2), emitted by the same mhc_post
        pass (stats_weight); (streams, row or None). The pending pre stays the caller's (the whole mixes)."""
        if hc is None or self.mode != "fused":
            return TtHcResidual.__call__(self, x, mix, y), None
        rows = x.shape[-2]
        n = self.n
        ro = so = None
        if self.out_sites[1] is not None:  # FORK_OUT_BUFFERS=on (P.16): the stats row / scratch's load-time outputs
            ro = host_alloc.fork_out(hc.mesh, self.out_sites[0], (1, 1, rows, 32), ttnn.float32)
            so = host_alloc.fork_out(
                hc.mesh, self.out_sites[1], self._scratch(hc.mesh, rows, y.shape[-1]), ttnn.float32
            )
        if self.fold:  # GLUE_FOLD=on (P.19): the packed mixes and a bf16 y as they are
            out, row = ttnn.bringup.mhc_post(
                y,
                x,
                mix,
                mix,
                stats_weight=hc.fn_t,
                compute_kernel_config=hc.post_cfg,
                post_col=n,
                comb_col=2 * n,
                stats_output=ro,
                stats_scratch=so,
            )
            return out, (row if ro is None else ro)  # (the kept object, DS41-149)
        post = ttnn.slice(mix, [0, 0, 0, n], [1, 1, rows, 2 * n], memory_config=DRAM)
        comb = ttnn.slice(mix, [0, 0, 0, 2 * n], [1, 1, rows, 2 * n + n * n], memory_config=DRAM)
        # stats_weight: fp32 X with fp32 F, or bf16 X (P.20) with bf16 F (mhc_post_ttnn README "Limits")
        yf = _fp32(y) if x.dtype == ttnn.float32 else (y if y.dtype == ttnn.bfloat16 else cast(y, ttnn.bfloat16))
        out, row = ttnn.bringup.mhc_post(
            yf,
            x,
            post,
            comb,
            stats_weight=hc.fn_t,
            compute_kernel_config=hc.post_cfg,
            stats_output=ro,
            stats_scratch=so,
        )
        row = row if ro is None else ro
        if yf is not y:
            dealloc(yf)
        dealloc(post)
        dealloc(comb)
        return out, row

    def _scratch(self, mesh, rows: int, cols: int):
        key = (rows, cols)
        s = self._scratch_shapes.get(key)
        if s is None:
            s = self._scratch_shapes[key] = tuple(host_alloc.post_scratch_shape(mesh, rows, cols))
        return s

    def _fused(self, x, mix, y, rows):
        n = self.n
        if self.fold:  # GLUE_FOLD=on (P.19): post / comb read from the packed mixes
            return ttnn.bringup.mhc_post(y, x, mix, mix, post_col=n, comb_col=2 * n)
        post = ttnn.slice(mix, [0, 0, 0, n], [1, 1, rows, 2 * n], memory_config=DRAM)
        comb = ttnn.slice(mix, [0, 0, 0, 2 * n], [1, 1, rows, 2 * n + n * n], memory_config=DRAM)
        out = ttnn.bringup.mhc_post(y, x, post, comb)  # comb_transposed=True (default): output k reads comb[j n + k]
        dealloc(post)
        dealloc(comb)
        return out

    def _addcmul(self, x, mix, y, rows, w):
        n = self.n
        col = lambda k: ttnn.slice(mix, [0, 0, 0, k], [1, 1, rows, k + 1], memory_config=DRAM)  # noqa: E731
        xs = [ttnn.slice(x, [0, 0, 0, j * w], [1, 1, rows, (j + 1) * w], memory_config=DRAM) for j in range(n)]
        outs = []
        for k in range(n):
            p = col(n + k)
            o = ttnn.multiply(y, p, dtype=ttnn.float32, memory_config=DRAM)
            dealloc(p)
            for j in range(n):
                c = col(2 * n + j * n + k)
                o2 = ttnn.addcmul(o, xs[j], c, memory_config=DRAM)
                dealloc(o)
                dealloc(c)
                o = o2
            outs.append(o)
        for t in xs:
            dealloc(t)
        z = ttnn.concat(outs, dim=-1, memory_config=DRAM)
        for t in outs:
            dealloc(t)
        return z


class TtHcFfnResidual(TtHcResidual):
    """ffn_residual: the same mix with ffn_hc's mixes, plus the pending pre for the next layer (DS41-3): ffn_hc's
    pre, mixes[:, 0:n], its own [1, 1, R, n] fp32 tensor (row-split, TP-replicated). Call with (h_mid streams, ffn_hc
    mixes, mlp_out); returns (out streams [1, 1, R, n x w] fp32, pre [1, 1, R, n] fp32). The [S, 4H + 4] packing is
    only the harness boundary's (hooks._ffn_residual_host_fn); on the device the two stay separate."""

    def __call__(self, x: ttnn.Tensor, mix: ttnn.Tensor, y: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        out = super().__call__(x, mix, y)
        pre = ttnn.slice(mix, [0, 0, 0, 0], [1, 1, mix.shape[-2], self.n], memory_config=DRAM)
        return out, pre


def build_residual(cfg, mode: str | None = None, step: str = "attn_residual") -> TtHcResidual:
    if step == "ffn_residual":
        return TtHcFfnResidual(n=cfg.hc, mode=mode)
    return TtHcResidual(n=cfg.hc, mode=mode)
