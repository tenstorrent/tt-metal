# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 distributed RMSNorm (attn_norm / ffn_norm / final norm; reference/ds41_ref.py:rms_norm, inference
RMSNorm) on the SP x TP mesh, fp32 throughout:

    out [S, H] = w * (x * rsqrt(mean(x^2) + eps))          eps = norm_eps (1e-20)

Per chip (r, c) the input is the chip's rows and hidden columns [1, 1, S/SP, H/TP] fp32 TILE (tt/collapse.py output);
the output has the same split. Composition (all fp32 on the SFPU, so every product and sum is an exact fp32 operation;
K-30: the FPU, which rms_norm_pre/post_all_gather use, reads fp32 at about TF32; K-12: those ops also leave junk in the
stats tile on an fp32 input):

    ttnn.bringup.mhc_pre_xing_pack(n=1)   the chip's exact sum x^2 over its H/TP columns -> column 3 of a [S/SP, 32]
                                          row (the op overwrites that column of the ``mix`` it is given; any fp32
                                          [S/SP, 32] tile does, so x's first 32 columns serve, no per-call constant)
    slice column 3 -> tt/common.py:tp_sum all_gather over axis 1 + ttnn.add in mesh-column order (exact, bit-identical
                                          on every chip of the row)
    x (1/H) + eps -> ttnn.rsqrt           [S/SP, 1]
    x * rstd ([S/SP, 1] column broadcast) * w ([1, H/TP] row broadcast; the chip's slice of w, fp32, built at load)

The fp8 linears that read the output (q_a / kv) cast it to bf16 (the released fp8_linear's act quant is not
emulated, agent rule 17); the compressor and the indexer's weights_proj read this fp32 output as is.

ffn_norm (``gather=True``) adds one ttnn.all_gather (dim 3, cluster_axis 1; a data move, exact) of the fp32 output
-> [1, 1, S/SP, H] fp32 per chip, replicated over the mesh row: the router reads these whole rows unquantized, and
``quant(y)`` (y in the experts' input dtype, a cast at most) gives the x the shared and the routed experts both
read, computed once per layer.
No host work in __call__.

NORM_IMPL=on (P.18, the default; tt/settings.py): every norm is ttnn.bringup.rms_norm at the fork's normal settings
(HiFi4, fp32 DEST), one call per norm on the rows (fork CHANGELOG 6 options):

    column-split (attn_norm / ffn_norm / final norm):
        rms_norm(x, stats_out=True)                 the chip's partial sums of x^2, [1, 1, S/SP, 32 B] fp32
        ccl.all_gather(dim 3, over TP)              [1, 1, S/SP, 32 B TP] (every element a partial; exact move)
        rms_norm(x, partial_stats=, stats_width=H, weight=, dtype=)
                                                    x rsqrt(sum of the row's partials / H + eps) w, the chip's columns
        (ffn_norm: + the output all_gather, as before)
    whole rows (split=False: q_norm / kv_norm / compressor norm / k_norm):
        rms_norm(x, weight=, [input_window=(c0, c1)], [dtype=])   q_norm / kv_norm read their columns of the fused
                                                    projection output in place (``window``); kv_norm / k_norm write
                                                    bf16 (``dtype``)
The statistic is FPU-grade (x^2 and the row sum read fp32 at TF32, DS41-44), well inside the norms' bf16 floor.
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, TP_AXIS, act_dtype, cast, fp32_acc, tp_sum
from .settings import settings

PACK_N = 1  # mhc_pre_xing_pack with one "stream": the sum of squares lands in column n (n + 2) = 3
SS_COL = PACK_N * (PACK_N + 2)


class TtDistributedRmsNorm:
    """One column-split RMSNorm. Call with the per-chip [1, 1, S/SP, H/TP] fp32 TILE; returns [1, 1, S/SP, H/TP]
    fp32 TILE (same split; ``dtype``: what the last multiply writes, e.g. bf16). Does not consume its input. ``split=False``: the rows are whole on every chip ([.., H],
    replicated over TP, e.g. q_norm / kv_norm): the same chain with a replicated weight and no exchange."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float, split: bool = True, gather: bool = False, dtype=None):
        assert split or not gather, "gather needs a column-split norm"
        hidden = weight.numel()
        self.gather = gather and mesh.shape[TP_AXIS] > 1
        tp = mesh.shape[TP_AXIS] if split else 1
        assert hidden % (32 * tp) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        self.mesh, self.hidden, self.eps, self.split, self.tp = mesh, hidden, float(eps), split, tp
        # ffn_norm (gather) writes FFN_NORM_DTYPE (P.8): bf16 = the last multiply's packer rounds the row once, the
        # all-gather moves half the bytes, the router and the experts read it as is
        bf16 = gather and settings.get("FFN_NORM_DTYPE") == "bf16"
        self.out_dtype = ttnn.bfloat16 if bf16 else ttnn.float32
        if dtype is not None:  # ROPE_IMPL=on (P.17): kv_norm / k_norm write the bf16 the stock rope reads;
            # BF16_ACTIVATIONS=on (P.20): attn_norm, q_norm, the compressor norm and the final norm write bf16 too
            assert not gather, "gather norms take FFN_NORM_DTYPE"
            self.out_dtype = dtype
        self.inv_h = 1.0 / float(hidden)
        self.impl = settings.get("NORM_IMPL")  # P.18: on = ttnn.bringup.rms_norm, off = the fp32 SFPU composition
        # the fork's normal settings: HiFi4 (its default), fp32 DEST (the statistic; FP32_ACC needed), exact SFPU
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc(True),
            packer_l1_acc=False,
        )
        # w as [1, 1, 1, H] fp32 TILE, the last dim split over the TP axis (chip column c holds w[c H/TP, (c+1) H/TP)),
        # replicated over the SP axis; split=False: replicated everywhere.
        dims = ((None, 3) if TP_AXIS == 1 else (3, None)) if split else (None, None)
        self.weight = ttnn.from_torch(
            weight.float().reshape(1, 1, 1, hidden).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )

    def rstd(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, R, H/TP] fp32 -> rsqrt(mean over the full row of x^2 + eps) [1, 1, R, 1] fp32, replicated over TP."""
        rows = x.shape[-2]
        mix = ttnn.slice(x, [0, 0, 0, 0], [1, 1, rows, 32], memory_config=DRAM)
        packed = ttnn.bringup.mhc_pre_xing_pack(mix, x, n=PACK_N)
        dealloc(mix)
        part = ttnn.slice(packed, [0, 0, 0, SS_COL], [1, 1, rows, SS_COL + 1], memory_config=DRAM)
        dealloc(packed)
        if self.split:
            ss = tp_sum(self.mesh, part)
            dealloc(part)
        else:
            ss = part
        ms = ttnn.multiply(ss, self.inv_h, dtype=ttnn.float32, memory_config=DRAM)
        dealloc(ss)
        t = ttnn.add(ms, self.eps, dtype=ttnn.float32, memory_config=DRAM)
        dealloc(ms)
        rs = ttnn.rsqrt(t, memory_config=DRAM)
        dealloc(t)
        return rs

    def _fork(self, x: ttnn.Tensor, window=None) -> ttnn.Tensor:
        """NORM_IMPL=on (P.18): the norm through ttnn.bringup.rms_norm (module docstring). ``window`` (split=False
        only): normalize x's columns [c0, c1) in place of a slice."""
        kw = dict(epsilon=self.eps, weight=self.weight, compute_kernel_config=self.ckc, memory_config=DRAM)
        if self.out_dtype != x.dtype:
            kw["dtype"] = self.out_dtype
        if not self.split:
            if window is not None and tuple(window) != (0, x.shape[-1]):
                kw["input_window"] = tuple(window)
            return ttnn.bringup.rms_norm(x, **kw)
        assert window is None, "a column-split norm takes its whole input"
        part = ttnn.bringup.rms_norm(x, stats_out=True, compute_kernel_config=self.ckc, memory_config=DRAM)
        if self.tp > 1:
            gathered = ccl.all_gather(part, 3, TP_AXIS, site="norm.stats")  # (HOST_ALLOC)
            dealloc(part)
        else:
            gathered = part
        y = ttnn.bringup.rms_norm(x, partial_stats=gathered, stats_width=self.hidden, **kw)
        dealloc(gathered)
        if not self.gather:
            return y
        full = ccl.all_gather(y, 3, TP_AXIS)  # ffn_norm: whole token rows on every chip of the mesh row
        dealloc(y)
        return full

    def __call__(self, x: ttnn.Tensor, window=None) -> ttnn.Tensor:
        if self.impl == "on":
            width = (window[1] - window[0]) if window is not None else x.shape[-1]
            assert width * self.tp == self.hidden, (x.shape, window, self.hidden)
            return self._fork(x, window)
        if window is not None:  # off (the P.17 path): the caller's slice, here
            rows = x.shape[-2]
            xs = ttnn.slice(x, [0, 0, 0, window[0]], [1, 1, rows, window[1]], memory_config=DRAM)
            y = self(xs)
            dealloc(xs)
            return y
        assert x.dtype == ttnn.float32, f"NORM_IMPL=off takes fp32 rows ({x.dtype}; BF16_ACTIVATIONS=on needs on)"
        assert x.shape[-1] * self.tp == self.hidden, (x.shape, self.hidden)
        rs = self.rstd(x)
        xn = ttnn.multiply(x, rs, dtype=ttnn.float32, memory_config=DRAM)
        dealloc(rs)
        y = ttnn.multiply(xn, self.weight, dtype=self.out_dtype, memory_config=DRAM)
        dealloc(xn)
        if not self.gather:
            return y
        # ffn_norm: whole token rows on every chip of the mesh row (router, dispatch, shared expert); chip column c's
        # slice lands at columns [c H/TP, (c+1) H/TP) (mesh-column order), so the rows are the reference's.
        full = ccl.all_gather(y, 3, TP_AXIS)
        dealloc(y)
        return full

    @staticmethod
    def quant(y: ttnn.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        """y in the experts' input dtype (a cast, or y itself when already ``dtype``; FIX.8: no fp8 emulation)."""
        return cast(y, dtype)


# norm steps -> checkpoint weight under layers.<i>.
NORM_WEIGHTS = {"attn_norm": "attn_norm.weight", "ffn_norm": "ffn_norm.weight"}
# norm steps whose output is gathered over the mesh columns (whole rows, replicated within a mesh row).
GATHERED_NORMS = {"ffn_norm"}


def build_norm(
    mesh, ck, cfg, layer: int, step: str, gather: bool | None = None, rows: bool = False
) -> TtDistributedRmsNorm:
    """gather: None = GATHERED_NORMS; False on ffn_norm: the column split kept (MOE_HEAD=fused gathers x itself), in
    FFN_NORM_DTYPE. ``rows`` (attn_norm, ATTN_SP32=on, P.10): the input is whole hidden rows (the chip's own rows):
    split=False, the whole weight replicated, one local norm (no statistics gather)."""
    w = ck.weight(f"layers.{layer}.{NORM_WEIGHTS[step]}", torch.float32)
    if rows:
        assert step not in GATHERED_NORMS, step
        return TtDistributedRmsNorm(mesh, w, cfg.eps, split=False, dtype=act_dtype())
    gathered = step in GATHERED_NORMS
    gather = gathered if gather is None else gather
    # BF16_ACTIVATIONS=on (P.20): attn_norm writes bf16 (the GPU code's RMSNorm .to(dtype)); ffn_norm: FFN_NORM_DTYPE
    dtype = None if gather else act_dtype()
    if gathered and not gather:
        dtype = ttnn.bfloat16 if settings.get("FFN_NORM_DTYPE") == "bf16" else None
    return TtDistributedRmsNorm(mesh, w, cfg.eps, gather=gather, dtype=dtype)
