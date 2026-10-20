# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``KdaDecode``: the fused KDA (Kimi Delta Attention) single-token decode block as a model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

One program does everything between the packed projection matmuls and o_proj: the causal conv (kernel 4)
+ SiLU on the device-owned bf16 history (tap-major tiles, shifted in place, raw tiles appended), the
per-head L2 norms, the per-key-channel forget gate ``g = lb * sigmoid(exp(A_log) (f + dt_bias))``,
``beta = sigmoid(b)``, the gated delta-rule step per user on the recurrent state (read once, written once,
in place) and the gated RMSNorm ``rmsnorm(o) * w * sigmoid(og)``, packed as bf16 TILE output. Work items
are (head, 32-user tile row, user part); the parts of a (head, row) slot sit in one column of cores: the
part-0 core reads the slot's inputs once and multicasts them, and sums the parts' output rows for the norm.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def selector_tiles() -> torch.Tensor:
    """[33*32, 32] constant (used by the DSA state op): tile r < 32 has a 1 at (0, r) (E_r lifts row r of a block
    into row 0; its transpose places row 0 at row r), tile 32 is all ones (row sums by matmul)."""
    sel = torch.zeros(33, TILE, TILE)
    sel[torch.arange(TILE), 0, torch.arange(TILE)] = 1.0
    sel[32] = 1.0
    return sel.reshape(33 * TILE, TILE)


def kda_selector_tiles() -> torch.Tensor:
    """[34*32, 32] constant of ``KdaDecode``: tile r < 32 is E_r^T (a 1 at (r, 0): e_r in column 0, i.e. a row mask
    under a column broadcast and a column picker under a matmul), tile 32 is all ones (row sums by matmul), tile
    33 is the identity (diagonal tiles by a column broadcast)."""
    sel = torch.zeros(34, TILE, TILE)
    sel[torch.arange(TILE), torch.arange(TILE), 0] = 1.0
    sel[32] = 1.0
    sel[33] = torch.eye(TILE)
    return sel.reshape(34 * TILE, TILE)


ONES_PAGE = 32  # selector page of the all-ones tile (identity at 33)


class KdaDecode:
    """Builds and launches the fused KDA decode program. All tensors interleaved (DRAM or L1); K = V = 128,
    conv kernel 4 (3 history taps). Call ``__call__`` with a pre-allocated output."""

    # CB plan (kept in sync with the kernels)
    CB = dict(
        hist=0, taps=1, gf=2, gog=3, gb=4, cc=5, onw=6, sel=7, selT=8, S=9, out=10, state_out=11, qn=12, kn=13,
        vc=14, dec=15, bcol=16, wg=17, tmp=18, col=19, kd=20, qd=21, kcol=22, decT=23, gather=24, dcol=25, pq=26,
        vread=27, oq=28, u=29, oacc=30, ofin=31,
    )  # fmt: skip
    # per-core semaphores: parts -> part-0 "ready", part-0 -> parts "valid" (multicast of the constant "one"),
    # parts -> part-0 "gather" (output blocks delivered)
    SEM_READY, SEM_VALID, SEM_ONE, SEM_GATHER = 0, 1, 2, 3

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int = 128,
        gate_lower_bound: float = -5.0,
        l2_eps: float = 1e-6,
        norm_eps: float = 1e-5,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    ):
        self.H, self.K = num_heads, head_dim
        self.V = head_dim
        assert head_dim == 128, "the kernels are written for K = V = 128"
        self.lb, self.l2_eps, self.norm_eps = gate_lower_bound, l2_eps, norm_eps
        self.fidelity = math_fidelity

    # ------------------------------------------------------------------ reference
    @staticmethod
    def golden(
        qkv,
        gates,
        conv,
        S,
        taps,
        dt_bias,
        a_exp,
        onw,
        *,
        num_heads: int,
        gate_lower_bound=-5.0,
        l2_eps=1e-6,
        norm_eps=1e-5,
    ):
        """Torch reference of one step. qkv [B, 3*H*K] (bf16-valued), gates [B, H*K + H*V + Hb] fp32, conv [B, 3, 3*H*K],
        S [B, H, K, V], taps [4, 3*H*K], dt_bias / a_exp [H*K], onw [V]. Returns (o [B, H*V], new state, new conv)."""
        B, H = qkv.shape[0], num_heads
        K = V = taps.shape[-1] // (3 * H)
        rows = torch.cat([conv, qkv[:, None, :]], dim=1)  # [B, 4, W], oldest first
        cv = torch.nn.functional.silu((rows * taps[None]).sum(1))
        q, k, v = (t.reshape(B, H, K) for t in cv.split(H * K, dim=-1))
        l2 = lambda t: t / torch.sqrt((t * t).sum(-1, keepdim=True) + l2_eps)
        q = l2(q) * K**-0.5
        k = l2(k)
        f = gates[:, : H * K]
        og = gates[:, H * K : H * K + H * V].reshape(B, H, V)
        b = gates[:, H * K + H * V : H * K + H * V + H]
        decay = torch.exp(gate_lower_bound * torch.sigmoid(a_exp * (f + dt_bias))).reshape(B, H, K)
        beta = torch.sigmoid(b)
        gate = torch.sigmoid(og)
        Sd = S * decay[..., None]
        vread = (Sd * k[..., None]).sum(-2)
        u = beta[..., None] * (v - vread)
        Sn = Sd + k[..., None] * u[:, :, None, :]
        o = (Sn * q[..., None]).sum(-2)
        o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + norm_eps) * onw * gate
        new_conv = torch.cat([conv[:, 1:], qkv[:, None, :]], dim=1)
        return o.reshape(B, H * V), Sn, new_conv

    # ------------------------------------------------------------------ program
    @staticmethod
    def placement(grid, slots: int):
        """Cores for ``slots`` (head, tile row) items: ``parts`` cores per slot, a vertical run in one column
        (NoC multicast rectangle); the largest ``parts`` that fits, else one core per item round-robin."""
        gx, gy = grid.x, grid.y
        parts = 1
        for ry in range(min(TILE, gy), 1, -1):
            if gx * (gy // ry) >= slots:
                parts = ry
                break
        if parts == 1:
            NC = min(slots, gx * gy)
            return 1, [ttnn.CoreCoord(c // gy, c % gy) for c in range(NC)]
        per_col = gy // parts
        cores = []
        for s in range(slots):
            x, y0 = s // per_col, (s % per_col) * parts
            cores += [ttnn.CoreCoord(x, y0 + p) for p in range(parts)]
        return parts, cores

    def _cb(self, idx, dtype, page, pages, cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    def __call__(self, qkv, gates, conv_hist, recurrent, taps, chan_consts, o_norm_w, selectors, out):
        """qkv [1,1,B,3*H*K] bf16 TILE; gates [1,1,B,H*K+H*V+Hb] fp32 TILE; conv_hist [3,B,3*H*K] bf16 TILE
        (tap-major history, tap 0 the oldest; updated in place); recurrent [B,H,K,V] fp32 TILE (updated in place);
        taps [1,1,4,3*H*K] bf16 TILE; chan_consts [1,1,2,H*K] fp32 TILE (row 0 dt_bias, row 1 exp(A_log)); o_norm_w
        [1,1,1,V] fp32 TILE; selectors [34*32,32] bf16 TILE (``kda_selector_tiles``); out [1,1,B,H*V] bf16 TILE
        (pre-allocated). Returns ``out``."""
        H, K, V = self.H, self.K, self.V
        B = recurrent.shape[0]
        Kt, Vt = K // TILE, V // TILE
        Bt = -(-B // TILE)
        HK_t, HV_t = H * Kt, H * Vt
        Wq_t, Wg_t = 3 * HK_t, HK_t + HV_t + (-(-H // TILE))
        kv = Kt * Vt
        device = qkv.device()
        grid = device.compute_with_storage_grid_size()
        parts, core_of = self.placement(grid, H * Bt)
        NI = H * Bt * parts
        NC = len(core_of)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_of])

        f32, f16 = ttnn.float32, ttnn.bfloat16
        t32, t16 = TILE * TILE * 4, TILE * TILE * 2
        cb = self.CB
        plan = [
            ("hist", f16, 12 * Kt), ("taps", f16, 3 * Kt), ("gf", f32, Kt), ("gog", f32, Vt), ("gb", f32, 1),
            ("cc", f32, Kt), ("onw", f32, Vt), ("sel", f16, 2), ("selT", f16, 2), ("S", f32, 2 * kv),
            ("out", f16, Vt), ("state_out", f32, 2 * kv), ("qn", f32, Kt), ("kn", f32, Kt), ("vc", f32, Vt),
            ("dec", f32, Kt), ("bcol", f32, 1), ("wg", f32, Vt), ("tmp", f32, 2 * Kt), ("col", f32, 2),
            ("kd", f32, Kt), ("qd", f32, Kt), ("kcol", f32, Kt), ("decT", f32, Kt),
            ("gather", f32, max(1, parts - 1) * Vt), ("dcol", f32, Kt), ("pq", f32, 4), ("vread", f32, Vt),
            ("oq", f32, Vt), ("u", f32, Vt), ("oacc", f32, 2 * Vt), ("ofin", f32, Vt),
        ]  # fmt: skip
        cbs = [self._cb(cb[name], dt, t32 if dt == f32 else t16, pages, cores) for name, dt, pages in plan]

        it_half = max(2, -(-min(B, 16) // 2))  # SFPU iterations per face (2 rows each) for tile rows with <= 16 users
        compute_ct = [Kt, Vt, Bt, B, parts, f32_bits(self.lb), f32_bits(self.l2_eps), f32_bits(self.norm_eps),
                      f32_bits(K**-0.5), f32_bits(1.0 / V), it_half]  # fmt: skip
        reader_ct = [Kt, Vt, Bt, B, H, Wq_t, Wg_t, HK_t, HV_t, parts, ONES_PAGE]
        for t in (qkv, conv_hist, taps, gates, chan_consts, o_norm_w, selectors, recurrent):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [Kt, Vt, Bt, B, H, parts, HV_t]
        for t in (out, recurrent):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(core_of):
            n_items = (NI - c + NC - 1) // NC if parts == 1 else 1
            s = c // parts
            p0 = device.worker_core_from_logical_core(core_of[s * parts])
            pe = device.worker_core_from_logical_core(core_of[s * parts + parts - 1])
            reader_rt[core.x][core.y] = [
                c, n_items, NC, qkv.buffer_address(), conv_hist.buffer_address(), taps.buffer_address(),
                gates.buffer_address(), chan_consts.buffer_address(), o_norm_w.buffer_address(),
                selectors.buffer_address(), recurrent.buffer_address(), p0.x, p0.y, pe.x, pe.y,
            ]  # fmt: skip
            writer_rt[core.x][core.y] = [c, n_items, NC, out.buffer_address(), recurrent.buffer_address(), p0.x, p0.y]
            compute_rt[core.x][core.y] = [c, n_items, NC]

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_kda_decode.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_kda_decode.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_kda_decode.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=self.fidelity, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False
                ),
            ),
        ]
        sems = [
            ttnn.SemaphoreDescriptor(sid, ttnn.CoreType.WORKER, cores, init)
            for sid, init in ((self.SEM_READY, 0), (self.SEM_VALID, 0), (self.SEM_ONE, 1), (self.SEM_GATHER, 0))
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return ttnn.generic_op([qkv, gates, conv_hist, recurrent, taps, chan_consts, o_norm_w, selectors, out], program)
