# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``KdaDecode``: the fused KDA (Kimi Delta Attention) single-token decode block as a model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

One program does everything between the packed projection matmuls and o_proj: the causal conv (kernel 4)
+ SiLU on the device-owned bf16 history (read, shifted in place, raw rows appended), the per-head L2
norms, the per-key-channel forget gate ``g = lb * sigmoid(exp(A_log) (f + dt_bias))``, ``beta =
sigmoid(b)``, the gated delta-rule step per user on the recurrent state (read once, written once, in
place) and the gated RMSNorm ``rmsnorm(o) * w * sigmoid(og)``. Work items are (head, 32-user tile row,
user part) dealt round-robin over the cores.
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
    """[33*32, 32] constant: tile r < 32 has a 1 at (0, r) (E_r lifts row r of a block into row 0; its transpose
    places row 0 at row r), tile 32 is all ones (row sums by matmul)."""
    sel = torch.zeros(33, TILE, TILE)
    sel[torch.arange(TILE), 0, torch.arange(TILE)] = 1.0
    sel[32] = 1.0
    return sel.reshape(33 * TILE, TILE)


class KdaDecode:
    """Builds and launches the fused KDA decode program. All tensors interleaved (DRAM or L1); K = V = 128,
    conv kernel 4 (3 history rows). Call ``__call__`` with a pre-allocated output."""

    # CB plan (kept in sync with the kernels)
    CB = dict(
        raw=0,
        hist=1,
        taps=2,
        gf=3,
        gog=4,
        gb=5,
        cc=6,
        onw=7,
        sel=8,
        S=9,
        out=10,
        state_out=11,
        rawrm=12,
        qn=13,
        kn=14,
        vc=15,
        dec=16,
        bcol=17,
        gate=18,
        ht=19,
        tmp=20,
        inv=21,
        rows=22,
        cols=23,
        vread=24,
        u=25,
        sd=26,
        s2=27,
        orow=28,
        bs=29,
        selT=30,
        oblk=31,
    )

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
    def _cb(self, idx, dtype, page, pages, cores):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    def __call__(self, qkv, gates, conv_state, recurrent, taps, chan_consts, o_norm_w, selectors, out):
        """qkv [1,1,B,3*H*K] bf16 TILE; gates [1,1,B,H*K+H*V+Hb] fp32 TILE; conv_state [B,3,3*H*K] bf16 ROW_MAJOR
        (updated in place); recurrent [B,H,K,V] fp32 TILE (updated in place); taps [1,1,4,3*H*K] fp32 TILE;
        chan_consts [1,1,2,H*K] fp32 TILE (row 0 dt_bias, row 1 exp(A_log)); o_norm_w [1,1,1,V] fp32 TILE;
        selectors [33*32,32] bf16 TILE; out [1,1,B,H*V] bf16 ROW_MAJOR (pre-allocated). Returns ``out``."""
        H, K, V = self.H, self.K, self.V
        B = recurrent.shape[0]
        Kt, Vt = K // TILE, V // TILE
        Bt = -(-B // TILE)
        HK_t, HV_t = H * Kt, H * Vt
        Wq_t, Wg_t = 3 * HK_t, HK_t + HV_t + (-(-H // TILE))
        kv = Kt * Vt
        device = qkv.device()
        grid = device.compute_with_storage_grid_size()
        n_cores = grid.x * grid.y
        parts = max(1, min(TILE, n_cores // (H * Bt)))
        NI = H * Bt * parts
        NC = min(NI, n_cores)
        core_of = [ttnn.CoreCoord(c // grid.y, c % grid.y) for c in range(NC)]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_of])

        f32, f16 = ttnn.float32, ttnn.bfloat16
        t32, t16 = TILE * TILE * 4, TILE * TILE * 2
        cb = self.CB
        cbs = [
            self._cb(cb["raw"], f16, t16, 3 * Kt, cores),
            self._cb(cb["hist"], f16, t16, 3 * 3 * Kt, cores),
            self._cb(cb["taps"], f32, t32, 3 * Kt, cores),
            self._cb(cb["gf"], f32, t32, Kt, cores),
            self._cb(cb["gog"], f32, t32, Vt, cores),
            self._cb(cb["gb"], f32, t32, 1, cores),
            self._cb(cb["cc"], f32, t32, Kt, cores),
            self._cb(cb["onw"], f32, t32, Vt, cores),
            self._cb(cb["sel"], f16, t16, 33, cores),
            self._cb(cb["S"], f32, t32, kv, cores),
            self._cb(cb["out"], f16, t16, 2 * Vt, cores),
            self._cb(cb["state_out"], f32, t32, 2 * kv, cores),
            self._cb(cb["rawrm"], f16, t16, 3 * Kt, cores),
            self._cb(cb["qn"], f32, t32, Kt, cores),
            self._cb(cb["kn"], f32, t32, Kt, cores),
            self._cb(cb["vc"], f32, t32, Vt, cores),
            self._cb(cb["dec"], f32, t32, Kt, cores),
            self._cb(cb["bcol"], f32, t32, 1, cores),
            self._cb(cb["gate"], f32, t32, Vt, cores),
            self._cb(cb["ht"], f32, t32, 3 * Kt, cores),
            self._cb(cb["tmp"], f32, t32, 2 * Kt, cores),
            self._cb(cb["inv"], f32, t32, 1, cores),
            self._cb(cb["rows"], f32, t32, 4 * Kt, cores),
            self._cb(cb["cols"], f32, t32, 2 * Kt, cores),
            self._cb(cb["vread"], f32, t32, Vt, cores),
            self._cb(cb["u"], f32, t32, Vt, cores),
            self._cb(cb["sd"], f32, t32, kv, cores),
            self._cb(cb["s2"], f32, t32, kv, cores),
            self._cb(cb["orow"], f32, t32, Vt, cores),
            self._cb(cb["bs"], f32, t32, 1, cores),
            self._cb(cb["selT"], f32, t32, 1, cores),
            self._cb(cb["oblk"], f32, t32, 2 * Vt, cores),
        ]

        compute_ct = [
            Kt,
            Vt,
            Bt,
            B,
            parts,
            f32_bits(self.lb),
            f32_bits(self.l2_eps),
            f32_bits(self.norm_eps),
            f32_bits(K**-0.5),
            f32_bits(1.0 / V),
        ]
        reader_ct = [Kt, Vt, Bt, B, H, Wq_t, Wg_t, HK_t, HV_t, parts]
        for t in (qkv, conv_state, taps, gates, chan_consts, o_norm_w, selectors, recurrent):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [Kt, Vt, Bt, B, H, HK_t, HV_t, parts]
        for t in (out, recurrent, conv_state):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()

        conv_page_bytes = conv_state.padded_shape[-1] * 2  # one [b, j] row of the bf16 history (aligned page)
        conv_page_bytes = -(-conv_page_bytes // 32) * 32
        o_page_bytes = -(-(out.padded_shape[-1] * 2) // 32) * 32
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for c, core in enumerate(core_of):
            n_items = (NI - c + NC - 1) // NC
            reader_rt[core.x][core.y] = [
                c,
                n_items,
                NC,
                qkv.buffer_address(),
                conv_state.buffer_address(),
                taps.buffer_address(),
                gates.buffer_address(),
                chan_consts.buffer_address(),
                o_norm_w.buffer_address(),
                selectors.buffer_address(),
                recurrent.buffer_address(),
                conv_page_bytes,
            ]
            writer_rt[core.x][core.y] = [
                c,
                n_items,
                NC,
                out.buffer_address(),
                recurrent.buffer_address(),
                conv_state.buffer_address(),
                conv_page_bytes,
                o_page_bytes,
            ]
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
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op(
            [qkv, gates, conv_state, recurrent, taps, chan_consts, o_norm_w, selectors, out], program
        )
