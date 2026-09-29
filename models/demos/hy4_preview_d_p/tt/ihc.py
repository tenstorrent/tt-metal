# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 iHC gates on the 2x2 mesh (hc_attn_layer / hc_mlp_layer, HF HYV4HyperConnection), all fp32:

    mixes = (flat [S, 4H] @ fn[8, 4H]^T) * rsqrt(mean(flat^2) + 1e-5)
    pre   = sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    post  = 2 * sigmoid(mixes[:, 4:] * scale[1] + base[4:]) + 1e-6
    gates = cat(pre, post) [S, 8]

Per chip (r, c) the streams are [1, 1, S/2, 4 x 3072] (tt/layout.py). The RMS has no weight, so its rsqrt commutes
with the linear and is applied after it (deepseek_v3_d_p/tt/mhc/tt_mhc.py:_project). Each chip computes its partial
mixes (its 12288 columns of fn, permuted on the host to the chip's stream-column order) and its partial sum of
squares; both go into one [S/2, 32] fp32 tile row (mixes in columns 0-7, sum of squares in column 8), and one
ttnn.all_reduce over axis 1 completes them. The rest is elementwise on [S/2, 32] ([1, 32] row constants built at
load), and the output is sliced to [1, 1, S/2, 8] on the device, identical on both chips of a row.

Every matmul and reduction runs HiFi4 + fp32 accumulation (owner rule); the fp32 elementwise ops run on the SFPU.
"""

from __future__ import annotations

import torch

import ttnn

from .layout import HC, streams_cols_to_chip_major

W = 32  # one tile row: 8 gate columns, the sum of squares in column SS_COL, the rest zero
SS_COL = 2 * HC


def _ckc(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _replicated_row(mesh, row: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        row.float().reshape(1, 1, 1, W),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class TtHcGates:
    """One iHC gate block (hc_attn_layer or hc_mlp_layer of one layer). Call with the per-chip streams
    [1, 1, S/2, 4 x H/2] fp32 TILE; returns gates [1, 1, S/2, 8] fp32 TILE (pre 4 | post 4), replicated over axis 1."""

    def __init__(
        self,
        mesh,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        hidden: int,
        norm_eps: float = 1e-5,
        hc_eps: float = 1e-6,
        magnitude: float = 2.0,
    ):
        n2 = 2 * HC
        assert fn.shape == (n2, HC * hidden), fn.shape
        self.mesh = mesh
        self.hidden = hidden
        self.inv_n = 1.0 / float(HC * hidden)  # mean over all 4H values of the row
        self.norm_eps = float(norm_eps)
        self.ckc = _ckc(mesh)
        # fn^T in the chip-major (c, j, k) row order, padded to 32 output columns; the rows are split over mesh
        # columns (axis 1) and replicated over rows (axis 0): chip (r, c) holds its [4 x H/2, 32] block.
        fn_t = torch.zeros(HC * hidden, W, dtype=torch.float32)
        fn_t[:, :n2] = streams_cols_to_chip_major(fn.float(), hidden).T
        self.fn_t = ttnn.from_torch(
            fn_t.reshape(1, 1, HC * hidden, W),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 2)),
        )
        one_hot = torch.zeros(W)
        one_hot[SS_COL] = 1.0
        a = torch.zeros(W)
        a[:HC], a[HC:n2] = float(scale[0]), float(scale[1])
        b = torch.zeros(W)
        b[:n2] = base.float()
        mag = torch.zeros(W)
        mag[:HC], mag[HC:n2] = 1.0, float(magnitude)
        eps = torch.zeros(W)
        eps[:n2] = float(hc_eps)
        self.ss_onehot = _replicated_row(mesh, one_hot)
        self.scale_row = _replicated_row(mesh, a)
        self.base_row = _replicated_row(mesh, b)
        self.mag_row = _replicated_row(mesh, mag)
        self.eps_row = _replicated_row(mesh, eps)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        s2 = x.shape[-2]
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Partial mixes [S/2, 32] (columns 8-31 zero) and partial sum of squares [S/2, 1] over this chip's columns.
        mix = ttnn.matmul(x, self.fn_t, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        sq = ttnn.multiply(x, x, dtype=ttnn.float32, memory_config=dram)
        ss = ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(sq)
        # Pack: column SS_COL <- sum of squares ([S/2, 1] x [1, 32] one-hot broadcast), then one all_reduce.
        packed = ttnn.add(mix, ttnn.multiply(ss, self.ss_onehot, dtype=ttnn.float32), dtype=ttnn.float32)
        ttnn.deallocate(mix)
        ttnn.deallocate(ss)
        red = ttnn.all_reduce(packed, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(packed)
        # rsqrt(sum / 4H + eps), broadcast down the row.
        ssum = ttnn.slice(red, [0, 0, 0, SS_COL], [1, 1, s2, SS_COL + 1])
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ssum, self.inv_n), self.norm_eps))
        ttnn.deallocate(ssum)
        y = ttnn.multiply(red, inv, dtype=ttnn.float32)
        ttnn.deallocate(red)
        ttnn.deallocate(inv)
        y = ttnn.add(ttnn.multiply(y, self.scale_row), self.base_row)
        g = ttnn.add(ttnn.multiply(ttnn.sigmoid(y), self.mag_row), self.eps_row)
        ttnn.deallocate(y)
        out = ttnn.slice(g, [0, 0, 0, 0], [1, 1, s2, 2 * HC], memory_config=dram)
        ttnn.deallocate(g)
        return out


class TtHcPre:
    """iHC pre-mix (HF HYV4HyperConnection, sublayer input): x = sum_j pre_j * stream_j, fp32.

    Call with the per-chip streams [1, 1, S/2, 4 x H/2] fp32 TILE (tt/layout.py) and the gates [1, 1, S/2, 8] fp32
    TILE (TtHcGates, replicated over axis 1); returns this chip's columns of the sublayer input [1, 1, S/2, H/2]
    (``dtype``, fp32 by default), split by rows over axis 0 and by hidden columns over axis 1. No collective: each
    chip mixes its own column block of every stream (deepseek_v3_d_p/tt/mhc/tt_mhc.py:_streams / _cols / _mix)."""

    def __init__(self, mesh, hidden: int, dtype=ttnn.float32):
        self.mesh = mesh
        self.hidden = hidden
        self.dtype = dtype

    def __call__(self, x: ttnn.Tensor, gates: ttnn.Tensor) -> ttnn.Tensor:
        s2, w = x.shape[-2], x.shape[-1] // HC
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Stream j is local columns [j*w, (j+1)*w) (stream-major packing); pre_j is gate column j ([S/2, 1]).
        st = [ttnn.slice(x, [0, 0, 0, j * w], [1, 1, s2, (j + 1) * w], memory_config=dram) for j in range(HC)]
        pre = [ttnn.slice(gates, [0, 0, 0, j], [1, 1, s2, j + 1], memory_config=dram) for j in range(HC)]
        y = ttnn.multiply(st[0], pre[0], dtype=ttnn.float32, memory_config=dram)
        for s, p in zip(st[1:], pre[1:]):
            y2 = ttnn.addcmul(y, s, p, memory_config=dram)
            ttnn.deallocate(y)
            y = y2
        for t in st + pre:
            ttnn.deallocate(t)
        if self.dtype != ttnn.float32:
            y2 = ttnn.typecast(y, self.dtype)
            ttnn.deallocate(y)
            y = y2
        return y
