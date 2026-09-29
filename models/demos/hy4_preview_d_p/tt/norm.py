# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 distributed RMSNorm on the 2x2 mesh (input_layernorm = attn_norm, HF HYV4RMSNorm: w * x * rsqrt(mean(x^2) +
eps), plain w).

Per chip (r, c) the input is this chip's hidden columns [1, 1, S/2, H/2] (tt/ihc.py:TtHcPre output, fp32), split by
rows over axis 0 and by columns over axis 1. Composition (deepseek_v3_d_p/tt/tt_distributed_rms_norm.py:
TtDistributedRmsNorm):

    ttnn.rms_norm_pre_all_gather   local sum(x^2) -> [S/2, 32] fp32 stats (one tile column)
    ttnn.all_gather over axis 1    -> [S/2, 64] (both column halves)
    ttnn.rms_norm_post_all_gather  x * rsqrt(sum / H + eps) * w_c  (w split by column, fp32 row-major)

rms_norm_pre_all_gather on an fp32 input leaves nonzero junk in columns 1-31 of its stats tile (|v| up to ~17 on the
Hy4 golden; column 0 is the exact sum; a bf16 input gives zeros there), and rms_norm_post_all_gather row-reduces the
whole tile, so the output came out 1-2% low (row norm ratio [0.979, 0.989]). The stats are therefore multiplied by a
[1, 32] one-hot row (column 0, built at load) before the gather.

Differences from the DeepSeek module: the stats mask, fp32 stats (not bf16), HiFi4 + fp32 dest acc on both halves
(owner rule; fp32 input needs fp32 dest anyway), the model's eps, plain ttnn.all_gather over the 2-chip axis (FABRIC_2D mesh, Linear).
Output stays column-split [1, 1, S/2, H/2] (``dtype``, bf16 by default), as the K-split projections after it want.
No host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


class TtDistributedRmsNorm:
    """One column-split RMSNorm. Call with the per-chip input [1, 1, S/2, H/2] TILE (fp32 or bf16); returns the
    per-chip normalized columns [1, 1, S/2, H/2] TILE in ``dtype``."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float, cluster_axis: int = 1, dtype=ttnn.bfloat16):
        hidden = weight.numel()
        tp = mesh.shape[cluster_axis]
        assert hidden % (TILE * tp) == 0, (hidden, tp)
        self.mesh = mesh
        self.hidden = hidden
        self.eps = float(eps)
        self.cluster_axis = cluster_axis
        self.dtype = dtype
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # w as [1, 1, H/32, 32] row-major fp32; dim 2 split over the cluster axis: chip column c holds
        # w[c*H/tp : (c+1)*H/tp] (rows c*H/(32 tp) .. of the reshaped weight), replicated over the other axis.
        dims = (None, 2) if cluster_axis == 1 else (2, None)
        self.weight = ttnn.from_torch(
            weight.float().reshape(1, 1, hidden // TILE, TILE),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        col0 = torch.zeros(1, 1, 1, TILE)
        col0[..., 0] = 1.0
        self.stats_mask = ttnn.from_torch(
            col0,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        raw = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        stats = ttnn.multiply(raw, self.stats_mask, dtype=ttnn.float32, memory_config=dram)  # keep column 0 only
        ttnn.deallocate(raw)
        gathered = ttnn.all_gather(
            stats,
            dim=3,
            cluster_axis=self.cluster_axis,
            topology=ttnn.Topology.Linear,
            memory_config=dram,
        )
        ttnn.deallocate(stats)
        y = ttnn.rms_norm_post_all_gather(
            x,
            gathered,
            epsilon=self.eps,
            weight=self.weight,
            memory_config=dram,
            compute_kernel_config=self.ckc,
            dtype=self.dtype,
        )
        ttnn.deallocate(gathered)
        return y
