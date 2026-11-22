# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 q_a stem on the 2x2 mesh: q_resid = q_a_layernorm(q_a_proj(attn_norm)), HF HYV4RMSNorm (plain w, eps 1e-6 =
hy4_ref.LATENT_NORM_EPS, not rms_norm_eps 1e-5).

Adapted from models/demos/deepseek_v3_d_p/tt/mla/mla.py:ttMLA._q_a_latent (K-split linear -> TP reduce -> rms_norm).
Per chip (r, c) the input is attn_norm's column split [1, 1, S/2, 3072] (tt/norm.py output, bf16):

    ttnn.linear                x_c @ W_c^T, W_c = q_a_proj[:, c*3072:(c+1)*3072]^T [3072, 2048] bf16, HiFi4 + fp32
                               dest, fp32 partial out (no bf16 rounding before the reduce)
    ttnn.all_reduce axis 1     sum the two K partials -> [1, 1, S/2, 2048] fp32, identical on both chips of a row
    ttnn.bringup.rms_norm      q_a_layernorm, eps 1e-6, fp32 in, fp32 weight, HiFi4 + fp32 dest -> fp32 -> bf16
                               (typecast)

Norm op: ``norm_impl="bringup"`` (default) runs the rms_norm_ttnn fork; ``"native"`` runs ttnn.rms_norm. On the
layer-0 golden the native op scales every row ~0.1% low (row norm ratio mean 0.99898 vs the fp32 CPU, rel 0.00124 at
fp32 out); the fork, whose fp32 sum-of-squares fix targets that bias, gives 0.99974 / 0.00062. The linear +
all_reduce alone is rel 0.00037.

Differences from ttMLA: the Hy4 eps (ttMLA passes config.rms_norm_eps), HiFi4 + fp32 dest (owner rule), fp32 partials,
and ttnn.all_reduce(cluster_axis=1) instead of reduce_scatter_minimal_async + high_bw_all_gather (no persistent
buffers or semaphores to own; the same pair on a 2-chip axis). Output q_resid [1, 1, S/2, 2048] bf16 per chip,
replicated over the 2 columns of a row (plan.md). No host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


class TtQa:
    """q_a_proj + q_a_layernorm of one layer. Call with the per-chip attn_norm [1, 1, S/2, H/2] TILE (bf16 or fp32);
    returns q_resid [1, 1, S/2, q_lora_rank] TILE ``dtype``, replicated over ``cluster_axis``."""

    def __init__(
        self,
        mesh,
        q_a_proj: torch.Tensor,  # [q_lora_rank, H] (HF nn.Linear layout)
        q_a_norm: torch.Tensor,  # [q_lora_rank]
        eps: float,
        cluster_axis: int = 1,
        dtype=ttnn.bfloat16,
        norm_impl: str = "bringup",
    ):
        assert norm_impl in ("bringup", "native"), norm_impl
        self.norm_op = ttnn.bringup.rms_norm if norm_impl == "bringup" else ttnn.rms_norm
        rank, hidden = q_a_proj.shape
        tp = mesh.shape[cluster_axis]
        assert hidden % (TILE * tp) == 0 and rank % TILE == 0, (rank, hidden, tp)
        assert q_a_norm.numel() == rank
        self.mesh, self.rank, self.eps, self.cluster_axis, self.dtype = mesh, rank, float(eps), cluster_axis, dtype
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # W^T [H, rank], K (dim 2) split over the cluster axis: chip column c holds rows c*H/tp .., the K slice that
        # matches its attn_norm columns; replicated over the other axis.
        dims = (None, 2) if cluster_axis == 1 else (2, None)
        self.weight = ttnn.from_torch(
            q_a_proj.t().contiguous().reshape(1, 1, hidden, rank),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        self.norm_w = ttnn.from_torch(
            q_a_norm.float().reshape(1, 1, rank // TILE, TILE),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        part = ttnn.linear(x, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=dram)
        qr = ttnn.all_reduce(part, cluster_axis=self.cluster_axis, memory_config=dram)
        ttnn.deallocate(part)
        y = self.norm_op(
            qr,
            weight=self.norm_w,
            epsilon=self.eps,
            memory_config=dram,
            compute_kernel_config=self.ckc,
        )
        ttnn.deallocate(qr)
        if y.dtype != self.dtype:
            y2 = ttnn.typecast(y, self.dtype, memory_config=dram)
            ttnn.deallocate(y)
            y = y2
        return y
