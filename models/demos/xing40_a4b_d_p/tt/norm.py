# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 distributed RMSNorm on the 4x2 mesh (attn_norm = input_layernorm: w * x * rsqrt(mean(x^2) + eps),
plain w, eps 1e-6).

Per chip (r, c) the input is the chip's rows and hidden columns [1, 1, S/4, 1792] (tt/collapse.py:TtHcCollapse
output, fp32). Composition (hy4_preview_d_p/tt/norm.py:TtDistributedRmsNorm, from deepseek_v3_d_p/tt/
tt_distributed_rms_norm.py):

    ttnn.rms_norm_pre_all_gather   local sum(x^2) -> [S/4, 32] fp32 stats (one tile column)
    ttnn.multiply by a one-hot     keep column 0 only (fp32 input leaves junk in columns 1-31, known issues)
    ttnn.all_gather over axis 1    -> [S/4, 64] (both column halves; Linear, FABRIC_2D mesh)
    ttnn.rms_norm_post_all_gather  x * rsqrt(sum / H + eps) * w_c  (w split by column, fp32 row-major)

ffn_norm (post_attention_layernorm) adds ttnn.all_gather(dim 3, cluster_axis 1) -> [1, 1, S/4, H] bf16 per chip,
replicated within a row (the input of every FFN consumer).

HiFi4 + fp32 dest on both halves. attn_norm output stays column-split [1, 1, S/4, 1792] in ``dtype`` (bf16 default): q_a_proj
and kv_a_proj_with_mqa take a K-split input. No host work in __call__; weight and mask built at load.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32


class TtDistributedRmsNorm:
    def __init__(
        self, mesh, weight: torch.Tensor, eps: float, cluster_axis: int = 1, dtype=ttnn.bfloat16, gather: bool = False
    ):
        self.gather = gather
        hidden = weight.numel()
        tp = mesh.shape[cluster_axis]
        assert hidden % (TILE * tp) == 0, (hidden, tp)
        self.mesh, self.hidden, self.eps, self.cluster_axis, self.dtype = mesh, hidden, float(eps), cluster_axis, dtype
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # w as [1, 1, H/32, 32] row-major fp32, dim 2 split over the cluster axis (chip column c holds
        # w[c H/tp : (c+1) H/tp]), replicated over the other axis.
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
        stats = ttnn.multiply(raw, self.stats_mask, dtype=ttnn.float32, memory_config=dram)
        ttnn.deallocate(raw)
        gathered = ttnn.all_gather(
            stats, dim=3, cluster_axis=self.cluster_axis, topology=ttnn.Topology.Linear, memory_config=dram
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
        if not self.gather:
            return y
        # ffn_norm: every FFN consumer (router, dispatch, dense MLP, shared expert) needs the full hidden, so gather
        # the normed column halves once here -> [1, 1, S/4, H] per chip, replicated within a mesh row.
        full = ttnn.all_gather(
            y, dim=3, cluster_axis=self.cluster_axis, topology=ttnn.Topology.Linear, memory_config=dram
        )
        ttnn.deallocate(y)
        return full


# norm steps -> checkpoint weight under model.layers.<i>.
NORM_WEIGHTS = {"attn_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight"}
# norm steps whose output is gathered over the mesh columns (full hidden, replicated within a row).
GATHERED_NORMS = {"ffn_norm"}


def build_norm(mesh, loader, cfg, layer: int, step: str, dtype=ttnn.bfloat16) -> TtDistributedRmsNorm:
    w = loader.get(f"model.layers.{layer}.{NORM_WEIGHTS[step]}").float()
    return TtDistributedRmsNorm(mesh, w, cfg.rms_norm_eps, cluster_axis=1, dtype=dtype, gather=step in GATHERED_NORMS)
