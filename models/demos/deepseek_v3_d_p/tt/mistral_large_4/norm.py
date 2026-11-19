# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Distributed RMSNorm over the TP-split residual (attn_norm, ffn_norm, final_norm).

Adapted from tt_distributed_rms_norm.py (non-fused path):
    rms_norm_pre_all_gather (local sum(x^2), [S/8, 32] per chip) -> all_gather(cluster_axis=1) -> rms_norm_post_all_gather
The input is bf16: an fp32 input leaves nonzero junk in columns 1-31 of the stats tile, which the post op sums into
the variance. The output stays TP-split, bf16 (or out_dtype).
"""

import torch

import ttnn

from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings


def compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("NORM_MATH_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("NORM_FP32_ACC"),
        packer_l1_acc=False,
    )


class TtDistributedRmsNorm:
    # read at run time, so a changed setting applies to an already loaded model
    @property
    def cfg(self):
        return compute_config(self.mesh)

    @property
    def stats_dtype(self):
        return getattr(ttnn, settings.get("NORM_STATS_DTYPE"))

    def __init__(
        self,
        mesh,
        weight: torch.Tensor,
        eps: float,
        topology=ttnn.Topology.Linear,
        num_links: int | None = None,  # None: NUM_LINKS, read at run time
        out_dtype=ttnn.bfloat16,
    ):
        """weight: [H] (any float dtype), split by column over the TP axis; replicated over SP. out_dtype: the
        output's dtype (the input stays bf16)."""
        self.mesh, self.eps, self.topology, self.num_links = mesh, eps, topology, num_links
        self.out_dtype = out_dtype
        h = weight.shape[-1]
        assert h % (32 * mesh.shape[TP_AXIS]) == 0, f"hidden {h} does not split into tiles over TP"
        self.weight = ttnn.from_torch(
            weight.float().reshape(1, 1, h // 32, 32),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 2)),
        )
        # the whole weight on every chip, for NORM_GATHER=input (the layer norms; set by the model)
        self.gather_input = False
        self.weight_full = ttnn.from_torch(
            weight.float().reshape(1, 1, h // 32, 32),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, 1, S/8, H/4] bf16 per chip -> the same shape in out_dtype, normalized over the full H. With
        gather_input and NORM_GATHER=input: x is gathered over TP first and the whole row [1, 1, S/8, H] is normalized
        on every chip (the readers all gather the norm's output anyway; 2-3 ops fewer per norm)."""
        mode = settings.get("NORM_GATHER") if self.gather_input else "stats"
        if mode == "rms_norm":
            # the gathered row through ttnn.rms_norm, which streams a row too wide for L1 (its large-tensor
            # kernels); an fp32 output (attn_norm) normalizes an fp32 copy of the row
            links = self.num_links or settings.get("NUM_LINKS")
            topo = getattr(ttnn.Topology, settings.get("NORM_GATHER_TOPOLOGY"))
            x_full = ttnn.all_gather(x, dim=3, cluster_axis=TP_AXIS, num_links=links, topology=topo)
            if self.out_dtype != x_full.dtype:
                xf = ttnn.typecast(x_full, self.out_dtype)
                ttnn.deallocate(x_full)
                x_full = xf
            out = ttnn.rms_norm(
                x_full,
                epsilon=self.eps,
                weight=self.weight_full,
                compute_kernel_config=self.cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_full)
            return out
        if mode == "input":
            links = self.num_links or settings.get("NUM_LINKS")
            topo = getattr(ttnn.Topology, settings.get("NORM_GATHER_TOPOLOGY"))
            x_full = ttnn.all_gather(x, dim=3, cluster_axis=TP_AXIS, num_links=links, topology=topo)
            # a whole 6144-wide row per core overflows L1 in both ops; the 2D grid splits each row's tiles over the
            # grid's rows
            stats = ttnn.rms_norm_pre_all_gather(
                x_full, dtype=self.stats_dtype, compute_kernel_config=self.cfg, use_2d_core_grid=True
            )
            out = ttnn.rms_norm_post_all_gather(
                x_full,
                stats,
                epsilon=self.eps,
                weight=self.weight_full,
                dtype=self.out_dtype,
                compute_kernel_config=self.cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                use_2d_core_grid=True,
            )
            ttnn.deallocate(stats)
            ttnn.deallocate(x_full)
            return out
        stats = ttnn.rms_norm_pre_all_gather(x, dtype=self.stats_dtype, compute_kernel_config=self.cfg)
        links = self.num_links or settings.get("NUM_LINKS")
        gathered = ttnn.all_gather(stats, dim=3, cluster_axis=TP_AXIS, num_links=links, topology=self.topology)
        ttnn.deallocate(stats)
        out = ttnn.rms_norm_post_all_gather(
            x,
            gathered,
            epsilon=self.eps,
            weight=self.weight,
            dtype=self.out_dtype,
            compute_kernel_config=self.cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gathered)
        return out
