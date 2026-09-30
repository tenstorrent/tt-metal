# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 RMSNorm under CP=4 on the 1x4 mesh: y = w * (x * (mean(x^2) + eps)^-0.5) (plain w, not 1 + w).

Copied from the 1x4 prior models/demos/mimo_v2_6_d_p/tt/rms_norm.py. The module is unchanged: rows are independent, so
each chip normalizes its own contiguous S/4 slice of the residual stream (no CCL). The weight is replicated, the kernel
(ttnn.bringup.rms_norm) runs HiFi4 with fp32 accumulation. Only the harness helpers differ: the input is sequence-sharded
(chip c holds rows [c*S/4, (c+1)*S/4)) instead of replicated.
"""

from __future__ import annotations

import os

import torch

import ttnn

SEQ_DIM = 2  # [1, 1, S, H]: the CP axis


class TtRMSNorm:
    def __init__(self, mesh, weight: torch.Tensor | None, eps: float = 1e-6, dtype=ttnn.bfloat16):
        self.mesh = mesh
        self.eps = eps
        self.dtype = dtype
        self.weight = None
        if weight is not None:
            self.weight = ttnn.from_torch(
                weight.float().reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: chip c's CP slice [1, 1, S/4, H] TILE. Returns the same shape and sharding.

        MIMO_NORM_IMPL=bringup (default) runs ttnn.bringup.rms_norm; MIMO_NORM_IMPL=native runs ttnn.rms_norm."""
        op = ttnn.rms_norm if os.environ.get("MIMO_NORM_IMPL", "bringup") == "native" else ttnn.bringup.rms_norm
        return op(
            x,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def fused_add(self, a: ttnn.Tensor, b: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """(norm(a + b), a + b) in one call on the local CP slices (ttnn.bringup.rms_norm return_residual_sum)."""
        return ttnn.bringup.rms_norm(
            b,
            residual_input_tensor=a,
            return_residual_sum=True,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            residual_sum_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def to_device_cp(mesh, h: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Host [S, H] -> device [1, 1, S/4, H] per chip, chip c holding contiguous slice c (harness boundary only)."""
    n = mesh.get_num_devices()
    s = h.shape[-2]
    assert s % (n * ttnn.TILE_SIZE) == 0, f"S={s} must split into {n} tile-aligned CP slices"
    return ttnn.from_torch(
        h.reshape(1, 1, *h.shape[-2:]).to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=SEQ_DIM),
    )


def cp_to_host(mesh, t: torch.Tensor) -> torch.Tensor:
    """Device CP slices [1, 1, S/4, H] -> host [S, H], concatenated in chip order (harness boundary only)."""
    y = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=SEQ_DIM))
    return y.reshape(-1, y.shape[-1])
