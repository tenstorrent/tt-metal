# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class TtRMSNorm:
    """Gemma-4 RMSNorm on device: x * w * rsqrt(mean(x^2) + eps). Weight is plain w (not 1 + w)."""

    def __init__(self, mesh, weight: torch.Tensor, eps: float = 1e-6):
        self.mesh = mesh
        self.eps = eps
        w = weight.float().reshape(1, 1, 1, -1)
        self.weight = ttnn.from_torch(
            w,
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

    def __call__(self, x):
        return ttnn.rms_norm(
            x,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )


def to_device_replicated(mesh, x_host: torch.Tensor):
    """Host [S, H] -> device [1, 1, S, H] bf16 TILE, replicated across the mesh."""
    t = x_host.float().reshape(1, 1, *x_host.shape[-2:])
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def replicated_to_host(y_device):
    """Device [1, 1, S, H] replicated -> host [S, H] (read from chip 0)."""
    ts = ttnn.get_device_tensors(y_device)
    t = ttnn.to_torch(ts[0])
    return t.reshape(t.shape[-2], t.shape[-1])
