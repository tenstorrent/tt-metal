# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared device helpers for GLM-5.3-Flash: compute config, replicated upload / readback (harness boundary only)."""

import torch

import ttnn


def hifi4_config(fp32_acc: bool = True):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=False
    )


def replicate(mesh, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """Host tensor -> replicated device tensor (load time or the harness boundary, never inside a forward)."""
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def replicated_to_host(t: ttnn.Tensor) -> torch.Tensor:
    """Replicated device tensor -> host (chip 0's copy; harness boundary only)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
