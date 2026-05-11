# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for Kokoro device bring-up."""

from __future__ import annotations

import ttnn


def default_compute_kernel_config(mesh_device, *, math_fidelity=None, fp32_dest_acc_en: bool = True):
    """Conservative high-accuracy matmul/linear settings for Kokoro blocks."""
    if math_fidelity is None:
        math_fidelity = ttnn.MathFidelity.HiFi4
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def dram_tile_config() -> ttnn.MemoryConfig:
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def linear_output_memory_config(
    mesh_device: ttnn.MeshDevice,
    *,
    use_dram: bool = True,
) -> ttnn.MemoryConfig:
    """Default output placement for Kokoro linear layers."""
    if use_dram:
        return dram_tile_config()
    return ttnn.L1_MEMORY_CONFIG
