# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute-kernel defaults for BGE-M3.

- Always use ``init_device_compute_kernel_config(mesh_device.arch(), ...)`` so the stack sees the
  correct architecture.
- **Wormhole matmuls** stay **HiFi2 + FP32 dest acc** (avoids known WH matmul quirks with HiFi4+FP32).
- **Blackhole matmuls** use **HiFi4 + FP32 dest acc** to tighten long-sequence PCC (S8192) where SDPA
  itself must stay BF16 (prim SDPA only allows BF16/BF8/BF4 inputs).
- **LayerNorm** uses **HiFi4 + FP32** on all archs, matching ``ttnn`` layer_norm defaults in Metal.
- **SDPA**: HiFi4 on Blackhole, HiFi2 on Wormhole.

NOTE: Blackhole-only smaller SDPA K tiles / disabling FP32 on matmuls previously produced broken
outputs (~0.2 PCC); do not reintroduce that combination.
"""

from __future__ import annotations

from ttnn.device import is_blackhole as ttnn_is_blackhole

import ttnn


def max_qkv_mm_chunk_seq_len(_mesh_device: ttnn.MeshDevice | None) -> int:
    """Max sequence rows per QKV matmul shard (must divide seq_len when chunking)."""
    return 8192


def max_wo_mm_chunk_seq_len(_mesh_device: ttnn.MeshDevice | None) -> int:
    """Max sequence rows per attention output (WO) matmul shard."""
    return 8192


def bge_m3_matmul_compute_kernel_config(mesh_device: ttnn.MeshDevice) -> ttnn.WormholeComputeKernelConfig:
    """Return a compute kernel config for matmul operations in BGE-M3.

    Selects math fidelity from device architecture:

    - **Blackhole:** HiFi4 (better accuracy for long sequences).
    - **Wormhole:** HiFi2 (avoids known matmul quirks with HiFi4 + FP32 on Wormhole).

    Always enables FP32 destination accumulation and L1 packer accumulation.
    """
    fidelity = ttnn.MathFidelity.HiFi4 if ttnn_is_blackhole(mesh_device) else ttnn.MathFidelity.HiFi2
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def bge_m3_sdpa_compute_kernel_config(mesh_device: ttnn.MeshDevice) -> ttnn.WormholeComputeKernelConfig:
    """Return the compute kernel config for scaled dot-product attention (SDPA) in BGE-M3.

    Math fidelity is architecture-specific:

    - **Blackhole:** HiFi4 for tighter precision.
    - **Wormhole:** HiFi2 for compatibility and stability.

    FP32 destination accumulation and L1 packer accumulation are always enabled.
    """
    sdpa_fidelity = ttnn.MathFidelity.HiFi4 if ttnn_is_blackhole(mesh_device) else ttnn.MathFidelity.HiFi2
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=sdpa_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def bge_m3_layernorm_compute_kernel_config(mesh_device: ttnn.MeshDevice) -> ttnn.WormholeComputeKernelConfig:
    """Return the compute kernel config for LayerNorm in BGE-M3 on the given mesh device.

    Matches ``ttnn`` layer_norm defaults in Metal (``layernorm.cpp``):

    - HiFi4 math fidelity
    - Approximate math off
    - FP32 destination accumulator enabled
    - L1 packer accumulation enabled
    """
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
