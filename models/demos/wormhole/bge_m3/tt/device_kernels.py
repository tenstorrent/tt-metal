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
    fidelity = ttnn.MathFidelity.HiFi4 if ttnn_is_blackhole(mesh_device) else ttnn.MathFidelity.HiFi2
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def bge_m3_sdpa_compute_kernel_config(mesh_device: ttnn.MeshDevice) -> ttnn.WormholeComputeKernelConfig:
    sdpa_fidelity = ttnn.MathFidelity.HiFi4 if ttnn_is_blackhole(mesh_device) else ttnn.MathFidelity.HiFi2
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=sdpa_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def bge_m3_layernorm_compute_kernel_config(mesh_device: ttnn.MeshDevice) -> ttnn.WormholeComputeKernelConfig:
    # Match ttnn layer_norm default (layernorm.cpp: HiFi4, approx off, fp32 acc).
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
