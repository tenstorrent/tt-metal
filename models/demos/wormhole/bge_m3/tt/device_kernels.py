# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Compute-kernel defaults for BGE-M3.

- **Matmul + SDPA:** **HiFi4 + FP32 dest acc + L1 packer acc** on **Blackhole and Wormhole** so full-model
  PCC stays **> 0.94** at **S8192** (and ~0.99 on BH). Wormhole used to default to HiFi2 for stability,
  but HiFi2 drifts to ~0.85 PCC at long sequence; HiFi4 aligns WH with the BH numerics path.
- **LayerNorm:** HiFi4 + FP32 on all archs (matches ``ttnn`` layer_norm defaults in Metal).
- ``max_seq_len`` on kernel helpers is kept for API compatibility / future tuning; fidelity does not
  downgrade on Wormhole by sequence length.

NOTE: Do not combine smaller SDPA K tiles with disabled FP32 matmul dest acc (previously ~0.2 PCC).
"""

from __future__ import annotations

from ttnn.device import is_blackhole as ttnn_is_blackhole
from ttnn.device import is_wormhole_b0 as ttnn_is_wormhole_b0

import ttnn


def is_wormhole_family_device(mesh_device: ttnn.MeshDevice) -> bool:
    """True when running on Wormhole (not Blackhole).

    Pytest's ``CreateDevice`` fixture sometimes does not match ``is_wormhole_b0(device)`` the same
    way as mesh bring-up; fall back to ``ttnn.get_arch_name()`` so long-seq HiFi4 still applies.
    """
    if ttnn_is_blackhole(mesh_device):
        return False
    if ttnn_is_wormhole_b0(mesh_device):
        return True
    return "wormhole" in ttnn.get_arch_name().lower()


def max_qkv_mm_chunk_seq_len(_mesh_device: ttnn.MeshDevice | None) -> int:
    """Max sequence rows per QKV matmul shard (must divide seq_len when chunking)."""
    return 8192


def max_wo_mm_chunk_seq_len(_mesh_device: ttnn.MeshDevice | None) -> int:
    """Max sequence rows per attention output (WO) matmul shard."""
    return 8192


def bge_m3_matmul_compute_kernel_config(
    mesh_device: ttnn.MeshDevice,
    max_seq_len: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """Return a compute kernel config for matmul operations in BGE-M3.

    **Blackhole** and **Wormhole** use **HiFi4** (required for PCC > 0.94 at S8192 on WH). Other archs
    fall back to HiFi2. ``max_seq_len`` is unused but kept for call-site compatibility.

    Always enables FP32 destination accumulation and L1 packer accumulation.
    """
    _ = max_seq_len
    if ttnn_is_blackhole(mesh_device) or is_wormhole_family_device(mesh_device):
        fidelity = ttnn.MathFidelity.HiFi4
    else:
        fidelity = ttnn.MathFidelity.HiFi2
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def bge_m3_sdpa_compute_kernel_config(
    mesh_device: ttnn.MeshDevice,
    max_seq_len: int | None = None,
) -> ttnn.WormholeComputeKernelConfig:
    """Return the compute kernel config for scaled dot-product attention (SDPA) in BGE-M3.

    **Blackhole** and **Wormhole** use **HiFi4**, matching matmul fidelity for end-to-end PCC.
    ``max_seq_len`` is unused but kept for call-site compatibility.

    FP32 destination accumulation and L1 packer accumulation are always enabled.
    """
    _ = max_seq_len
    if ttnn_is_blackhole(mesh_device) or is_wormhole_family_device(mesh_device):
        sdpa_fidelity = ttnn.MathFidelity.HiFi4
    else:
        sdpa_fidelity = ttnn.MathFidelity.HiFi2
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
