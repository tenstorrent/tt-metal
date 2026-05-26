# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for fused_gated_rms_norm_step (partial silu+mul fusion).

Compares fused (ttnn.rms_norm + fused silu+mul kernel) against
torch reference, on a [h_per_dev, head_v_dim] 2D shape that matches
what the trace path passes after Option A (head-sharded gated_rms_norm).

Run with:
  export TTNN_GDN_KERNEL=1
  export MESH_DEVICE=QB2
  pytest models/experimental/tt_symbiote/tests/test_gated_rms_norm_kernel_unit.py -s
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn


MESH_DEVICE_MAP = {
    "T3K": (1, 8),
    "QB2": (1, 4),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def _replicated(t, mesh_device):
    return ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
    )


def _to_torch_replicated(t_ttnn, mesh_device):
    return ttnn.to_torch(t_ttnn, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]


@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_fused_gated_rms_norm_matches_torch(mesh_device):
    """fused (rms_norm + silu+mul kernel) vs torch reference."""
    from models.experimental.tt_symbiote.modules.gated_rms_norm_kernel import (
        fused_gated_rms_norm_step,
    )

    torch.manual_seed(0)
    H = 32  # rows (will represent heads in the kernel call)
    D = 128  # head_v_dim
    eps = 1e-6

    # Use a leading batch=1 dim so _to_torch_replicated's [0:1] slice
    # extracts one device's worth of data cleanly.
    x = torch.randn(1, H, D, dtype=torch.bfloat16)
    gate = torch.randn(1, H, D, dtype=torch.bfloat16)
    weight = torch.randn(D, dtype=torch.bfloat16) * 0.1 + 1.0

    # Torch reference.
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    h = weight.float() * h
    ref = (h * F.silu(gate.float())).to(torch.bfloat16)

    # Fused TTNN+tt-lang
    x_tt = _replicated(x, mesh_device)
    gate_tt = _replicated(gate, mesh_device)
    weight_tt = _replicated(weight.unsqueeze(0), mesh_device)
    y_out = _replicated(torch.zeros(1, H, D, dtype=torch.bfloat16), mesh_device)

    fused_gated_rms_norm_step(x_tt, gate_tt, weight_tt, eps, y_out)

    y = _to_torch_replicated(y_out, mesh_device).reshape(1, H, D)
    pcc = _pcc(y, ref)
    mae = (y.float() - ref.float()).abs().max().item()
    print(f"\nFUSED_GATED_RMS_NORM  pcc={pcc:.6f}  max_abs_err={mae:.4e}")
    print(f"ref[0,0,:6]   = {ref[0,0,:6].tolist()}")
    print(f"fused[0,0,:6] = {y[0,0,:6].tolist()}")
    print(f"ref[0,15,:6]   = {ref[0,15,:6].tolist()}")
    print(f"fused[0,15,:6] = {y[0,15,:6].tolist()}")

    assert pcc > 0.99, f"pcc too low: {pcc}"
