# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit test for TTNNLinearMeshShard.forward.

The smoke test in test_pipeline_smoke.py validates that the sharded Linear
PLACES on a 32-chip BH Galaxy. This unit test validates that a single
sharded Linear's forward (matmul + all-gather) produces the same output
as the torch reference.

Approach: wrap a small (1024 → 2048) nn.Linear, place across the mesh,
run forward on synthetic input, compare against torch reference via PCC.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.utils.test import line_params


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation (single scalar) between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(4, 8), line_params],  # full 32-chip BH Galaxy
    ],
    indirect=True,
    ids=["bh_4x8"],
)
@pytest.mark.timeout(600)
def test_linear_mesh_shard_forward(mesh_device):
    """One sharded Linear, one forward, compare with torch."""
    from models.tt_dit.experimental.cosmos3_i2v.tt_modules.linear_mesh_shard import TTNNLinearMeshShard

    torch.manual_seed(0)
    in_features = 1024
    out_features = 2048  # divisible by 8 (mesh axis 1)
    batch = 1
    seq = 64

    linear = torch.nn.Linear(in_features, out_features, bias=True).to(torch.bfloat16)
    x = torch.randn(batch, seq, in_features, dtype=torch.bfloat16)
    y_ref = linear(x)

    tt_linear = TTNNLinearMeshShard.from_torch(linear)
    tt_linear.to_device(mesh_device)
    tt_linear.preprocess_weights()
    tt_linear.move_weights_to_device()

    # Replicate input across all 32 chips
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    y_tt = tt_linear.forward(x_tt)

    # Forward rewrites the output's TensorTopology to reflect
    # PlacementShard(-1) on axis 1, so auto_compose now sees the right
    # placement and correctly concatenates the 8 chip shards.
    from models.common.auto_compose import to_torch_auto_compose

    y_torch = to_torch_auto_compose(y_tt, device=mesh_device).to(torch.bfloat16)

    pcc = _pcc(y_torch, y_ref)
    print(f"y_ref shape: {tuple(y_ref.shape)}, y_torch shape: {tuple(y_torch.shape)}")
    print(f"PCC: {pcc:.6f}")
    print(f"max abs error: {(y_torch - y_ref).abs().max().item():.4f}")

    assert y_torch.shape == y_ref.shape, f"shape mismatch: {y_torch.shape} vs {y_ref.shape}"
    assert pcc > 0.99, f"PCC too low: {pcc:.4f}"
