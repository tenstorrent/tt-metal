# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that the wo output projection weight is correctly sharded for MHA
models (n_heads == n_kv_heads) on multi-device meshes.

Regression test for: ShardTensor2dMesh producing incorrect weight shapes
when n_heads == n_kv_heads on N300 (1x2 mesh).
"""

import os

import pytest
import torch

import ttnn


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mha_wo_sharding(mesh_device):
    """On a multi-device mesh, wo sharded via ShardTensorToMesh(dim=2) must
    produce per-device shape [1, 1, n_heads*head_dim // num_devices, dim]
    and the subsequent ttnn.linear must not crash."""

    num_devices = mesh_device.get_num_devices()
    if num_devices < 2:
        pytest.skip("MHA sharding test requires >= 2 devices")

    n_heads = 16
    head_dim = 128
    dim = 2048
    seq_len = 128
    n_local_heads = n_heads // num_devices
    qkv_inner = n_local_heads * head_dim

    pt_wo = torch.randn(1, 1, n_heads * head_dim, dim)
    wo = ttnn.as_tensor(
        pt_wo,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    expected_sharded_dim = (n_heads * head_dim) // num_devices
    assert wo.shape[-2] == expected_sharded_dim, f"wo dim 2 should be {expected_sharded_dim}, got {wo.shape[-2]}"

    attn_output = ttnn.as_tensor(
        torch.randn(1, 1, seq_len, qkv_inner),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    output = ttnn.linear(
        attn_output,
        wo,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert output.shape[-1] == dim, f"output width should be {dim}, got {output.shape[-1]}"
    assert output.shape[-2] == seq_len, f"output seq_len should be {seq_len}, got {output.shape[-2]}"

    ttnn.deallocate(wo)
    ttnn.deallocate(attn_output)
    ttnn.deallocate(output)
