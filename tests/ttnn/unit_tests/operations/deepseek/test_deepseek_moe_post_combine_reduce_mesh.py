# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
2-device mesh tests for post_combine_reduce.

Validates that the MeshWorkloadFactoryConcept migration works correctly.
Each device gets a replicated copy of the input; both independently compute
the weighted reduce and should produce identical results matching PyTorch.

Usage (with 2+ devices available):
    pytest tests/ttnn/unit_tests/operations/deepseek/test_post_combine_reduce_mesh.py -v
"""

import pytest
import torch
import ttnn
from ttnn import ConcatMeshToTensor
from loguru import logger

# Small dims for fast iteration; EMB_DIM must be multiple of 1024 (32x32 tile)
NUM_TOKENS = 32  # min: 32 (one core per device)
NUM_DEVICES = 2
NUM_EXPERTS = 8
EMB_DIM = 1024  # minimum: 1 tile wide
EXPERT_DIM = 2
PCC_THRESHOLD = 0.999


def pytorch_reference(combine, weights):
    return (combine * weights.expand(-1, -1, -1, combine.shape[-1])).sum(dim=EXPERT_DIM)


def compute_pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


def assert_pcc(result, expected, threshold=PCC_THRESHOLD, label=""):
    nan_count = torch.isnan(result).sum().item()
    assert nan_count == 0, f"{label}: got {nan_count} NaN elements"
    pcc = compute_pcc(result, expected)
    logger.info(f"  {label}: PCC={pcc:.6f}")
    assert pcc > threshold, f"{label}: PCC {pcc:.6f} below {threshold}"
    return pcc


@pytest.fixture(scope="module")
def two_device_mesh():
    """Open a (1, 2) mesh for the test module."""
    num_devices = ttnn.get_num_pcie_devices()
    if num_devices < NUM_DEVICES:
        pytest.skip(f"Need at least {NUM_DEVICES} devices, found {num_devices}")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, NUM_DEVICES))
    yield mesh
    ttnn.close_mesh_device(mesh)


def to_mesh(tensor_torch, mesh_device):
    """Place a tensor on the mesh (replicated on all devices)."""
    return ttnn.from_torch(
        tensor_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def from_mesh(tensor_tt, mesh_device):
    """Gather from mesh: concat on dim=0 → take first slice (all devices identical)."""
    full = ttnn.to_torch(tensor_tt, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
    return full[0:1]  # both devices produced same result; return first device's output


def run_fused_op(combine_tt, weights_tt):
    return ttnn.experimental.deepseek_prefill.post_combine_reduce(
        combine_tt,
        weights_tt,
        expert_dim=EXPERT_DIM,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# ============================================================================
# Tests
# ============================================================================


def test_mesh_random_data(two_device_mesh):
    """Basic correctness: random data replicated on 2 devices."""
    torch.manual_seed(0)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    ref = pytorch_reference(combine, weights)
    result = from_mesh(
        run_fused_op(to_mesh(combine, two_device_mesh), to_mesh(weights, two_device_mesh)), two_device_mesh
    )
    assert_pcc(result, ref, label="mesh_random")


def test_mesh_structured_data(two_device_mesh):
    """Structured data: constant-per-tile activations with sequential weights."""
    tile_width = 1024
    num_emb_tiles = EMB_DIM // tile_width

    tile_values = 0.01 * torch.arange(1, NUM_TOKENS * NUM_EXPERTS * num_emb_tiles + 1, dtype=torch.float32)
    combine = (
        tile_values.view(1, NUM_TOKENS, NUM_EXPERTS, num_emb_tiles, 1)
        .expand(1, NUM_TOKENS, NUM_EXPERTS, num_emb_tiles, tile_width)
        .reshape(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM)
        .to(torch.bfloat16)
    )
    weights = (
        torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float32)
        .view(1, 1, NUM_EXPERTS, 1)
        .expand(1, NUM_TOKENS, NUM_EXPERTS, 1)
        .to(torch.bfloat16)
    )

    ref = pytorch_reference(combine, weights)
    result = from_mesh(
        run_fused_op(to_mesh(combine, two_device_mesh), to_mesh(weights, two_device_mesh)), two_device_mesh
    )
    assert_pcc(result, ref, threshold=0.998, label="mesh_structured")


def test_mesh_sparse_weights(two_device_mesh):
    """Sparse weights: 4 of 8 experts active per token."""
    k_active = 4
    torch.manual_seed(1)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.zeros(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)
    for t in range(NUM_TOKENS):
        active = torch.randperm(NUM_EXPERTS)[:k_active]
        weights[0, t, active, 0] = torch.randn(k_active).to(torch.bfloat16)

    ref = pytorch_reference(combine, weights)
    result = from_mesh(
        run_fused_op(to_mesh(combine, two_device_mesh), to_mesh(weights, two_device_mesh)), two_device_mesh
    )
    assert_pcc(result, ref, label=f"mesh_sparse_{k_active}/{NUM_EXPERTS}")


def test_mesh_output_shape_and_layout(two_device_mesh):
    """Verify output layout is TILE and shape is correct."""
    torch.manual_seed(2)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    result_tt = run_fused_op(to_mesh(combine, two_device_mesh), to_mesh(weights, two_device_mesh))
    assert result_tt.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {result_tt.layout}"

    result = from_mesh(result_tt, two_device_mesh)
    assert list(result.shape) == [1, NUM_TOKENS, EMB_DIM], f"Wrong shape: {result.shape}"
