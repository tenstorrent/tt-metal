"""
PCC test for distributed RMSNorm.

Compares distributed RMSNorm (rms_norm_pre_all_gather + all_gather + rms_norm_post_all_gather)
against PyTorch reference implementation.

Tests that hidden dimension sharding across chips produces correct results when
using the distributed RMSNorm approach with all-gather for global statistics.

Note: Replicated RMSNorm (full 7168 hidden_dim on single chip) exceeds L1 memory,
which is exactly why distributed RMSNorm with hidden dimension sharding is used.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Constants matching DeepSeek 671B dimensions
# Input shape: (1, 1, batch_rows, hidden_dim) - 4D format required for distributed RMSNorm
BATCH_ROWS = 32  # Tile-aligned batch dimension
HIDDEN_DIM = 7168
EPSILON = 1e-6


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference RMSNorm implementation."""
    x_float = x.float()
    y = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="linear"),
            id="linear-4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_rmsnorm_distributed(mesh_device, device_params):
    """
    Test distributed RMSNorm against PyTorch reference.

    Input tensor is sharded across 4 chips along hidden dimension.
    Each chip computes local statistics, then all-gather combines them
    for global normalization.

    Configuration:
    - Input shape: (1, 1, 32, 7168) - 4D format
    - Per device: (1, 1, 32, 1792) - 7168 / 4 = 1792
    - Stats gathered along cluster_axis=1 (mesh columns)
    """
    torch.manual_seed(1234)

    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    per_device_width = HIDDEN_DIM // num_devices

    # 4D shapes for distributed RMSNorm
    inp_shape_full = (1, 1, BATCH_ROWS, HIDDEN_DIM)
    inp_shape_per_device = (1, 1, BATCH_ROWS, per_device_width)

    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.info(f"Full input: {inp_shape_full}, per-device: {inp_shape_per_device}")

    signpost(f"RMSNorm PCC test - mesh {mesh_shape}")

    # Create random input and weight tensors
    torch_input = torch.randn(inp_shape_full, dtype=torch.bfloat16).float()
    torch_weight = torch.rand(HIDDEN_DIM, dtype=torch.bfloat16).float() * 2 - 1
    logger.info(f"Created torch input: {torch_input.shape}, weight: {torch_weight.shape}")

    # Compute PyTorch reference
    logger.info("Computing PyTorch reference output")
    torch_reference = reference_rmsnorm(torch_input, torch_weight, EPSILON)
    logger.info(f"PyTorch reference output shape: {torch_reference.shape}")

    # ============================================
    # Distributed RMSNorm
    # ============================================
    logger.info("Running distributed RMSNorm (tensor sharded across chips)")

    # Shard input across devices along width dimension (dim=3)
    tt_input_sharded = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, 3)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.info(f"Created sharded input: {tt_input_sharded.shape}")

    # Shard gamma weights across devices
    # Reshape weight to [1, 1, hidden_size//32, 32] for optimal performance
    tt_weight_sharded = ttnn.from_torch(
        torch_weight.reshape(1, 1, HIDDEN_DIM // 32, 32),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, 2)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.info(f"Created sharded weight: {tt_weight_sharded.shape}")

    # Step 1: Pre-all-gather - each device computes local sum(x^2)
    tt_stats = ttnn.rms_norm_pre_all_gather(
        tt_input_sharded,
        dtype=ttnn.bfloat16,
    )
    logger.info(f"Pre-all-gather stats shape: {tt_stats.shape}")

    # Step 2: All-gather stats along cluster_axis=1 (columns)
    tt_gathered_stats = ttnn.all_gather(
        tt_stats,
        dim=3,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
    )
    ttnn.deallocate(tt_stats)
    logger.info(f"Gathered stats shape: {tt_gathered_stats.shape}")

    # Step 3: Post-all-gather - normalize using gathered global stats
    tt_output_distributed = ttnn.rms_norm_post_all_gather(
        tt_input_sharded,
        tt_gathered_stats,
        epsilon=EPSILON,
        weight=tt_weight_sharded,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(tt_gathered_stats)
    logger.info(f"Distributed output shape (sharded): {tt_output_distributed.shape}")

    # Convert distributed output to torch (concat shards along dim 3)
    tt_distributed_torch = ttnn.to_torch(
        tt_output_distributed,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    logger.info(f"Distributed output (concatenated) shape: {tt_distributed_torch.shape}")

    # Clean up tensors
    ttnn.deallocate(tt_input_sharded)
    ttnn.deallocate(tt_weight_sharded)
    ttnn.deallocate(tt_output_distributed)

    # ============================================
    # Compare output against PyTorch reference
    # ============================================
    logger.info("Comparing distributed RMSNorm vs PyTorch reference")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_reference.to(torch.float32),
        tt_distributed_torch.to(torch.float32),
        pcc=0.99,
    )
    logger.info(f"Distributed vs PyTorch: {pcc_message}")
    assert pcc_passed, f"Distributed RMSNorm PCC test failed: {pcc_message}"

    logger.info("PCC test passed!")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="linear"),
            id="linear-4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_rmsnorm_replicated(mesh_device, device_params):
    """
    Test replicated RMSNorm (full tensor on each chip) against PyTorch reference.

    Note: This test may fail with L1 memory exceeded error for large hidden dimensions.
    Used for debugging the replicated path.
    """
    torch.manual_seed(1234)

    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape

    # 4D shapes
    inp_shape_full = (1, 1, BATCH_ROWS, HIDDEN_DIM)

    logger.info(f"Testing replicated RMSNorm with mesh_shape={mesh_shape}")
    logger.info(f"Full input: {inp_shape_full}")

    signpost(f"RMSNorm replicated test - mesh {mesh_shape}")

    # Create random input and weight tensors
    torch_input = torch.randn(inp_shape_full, dtype=torch.bfloat16).float()
    torch_weight = torch.rand(HIDDEN_DIM, dtype=torch.bfloat16).float() * 2 - 1
    logger.info(f"Created torch input: {torch_input.shape}, weight: {torch_weight.shape}")

    # Compute PyTorch reference
    logger.info("Computing PyTorch reference output")
    torch_reference = reference_rmsnorm(torch_input, torch_weight, EPSILON)
    logger.info(f"PyTorch reference output shape: {torch_reference.shape}")

    # ============================================
    # Replicated RMSNorm
    # ============================================
    logger.info("Running replicated RMSNorm (full tensor on each chip)")

    tt_input_replicated = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_weight_replicated = ttnn.from_torch(
        torch_weight.reshape(1, 1, 1, HIDDEN_DIM),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_output_replicated = ttnn.rms_norm(
        tt_input_replicated,
        epsilon=EPSILON,
        weight=tt_weight_replicated,
    )

    # Get output from first device (all are identical since replicated)
    tt_replicated_torch = ttnn.to_torch(
        tt_output_replicated,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    # Since replicated, take first replica
    tt_replicated_torch = tt_replicated_torch[:1, :1, :BATCH_ROWS, :HIDDEN_DIM]
    logger.info(f"Replicated output shape: {tt_replicated_torch.shape}")

    # Clean up
    ttnn.deallocate(tt_input_replicated)
    ttnn.deallocate(tt_weight_replicated)
    ttnn.deallocate(tt_output_replicated)

    # ============================================
    # Compare output against PyTorch reference
    # ============================================
    logger.info("Comparing replicated RMSNorm vs PyTorch reference")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_reference.to(torch.float32),
        tt_replicated_torch.to(torch.float32),
        pcc=0.99,
    )
    logger.info(f"Replicated vs PyTorch: {pcc_message}")
    assert pcc_passed, f"Replicated RMSNorm PCC test failed: {pcc_message}"

    logger.info("PCC test passed!")
