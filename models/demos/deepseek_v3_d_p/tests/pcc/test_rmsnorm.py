"""
PCC test for distributed RMSNorm.

Compares distributed RMSNorm (rms_norm_pre_all_gather + all_gather + rms_norm_post_all_gather)
against PyTorch reference implementation.

Tests that hidden dimension sharding across chips produces correct results when
using the distributed RMSNorm approach with all-gather for global statistics.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from tests.ttnn.utils_for_testing import assert_with_pcc

# Constants matching DeepSeek 671B dimensions
# Input shape: (1, 1, batch_rows, hidden_dim) - 4D format required for distributed RMSNorm
BATCH_ROWS = 32  # Tile-aligned batch dimension
HIDDEN_DIM = 7168
EPSILON = 1e-6


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="linear"),
            id="linear-4",
        ),
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="ring-4",
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

    # Create random input tensor
    torch_input = torch.randn(inp_shape_full, dtype=torch.bfloat16).float()
    logger.info(f"Created torch input: {torch_input.shape}")

    # Create PyTorch RMSNorm reference with random weights
    rmsnorm = torch.nn.RMSNorm(HIDDEN_DIM, eps=EPSILON)
    rmsnorm.weight.data.uniform_(-1, 1)
    logger.info(f"Created torch.nn.RMSNorm with weight: {rmsnorm.weight.shape}")

    # Compute PyTorch reference
    logger.info("Computing PyTorch reference output")
    torch_reference = rmsnorm(torch_input)
    logger.info(f"PyTorch reference output shape: {torch_reference.shape}")

    # ============================================
    # Distributed RMSNorm using TtDistributedRmsNorm module
    # ============================================
    logger.info("Running distributed RMSNorm (tensor sharded across chips)")

    # Determine topology from fabric config
    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear
    logger.info(f"Using topology: {topology}")

    # Initialize TtDistributedRmsNorm module
    distributed_rmsnorm = TtDistributedRmsNorm(
        mesh_device=mesh_device,
        hidden_dim=HIDDEN_DIM,
        epsilon=EPSILON,
        torch_weight=rmsnorm.weight,
        cluster_axis=1,
        num_links=1,
        topology=topology,
    )

    # Shard input across devices along width dimension (dim=3)
    tt_input_sharded = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(None, 3)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.info(f"Created sharded input: {tt_input_sharded.shape}")

    # Run distributed RMSNorm forward pass
    tt_output_distributed = distributed_rmsnorm(tt_input_sharded)
    logger.info(f"Distributed output shape (sharded): {tt_output_distributed.shape}")

    # Convert distributed output to torch (concat shards along dim 3)
    tt_distributed_torch = ttnn.to_torch(
        tt_output_distributed,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    logger.info(f"Distributed output (concatenated) shape: {tt_distributed_torch.shape}")

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
