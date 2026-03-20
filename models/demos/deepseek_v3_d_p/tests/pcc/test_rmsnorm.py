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


@pytest.mark.parametrize(
    "isl_per_chip, hidden_dim, epsilon, num_links", [(3200, 7168, 1e-6, 1), (4096, 7168, 1e-6, 1)], ids=["3.2K", "4K"]
)
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
def test_rmsnorm_distributed(mesh_device, device_params, isl_per_chip, hidden_dim, epsilon, num_links):
    """
    Test distributed RMSNorm against PyTorch reference.

    Input tensor is sharded across 4 chips along hidden dimension.
    Each chip computes local statistics, then all-gather combines them
    for global normalization.

    Configuration:
    - Input shape: (1, 1, isl_per_chip, 7168) - 4D format
    - Per device: (1, 1, isl_per_chip, 1792) - 7168 / 4 = 1792
    - Stats gathered along cluster_axis=1 (mesh columns)
    """
    torch.manual_seed(1234)

    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    per_device_width = hidden_dim // num_devices

    # 4D shapes for distributed RMSNorm
    inp_shape_full = (1, 1, isl_per_chip, hidden_dim)
    inp_shape_per_device = (1, 1, isl_per_chip, per_device_width)

    logger.debug(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.debug(f"Full input: {inp_shape_full}, per-device: {inp_shape_per_device}")

    # Determine topology from fabric config
    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    topology = ttnn.Topology.Ring if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING else ttnn.Topology.Linear
    logger.debug(f"Using topology: {topology}")

    signpost(f"RMSNorm PCC test - {mesh_shape=} {isl_per_chip=} {hidden_dim=} {num_links=} {topology=}")

    # Create random input tensor
    torch_input = torch.randn(inp_shape_full, dtype=torch.bfloat16).float()
    logger.debug(f"Created torch input: {torch_input.shape}")

    # Create PyTorch RMSNorm reference with random weights
    rmsnorm = torch.nn.RMSNorm(hidden_dim, eps=epsilon)
    rmsnorm.weight.data.uniform_(-1, 1)
    logger.debug(f"Created torch.nn.RMSNorm with weight: {rmsnorm.weight.shape}")

    # Compute PyTorch reference
    logger.debug("Computing PyTorch reference output")
    torch_reference = rmsnorm(torch_input)
    logger.debug(f"PyTorch reference output shape: {torch_reference.shape}")

    # ============================================
    # Distributed RMSNorm using TtDistributedRmsNorm module
    # ============================================
    logger.debug("Running distributed RMSNorm (tensor sharded across chips)")

    # Initialize TtDistributedRmsNorm module
    distributed_rmsnorm = TtDistributedRmsNorm(
        mesh_device=mesh_device,
        hidden_dim=hidden_dim,
        epsilon=epsilon,
        torch_weight=rmsnorm.weight,
        cluster_axis=1,
        num_links=num_links,
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
    logger.debug(f"Created sharded input: {tt_input_sharded.shape}")

    # Run distributed RMSNorm forward pass
    tt_output_distributed = distributed_rmsnorm(tt_input_sharded)
    logger.debug(f"Distributed output shape (sharded): {tt_output_distributed.shape}")

    # Convert distributed output to torch (concat shards along dim 3)
    tt_distributed_torch = ttnn.to_torch(
        tt_output_distributed,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(0, 3)),
    )
    logger.debug(f"Distributed output (concatenated) shape: {tt_distributed_torch.shape}")

    # ============================================
    # Compare output against PyTorch reference
    # ============================================
    logger.debug("Comparing distributed RMSNorm vs PyTorch reference")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_reference.to(torch.float32),
        tt_distributed_torch.to(torch.float32),
        pcc=0.99,
    )
    logger.debug(f"Distributed vs PyTorch: {pcc_message}")
    assert pcc_passed, f"Distributed RMSNorm PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")


@pytest.mark.parametrize(
    "isl_per_chip, hidden_dim, epsilon", [(3200, 7168, 1e-6), (4096, 7168, 1e-6)], ids=["3.2K", "4K"]
)
def test_rmsnorm_single_chip(device, isl_per_chip, hidden_dim, epsilon):
    """
    Test single-chip full dimension RMSNorm against PyTorch reference.

    Input tensor runs on a single device with full hidden dimension.

    Configuration:
    - Input shape: (1, 1, isl_per_chip, hidden_dim) - 4D format
    - Full hidden dimension on single chip
    """
    torch.manual_seed(1234)

    inp_shape = (1, 1, isl_per_chip, hidden_dim)

    logger.debug(f"Testing single-chip RMSNorm with shape={inp_shape}")
    signpost(f"RMSNorm PCC test - single-chip {isl_per_chip=} {hidden_dim=}")

    # Create random input tensor
    torch_input = torch.randn(inp_shape, dtype=torch.bfloat16).float()
    logger.debug(f"Created torch input: {torch_input.shape}")

    # Create PyTorch RMSNorm reference with random weights
    rmsnorm = torch.nn.RMSNorm(hidden_dim, eps=epsilon)
    rmsnorm.weight.data.uniform_(-1, 1)
    logger.debug(f"Created torch.nn.RMSNorm with weight: {rmsnorm.weight.shape}")

    # Compute PyTorch reference
    torch_reference = rmsnorm(torch_input)
    logger.debug(f"PyTorch reference output shape: {torch_reference.shape}")

    # Convert input to TT tensor
    tt_input = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    # Convert weight to TT tensor
    tt_weight = ttnn.from_torch(
        rmsnorm.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    # Run single-chip RMSNorm
    tt_output = ttnn.rms_norm(tt_input, weight=tt_weight, epsilon=epsilon)
    logger.debug(f"TT output shape: {tt_output.shape}")

    # Convert output to torch
    tt_output_torch = ttnn.to_torch(tt_output)
    logger.debug(f"TT output (torch) shape: {tt_output_torch.shape}")

    # Compare output against PyTorch reference
    logger.debug("Comparing single-chip RMSNorm vs PyTorch reference")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_reference.to(torch.float32),
        tt_output_torch.to(torch.float32),
        pcc=0.99,
    )
    logger.debug(f"Single-chip vs PyTorch: {pcc_message}")
    assert pcc_passed, f"Single-chip RMSNorm PCC test failed: {pcc_message}"

    logger.debug("PCC test passed!")
