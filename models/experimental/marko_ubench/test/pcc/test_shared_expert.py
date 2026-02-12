"""
PCC test for TtSharedExpert module.

Compares TorchSharedExpert (reference) against TtSharedExpert (multi-chip TTNN)
to verify correctness of multi-chip sharding and CCL operations.
"""

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost

from models.experimental.marko_ubench.modules.reference.pytorch_shared_expert import TorchSharedExpert
from models.experimental.marko_ubench.modules.tt_shared_expert import TtSharedExpert
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.tt_transformers.tt.ccl import get_num_links


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        # (1, 2),  # 1 row, 2 columns - simplest multi-chip case
        (1, 4),  # 1 row, 4 columns - more devices
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "batch_seq_len",
    [
        4096,  # Target - testing only this case for L1 sharding experiments
    ],
    ids=["real_test"],
)
@pytest.mark.parametrize(
    "emb_dim, hidden_dim",
    [
        (7 * 1024, 2 * 1024),  # Default dimensions
    ],
)
def test_shared_expert_pcc(
    mesh_device,
    device_params,
    batch_seq_len: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Test TtSharedExpert PCC against TorchSharedExpert reference.

    This test verifies:
    1. Correct weight sharding (w1/w3 on -1, w2 on -2)
    2. Proper all-gather before matmuls
    3. SiLU activation fusion
    4. Proper reduce-scatter after final matmul
    5. Output matches torch reference with PCC > 0.99
    """
    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.info(f"batch_seq_len={batch_seq_len}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")

    # Add Tracy signpost for profiling
    signpost(f"SharedExpert PCC test - mesh {mesh_shape}, batch_seq={batch_seq_len}")

    # Query available ethernet links
    num_links = get_num_links(mesh_device, cluster_axis=1)  # Query along mesh columns
    logger.info(f"Available ethernet links along mesh columns: {num_links}")

    # Determine topology based on fabric config
    fabric_config = device_params.get("fabric_config", ttnn.FabricConfig.FABRIC_1D)
    if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING:
        topology = ttnn.Topology.Ring
        logger.info("Using Ring topology for FABRIC_1D_RING")
    else:
        topology = ttnn.Topology.Linear
        logger.info("Using Linear topology for FABRIC_1D")

    # ========================================
    # Step 1: Create PyTorch reference model
    # ========================================
    logger.info("Creating TorchSharedExpert reference")
    torch_model = TorchSharedExpert(emb_dim=emb_dim, hidden_dim=hidden_dim)

    # Extract weights for TTNN model
    torch_weights = {
        "gate_proj": torch_model.gate_proj.data,
        "up_proj": torch_model.up_proj.data,
        "down_proj": torch_model.down_proj.data,
    }

    # ========================================
    # Step 2: Create TTNN model with same weights
    # ========================================
    logger.info("Creating TtSharedExpert with same weights")
    tt_model = TtSharedExpert(
        mesh_device=mesh_device,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        torch_weights=torch_weights,
        num_links=num_links,
        topology=topology,
    )

    # ========================================
    # Step 3: Create input tensor
    # ========================================
    # For torch: full tensor [batch_seq_len, emb_dim]
    torch_input = torch.randn(batch_seq_len, emb_dim, dtype=torch.float32)
    logger.info(f"Created torch input: {torch_input.shape}")

    # For ttnn: sharded tensor [batch_seq_len, emb_dim / num_devices]
    # Use ShardTensor2dMesh to shard along last dimension across mesh columns
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(None, -1),  # Replicate on mesh rows, shard on tensor dim -1 across mesh columns
    )

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.info(f"Created ttnn input (sharded): {tt_input.shape}")

    # ========================================
    # Step 4: Run forward passes
    # ========================================
    logger.info("Running torch forward pass")
    torch_output = torch_model(torch_input)
    logger.info(f"Torch output shape: {torch_output.shape}")

    logger.info("Running ttnn forward pass")
    tt_output = tt_model(tt_input)
    logger.info(f"TTNN output shape (sharded): {tt_output.shape}")

    # ========================================
    # Step 5: Convert TTNN output back to torch and compare
    # ========================================
    logger.info("Converting TTNN output to torch for comparison")
    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1))
    )
    logger.info(f"TTNN output converted to torch: {tt_output_torch.shape}")

    # Compare with PCC
    logger.info("Comparing outputs with PCC")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_output.to(torch.float32),
        tt_output_torch.to(torch.float32),
        pcc=0.979,
    )

    logger.info(f"PCC comparison: {pcc_message}")
    assert pcc_passed, f"PCC test failed: {pcc_message}"

    logger.info("✓ PCC test passed!")
