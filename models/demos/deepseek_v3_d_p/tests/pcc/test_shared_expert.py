"""
PCC test for TtSharedExpert module.

Compares TorchSharedExpert (reference) against TtSharedExpert (multi-chip TTNN)
to verify correctness of multi-chip sharding and CCL operations.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.moe.shared_expert import TorchSharedExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
from models.tt_transformers.tt.ccl import get_num_links
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "batch_seq_len, emb_dim, hidden_dim",
    [
        (4096, 7 * 1024, 2 * 1024),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="linear"),
            id="linear-4",
        ),
        pytest.param(
            (1, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="ring-4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_shared_expert_pcc(
    mesh_device,
    device_params,
    batch_seq_len: int,
    emb_dim: int,
    hidden_dim: int,
    num_links: int,
    topology: ttnn.Topology,
):
    """
    Test TtSharedExpert PCC against TorchSharedExpert reference.

    This test verifies:
    1. Correct weight sharding (gate_proj/up_proj on -1, down_proj on -2)
    2. Proper all-gather before matmuls
    3. SiLU activation fusion
    4. Proper reduce-scatter after final matmul
    5. Output matches torch reference with PCC > 0.97
    """
    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.info(f"batch_seq_len={batch_seq_len}, emb_dim={emb_dim}, hidden_dim={hidden_dim}")

    # Add Tracy signpost for profiling
    signpost(f"SharedExpert PCC test - mesh {mesh_shape}, batch_seq={batch_seq_len}")

    # Query available ethernet links
    actual_num_links = get_num_links(mesh_device, cluster_axis=1)  # Query along mesh columns
    logger.info(f"Available ethernet links along mesh columns: {actual_num_links}")
    logger.info(f"Using num_links={num_links}, topology={topology}")

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

    # For ttnn: replicated tensor [batch_seq_len, emb_dim] on all devices
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.info(f"Created ttnn input (replicated): {tt_input.shape}")

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
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
    )
    logger.info(f"TTNN output converted to torch: {tt_output_torch.shape}")

    # Compare with PCC
    logger.info("Comparing outputs with PCC")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_output.to(torch.float32),
        tt_output_torch.to(torch.float32),
        pcc=0.97,
    )

    logger.info(f"PCC comparison: {pcc_message}")
    assert pcc_passed, f"PCC test failed: {pcc_message}"

    logger.info("PCC test passed!")
