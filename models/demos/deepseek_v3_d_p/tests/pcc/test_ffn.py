"""
PCC test for TtFFN module.

Compares TorchSharedExpert (reference) against TtFFN (multi-chip TTNN)
to verify correctness with DeepSeek 671B FFN dimensions.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.moe.shared_expert import TorchSharedExpert
from models.demos.deepseek_v3_d_p.tt.tt_ffn import EMB_DIM, HIDDEN_DIM, TtFfn
from models.tt_transformers.tt.ccl import get_num_links
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_seq_len", [4096])
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
def test_ffn_pcc(
    mesh_device,
    device_params,
    batch_seq_len: int,
    num_links: int,
    topology: ttnn.Topology,
):
    """
    Test TtFfn PCC against TorchSharedExpert reference.

    Uses DeepSeek 671B dimensions:
        - emb_dim: 7168
        - hidden_dim: 18432
        - activations: bfloat16
    """
    num_devices = mesh_device.get_num_devices()
    mesh_shape = mesh_device.shape
    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    logger.info(f"batch_seq_len={batch_seq_len}, emb_dim={EMB_DIM}, hidden_dim={HIDDEN_DIM}")

    signpost(f"FFN PCC test - mesh {mesh_shape}, batch_seq={batch_seq_len}")

    actual_num_links = get_num_links(mesh_device, cluster_axis=1)
    logger.info(f"Available ethernet links along mesh columns: {actual_num_links}")
    logger.info(f"Using num_links={num_links}, topology={topology}")

    # Create PyTorch reference model with FFN dimensions
    logger.info("Creating TorchSharedExpert reference with FFN dimensions")
    torch_model = TorchSharedExpert(emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)

    torch_weights = {
        "gate_proj": torch_model.gate_proj.data,
        "up_proj": torch_model.up_proj.data,
        "down_proj": torch_model.down_proj.data,
    }

    # Create TTNN FFN model
    logger.info("Creating TtFfn with same weights")
    tt_model = TtFfn(
        mesh_device=mesh_device,
        torch_weights=torch_weights,
        num_links=num_links,
        topology=topology,
    )

    # Create input tensor (replicated across all devices)
    torch_input = torch.randn(batch_seq_len, EMB_DIM, dtype=torch.float32)
    logger.info(f"Created torch input: {torch_input.shape}")

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    logger.info(f"Created ttnn input (replicated): {tt_input.shape}")

    # Run forward passes
    logger.info("Running torch forward pass")
    torch_output = torch_model(torch_input)
    logger.info(f"Torch output shape: {torch_output.shape}")

    logger.info("Running ttnn forward pass")
    tt_output = tt_model(tt_input)
    logger.info(f"TTNN output shape (sharded): {tt_output.shape}")

    # Convert and compare
    logger.info("Converting TTNN output to torch for comparison")
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
    )
    logger.info(f"TTNN output converted to torch: {tt_output_torch.shape}")

    logger.info("Comparing outputs with PCC")
    pcc_passed, pcc_message = assert_with_pcc(
        torch_output.to(torch.float32),
        tt_output_torch.to(torch.float32),
        pcc=0.97,
    )

    logger.info(f"PCC comparison: {pcc_message}")
    assert pcc_passed, f"PCC test failed: {pcc_message}"

    logger.info("PCC test passed!")
