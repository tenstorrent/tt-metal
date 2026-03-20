"""
Minimal single-device, single-expert test for TtRoutedExpert profiling.

The simplest scenario: 1 chip, 1 expert, minimal dimensions.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (1024, 7168, 2048),  # DeepSeek V3 dims, 1K tokens
        (1600, 7168, 2048),  # DeepSeek V3 dims, 1.6K tokens
        (2048, 7168, 2048),  # DeepSeek V3 dims, 2K tokens
        (3200, 7168, 2048),  # DeepSeek V3 dims, 3.2K tokens
        (4096, 7168, 2048),  # DeepSeek V3 dims, 4K tokens
    ],
    ids=["ds-v3-1k", "ds-v3-1.6k", "ds-v3-2k", "ds-v3-3.2k", "ds-v3-4k"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-chip",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_single_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """
    Simplest test: 1 chip, 1 expert.

    Perfect for profiling the core FFN computation without any mesh complexity.
    """
    experts_per_chip = 1

    signpost(f"SingleRoutedExpert {num_tokens=} {emb_dim=} {hidden_dim=}")

    logger.debug(f"Testing single routed expert: {num_tokens=}, {emb_dim=}, {hidden_dim=}")
    logger.debug(f"Mesh: {mesh_device.shape}, num_devices={mesh_device.get_num_devices()}")

    # Create random weights
    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    # Create torch reference
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)

    # Create random input: (experts_per_chip, num_tokens, emb_dim)
    torch_input = torch.randn(experts_per_chip, num_tokens, emb_dim, dtype=torch.float32)
    logger.debug(f"Input shape: {torch_input.shape}")

    # Run torch reference
    logger.debug("Running torch reference...")
    with torch.no_grad():
        torch_output = torch_expert(torch_input[0])  # Process first (only) expert's tokens
    logger.debug(f"Torch output shape: {torch_output.shape}")

    # Create TTNN input
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.debug(f"TTNN input shape: {tt_input.shape}")

    # Create TtRoutedExpert
    logger.debug("Creating TtRoutedExpert...")
    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=num_tokens,
        torch_weights=[weights],  # List with single expert weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    tt_output = tt_expert(tt_input)
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Convert back to torch for comparison
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    # Extract the single expert output: (experts_per_chip, num_tokens, emb_dim) -> (num_tokens, emb_dim)
    tt_output_single = tt_output_torch[0]
    logger.debug(f"TTNN output (torch) shape: {tt_output_single.shape}")

    # Compare PCC
    _, pcc = comp_pcc(torch_output, tt_output_single)
    logger.debug(f"PCC: {pcc:.6f}")

    # Validate
    pcc_threshold = 0.97
    assert pcc >= pcc_threshold, f"PCC {pcc:.6f} below threshold {pcc_threshold}"
    assert not torch.isnan(tt_output_torch).any(), "Output contains NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output contains Inf"

    logger.debug("Test PASSED!")
