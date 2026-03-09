"""
PCC test for TtRoutedExpert module.

Tests that TTNN TtRoutedExpert produces matching outputs to torch reference TorchExpert.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.pcc.test_moe import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


def run_torch_routed_experts(
    dispatched_buffer: torch.Tensor,
    experts: list[TorchExpert],
    experts_per_chip: int,
    dispatch_group_size: int,
) -> torch.Tensor:
    """
    Run torch routed experts on dispatched buffer.

    Args:
        dispatched_buffer: (1, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
        experts: List of TorchExpert modules (total = dispatch_group_size * experts_per_chip)
        experts_per_chip: Number of experts per chip
        dispatch_group_size: Number of chips

    Returns:
        expert_outputs: Same shape as dispatched_buffer
    """
    expert_outputs = torch.zeros_like(dispatched_buffer)

    for chip in range(dispatch_group_size):
        for local_expert in range(experts_per_chip):
            global_expert = chip * experts_per_chip + local_expert

            # Get input for this expert
            expert_input = dispatched_buffer[0, chip, local_expert, :, :]

            # Run expert
            with torch.no_grad():
                expert_output = experts[global_expert](expert_input.float())

            expert_outputs[0, chip, local_expert, :, :] = expert_output

    return expert_outputs


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        # DeepSeek V3 routed expert dims: emb_dim=7168, hidden_dim=2048
        (3200, 7168, 2048, 8, 4, 2),  # 8 experts per chip, isl=3200
    ],
    ids=["deepseek-v3-dims"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-1",
        ),
        pytest.param(
            (4, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4",
        ),
        pytest.param(
            (8, 1),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
        pytest.param(
            (4, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-4x2"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_routed_expert(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
):
    """
    Test TtRoutedExpert with DeepSeek V3 dimensions on various mesh configurations.

    Validates that the TTNN routed expert FFN computation matches torch reference.
    Each device processes its local experts independently (no CCL needed).
    """
    num_devices = mesh_device.get_num_devices()
    dispatch_group_size = num_devices  # Each device is one chip in the dispatch group

    # Compute configuration constants
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, capacity_factor
    )

    signpost(
        f"TtRoutedExpert {mesh_device.shape=} {num_devices=} {experts_per_chip=} "
        f"{seq_len_per_chip=} {emb_dim=} {hidden_dim=} {num_routed_experts=} "
        f"{num_experts_per_tok=} {capacity_factor=}"
    )

    logger.info(f"\n{'='*60}")
    logger.info("TtRoutedExpert PCC Test")
    logger.info(f"{'='*60}")
    logger.info(f"mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    logger.info(f"dispatch_group_size={dispatch_group_size}, experts_per_chip={experts_per_chip}")
    logger.info(
        f"seq_len_per_chip={seq_len_per_chip}, max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}"
    )
    logger.info(f"emb_dim={emb_dim}, hidden_dim={hidden_dim}")

    total_experts = dispatch_group_size * experts_per_chip
    logger.info(f"Total experts: {total_experts}")

    # Create random input
    torch.manual_seed(42)
    dispatched_buffer_torch = torch.randn(
        1, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, emb_dim, dtype=torch.float32
    )
    logger.info(f"Input shape: {dispatched_buffer_torch.shape}")

    # Create torch experts with random weights (one per global expert)
    torch_experts = []
    torch_weights_list = []

    for i in range(total_experts):
        weights = {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        torch_experts.append(TorchExpert(emb_dim, torch_weights=weights))
        torch_weights_list.append(weights)

    # Run torch reference
    logger.info("Running torch reference...")
    torch_outputs = run_torch_routed_experts(
        dispatched_buffer_torch,
        torch_experts,
        experts_per_chip,
        dispatch_group_size,
    )
    logger.info(f"Torch output shape: {torch_outputs.shape}")
    logger.info(f"Torch output stats - min: {torch_outputs.min():.4f}, max: {torch_outputs.max():.4f}")

    # Create TTNN input - shard across devices along dispatch_group_size dimension
    # dims=(0, None) means: shard tensor dim 0 (after removing batch) across mesh rows, replicate across cols
    # But our tensor is (1, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)
    # We want to shard dim 1 (dispatch_group_size) across devices
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(1, None),  # Shard dim 1 across mesh rows (devices), no replication on cols
    )

    dispatched_buffer_tt = ttnn.from_torch(
        dispatched_buffer_torch,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    logger.info(f"TTNN input shape: {dispatched_buffer_tt.shape}")

    # Create TtRoutedExpert - each device gets its local experts' weights
    # For device i, it gets experts [i * experts_per_chip : (i+1) * experts_per_chip]
    # Since TtRoutedExpert replicates weights to all devices, we just pass the first chip's weights
    # In a real multi-device scenario, each device would have different weights loaded
    # For this test, we use the same experts_per_chip weights on all devices (replicated)
    expert_weights_for_device = torch_weights_list[:experts_per_chip]

    logger.info("Creating TtRoutedExpert...")
    tt_routed_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        torch_weights=expert_weights_for_device,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    # Run TTNN forward
    logger.info("Running TTNN forward...")
    ttnn_outputs = tt_routed_expert(dispatched_buffer_tt)
    logger.info(f"TTNN output shape: {ttnn_outputs.shape}")

    # Convert back to torch - concat shards back along dim 1
    # dims=[1, 0] means: mesh axis 0 (rows) maps to tensor dim 1, mesh axis 1 (cols) maps to tensor dim 0
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(dims=[1, 0]),
    )
    ttnn_outputs_torch = ttnn.to_torch(ttnn_outputs, mesh_composer=mesh_composer)
    logger.info(f"TTNN output (torch) shape: {ttnn_outputs_torch.shape}")
    logger.info(f"TTNN output stats - min: {ttnn_outputs_torch.min():.4f}, max: {ttnn_outputs_torch.max():.4f}")

    # For multi-device test, weights are replicated so each device computes with same weights
    # But torch reference uses different weights per global expert
    # So we compare only device 0's output with torch's device 0 output
    # (since TtRoutedExpert uses experts_per_chip weights replicated to all devices)
    torch_chip0 = torch_outputs[0, 0, :, :, :]  # (experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)
    ttnn_chip0 = ttnn_outputs_torch[0, 0, :, :, :]

    # Compute PCC per expert for chip 0
    pcc_values = []
    for expert_idx in range(experts_per_chip):
        _, pcc = comp_pcc(torch_chip0[expert_idx], ttnn_chip0[expert_idx])
        pcc_values.append(pcc)
        logger.info(f"Expert {expert_idx} PCC: {pcc:.6f}")

    min_pcc = min(pcc_values)
    avg_pcc = sum(pcc_values) / len(pcc_values)
    logger.info(f"\nMin PCC: {min_pcc:.6f}, Avg PCC: {avg_pcc:.6f}")

    # Threshold for bfp8/bfp4 precision (actual PCC ~0.98)
    pcc_threshold = 0.97
    assert min_pcc >= pcc_threshold, f"PCC {min_pcc:.6f} below threshold {pcc_threshold}"

    # Verify no NaN/Inf
    assert not torch.isnan(ttnn_outputs_torch).any(), "Output contains NaN"
    assert not torch.isinf(ttnn_outputs_torch).any(), "Output contains Inf"

    logger.info(f"\n{'='*60}")
    logger.info("TtRoutedExpert PCC Test PASSED!")
    logger.info(f"{'='*60}")
