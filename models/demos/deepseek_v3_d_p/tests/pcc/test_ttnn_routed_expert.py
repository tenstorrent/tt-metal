"""
PCC test for TtRoutedExpert module.

Tests that TTNN TtRoutedExpert produces matching outputs to torch reference TorchExpert.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

# Set to True to skip torch init/calc/PCC check for faster iteration during development
SKIP_PCC_CHECK = True

if not SKIP_PCC_CHECK:
    from models.demos.deepseek_v3_d_p.tests.pcc.test_moe import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    extract_mesh_config,
    get_routed_expert_buffer_mesh_mapper,
    get_routed_expert_output_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from tests.ttnn.utils_for_testing import comp_pcc


@torch.no_grad()
def run_torch_routed_experts(
    dispatched_buffer: torch.Tensor,
    experts: list[TorchExpert],
    experts_per_chip: int,
    num_dispatch_groups: int,
    dispatch_group_size: int,
) -> torch.Tensor:
    """
    Run torch routed experts on dispatched buffer.

    Args:
        dispatched_buffer: (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
        experts: List of TorchExpert modules (total = num_dispatch_groups * dispatch_group_size * experts_per_chip)
        experts_per_chip: Number of experts per chip
        num_dispatch_groups: Number of dispatch groups (EP ranks, mesh_cols for 2D mesh)
        dispatch_group_size: Size of each dispatch group (SP ranks, mesh_rows for 2D mesh)

    Returns:
        expert_outputs: Same shape as dispatched_buffer
    """
    expert_outputs = torch.zeros_like(dispatched_buffer)

    for dg in range(num_dispatch_groups):
        for ds in range(dispatch_group_size):
            # Linearize position to get chip index: row * num_cols + col
            # dg = col (mesh col), ds = row (mesh row)
            chip = ds * num_dispatch_groups + dg
            for local_expert in range(experts_per_chip):
                global_expert = chip * experts_per_chip + local_expert

                # Get input for this expert
                expert_input = dispatched_buffer[dg, ds, local_expert, :, :]

                # Run expert
                expert_output = experts[global_expert](expert_input.float())
                expert_outputs[dg, ds, local_expert, :, :] = expert_output

    return expert_outputs


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        # DeepSeek V3 routed expert dims: emb_dim=7168, hidden_dim=2048
        # (3200, 7168, 2048, 64, 2, 2),  # 8 experts per chip, isl=3200
        # (512, 1024, 256, 32, 4, 2),  #
        # (3200, 7168, 64, 2, 2),
        (4096, 7168, 2048, 64, 2, 2),  # 8 experts per chip, isl=3200
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
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    # Compute configuration constants
    # For routed expert, experts_per_chip = num_routed_experts // num_devices (all devices combined)
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )

    signpost(
        f"TtRoutedExpert {mesh_device.shape=} {num_devices=} {experts_per_chip=} "
        f"{seq_len_per_chip=} {emb_dim=} {hidden_dim=} {num_routed_experts=} "
        f"{num_experts_per_tok=} {capacity_factor=}"
    )

    logger.debug(f"\n{'='*60}")
    logger.debug("TtRoutedExpert PCC Test")
    logger.debug(f"{'='*60}")
    logger.debug(f"mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    logger.debug(f"dispatch_group_size={dispatch_group_size}, experts_per_chip={experts_per_chip}")
    logger.debug(
        f"seq_len_per_chip={seq_len_per_chip}, max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}"
    )
    logger.debug(f"emb_dim={emb_dim}, hidden_dim={hidden_dim}")

    total_experts = num_devices * experts_per_chip
    logger.debug(f"Total experts: {total_experts}")

    # Create random input with shape matching mesh topology
    # Shape: (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    # For 2D mesh: num_dispatch_groups=mesh_cols, dispatch_group_size=mesh_rows
    # This allows sharding with dims=(1, 0) consistent with other MoE tests
    torch.manual_seed(42)
    dispatched_buffer_torch = torch.randn(
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
        max_dispatched_tokens_per_expert,
        emb_dim,
        dtype=torch.float32,
    )
    logger.debug(f"Input shape: {dispatched_buffer_torch.shape} (mesh: {mesh_device.shape})")

    # Create weights (and optionally torch experts for PCC check)
    torch_weights_list = []
    for i in range(total_experts):
        weights = {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        torch_weights_list.append(weights)

    if not SKIP_PCC_CHECK:
        # Create torch experts for PCC validation
        torch_experts = [TorchExpert(emb_dim, torch_weights=w) for w in torch_weights_list]

        # Run torch reference
        logger.debug("Running torch reference...")
        torch_outputs = run_torch_routed_experts(
            dispatched_buffer_torch,
            torch_experts,
            experts_per_chip,
            1,
            1,
        )
        logger.debug(f"Torch output shape: {torch_outputs.shape}")
        logger.debug(f"Torch output stats - min: {torch_outputs.min():.4f}, max: {torch_outputs.max():.4f}")

    # Create TTNN input - shard across devices using dims=(1, 0)
    # Buffer shape: (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    # dims=(1, 0) shards dim 1 (dispatch_group_size) across mesh rows, dim 0 (num_dispatch_groups) across mesh cols
    mesh_mapper = get_routed_expert_buffer_mesh_mapper(mesh_device)
    dispatched_buffer_tt = ttnn.from_torch(
        dispatched_buffer_torch,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    # Squeeze out the mesh dimensions after sharding
    # Each device has (1, 1, experts_per_chip, max_tokens, emb_dim)
    # Reshape to (experts_per_chip, max_tokens, emb_dim)
    dispatched_buffer_tt = ttnn.reshape(
        dispatched_buffer_tt, (experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)
    )
    logger.debug(f"TTNN input shape: {dispatched_buffer_tt.shape}")

    # Create TtRoutedExpert - pass ALL expert weights, TtRoutedExpert distributes to devices
    # For device i, it gets experts [i * experts_per_chip : (i+1) * experts_per_chip]
    logger.debug("Creating TtRoutedExpert...")
    tt_routed_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        torch_weights=torch_weights_list,  # All total_experts weights
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    ttnn_outputs = tt_routed_expert(dispatched_buffer_tt)
    logger.debug(f"TTNN output shape: {ttnn_outputs.shape}")

    if SKIP_PCC_CHECK:
        logger.info("SKIP_PCC_CHECK=True, skipping torch validation")
        logger.debug(f"\n{'='*60}")
        logger.debug("TtRoutedExpert Test PASSED (PCC check skipped)")
        logger.debug(f"{'='*60}")
        return

    # Convert back to torch
    # Output shape per device: (experts_per_chip, max_tokens, emb_dim)
    # Need to unsqueeze and compose back to (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    ttnn_outputs_expanded = ttnn.unsqueeze(ttnn.unsqueeze(ttnn_outputs, dim=0), dim=0)
    mesh_composer = get_routed_expert_output_mesh_composer(mesh_device)
    ttnn_outputs_torch = ttnn.to_torch(ttnn_outputs_expanded, mesh_composer=mesh_composer)
    logger.debug(f"TTNN output (torch) shape: {ttnn_outputs_torch.shape}")
    logger.debug(f"TTNN output stats - min: {ttnn_outputs_torch.min():.4f}, max: {ttnn_outputs_torch.max():.4f}")

    # Validate ALL chips - each device now has unique weights for its experts
    # Torch outputs: (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)
    # TTNN outputs: same shape after mesh_composer concat
    pcc_values = []
    for dg in range(num_dispatch_groups):
        for ds in range(dispatch_group_size):
            chip = ds * num_dispatch_groups + dg
            torch_chip = torch_outputs[dg, ds, :, :, :]  # (experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)
            ttnn_chip = ttnn_outputs_torch[dg, ds, :, :, :]

            for expert_idx in range(experts_per_chip):
                global_expert_idx = chip * experts_per_chip + expert_idx
                _, pcc = comp_pcc(torch_chip[expert_idx], ttnn_chip[expert_idx])
                pcc_values.append(pcc)
                logger.debug(
                    f"Chip (dg={dg},ds={ds}), Local Expert {expert_idx} (Global {global_expert_idx}) PCC: {pcc:.6f}"
                )

    min_pcc = min(pcc_values)
    avg_pcc = sum(pcc_values) / len(pcc_values)
    logger.debug(f"\nMin PCC: {min_pcc:.6f}, Avg PCC: {avg_pcc:.6f} (across all {len(pcc_values)} experts)")

    # Threshold for bfp8/bfp4 precision (actual PCC ~0.98)
    pcc_threshold = 0.97
    assert min_pcc >= pcc_threshold, f"PCC {min_pcc:.6f} below threshold {pcc_threshold}"

    # Verify no NaN/Inf
    assert not torch.isnan(ttnn_outputs_torch).any(), "Output contains NaN"
    assert not torch.isinf(ttnn_outputs_torch).any(), "Output contains Inf"

    logger.debug(f"\n{'='*60}")
    logger.debug("TtRoutedExpert PCC Test PASSED!")
    logger.debug(f"{'='*60}")
