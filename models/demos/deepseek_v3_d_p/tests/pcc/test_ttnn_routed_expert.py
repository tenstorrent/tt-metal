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
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
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
            # dg = col (mesh col / dispatch group), ds = row (mesh row / chip within group)
            for local_expert in range(experts_per_chip):
                # Use column-major ordering to match TorchMoe and TtRoutedExpert
                global_expert = ExpertMapping.get_global_expert_idx(
                    group=dg,
                    chip=ds,
                    local_expert=local_expert,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    is_col_major=True,
                )

                # Get input for this expert
                expert_input = dispatched_buffer[dg, ds, local_expert, :, :]

                # Run expert
                expert_output = experts[global_expert](expert_input.float())
                expert_outputs[dg, ds, local_expert, :, :] = expert_output

    return expert_outputs


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor, run_pcc_check",
    [
        # DeepSeek V3 routed expert dims: emb_dim=7168, hidden_dim=2048
        (3200, 7168, 2048, 64, 2, 2, False),  # 8 experts per chip, isl=3200
        # (3200, 7168, 2048, 8, 1, 2, True),
        # (4096, 7168, 2048, 64, 2, 2, False),  # 8 experts per chip, isl=4096
    ],
    # ids=["deepseek-v3-dims"],
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
    run_pcc_check,
):
    """
    Test TtRoutedExpert with DeepSeek V3 dimensions on various mesh configurations.

    Validates that the TTNN routed expert FFN computation matches torch reference.
    Each device processes its local experts independently (no CCL needed).
    """
    profiler.clear()
    profiler.start("test_ttnn_routed_expert")
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
        f"TtRoutedExpert {mesh_device.shape=} {num_devices=} {experts_per_chip=}"
        f"\n{seq_len_per_chip=} {emb_dim=} {hidden_dim=} {num_routed_experts=}"
        f"\n{num_experts_per_tok=} {capacity_factor=}"
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

    # Input shape: (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
    # For 2D mesh: num_dispatch_groups=mesh_cols, dispatch_group_size=mesh_rows
    # Final per-device shape after sharding: (experts_per_chip, max_tokens, emb_dim)
    per_device_shape = (experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)

    # Create weights and input only if PCC check is enabled
    # When run_pcc_check=False, use fast device-side allocation (no host-to-device transfer)
    torch_weights_list = None
    if run_pcc_check:
        # Create torch input for reference comparison
        torch.manual_seed(42)
        profiler.start("torch_input_creation")
        dispatched_buffer_torch = torch.randn(
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            max_dispatched_tokens_per_expert,
            emb_dim,
            dtype=torch.float32,
        )
        profiler.end("torch_input_creation")
        logger.debug(f"Input shape: {dispatched_buffer_torch.shape} (mesh: {mesh_device.shape})")

        # Create weights for all experts
        profiler.start("torch_expert_weights_creation")
        torch_weights_list = []
        for i in range(total_experts):
            weights = {
                "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
            }
            torch_weights_list.append(weights)
        profiler.end("torch_expert_weights_creation")

        # Create torch experts for PCC validation
        profiler.start("torch_expert_creation")
        torch_experts = [TorchExpert(emb_dim, torch_weights=w) for w in torch_weights_list]
        profiler.end("torch_expert_creation")

        # Run torch reference
        profiler.start("torch_forward")
        logger.debug("Running torch reference...")
        torch_outputs = run_torch_routed_experts(
            dispatched_buffer_torch,
            torch_experts,
            experts_per_chip,
            num_dispatch_groups,
            dispatch_group_size,
        )
        profiler.end("torch_forward")
        logger.debug(f"Torch output shape: {torch_outputs.shape}")
        logger.debug(f"Torch output stats - min: {torch_outputs.min():.4f}, max: {torch_outputs.max():.4f}")

        # Create TTNN input from torch - shard across devices using dims=(1, 0)
        mesh_mapper = get_routed_expert_buffer_mesh_mapper(mesh_device)
        profiler.start("input_to_device")
        dispatched_buffer_tt = ttnn.from_torch(
            dispatched_buffer_torch,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
        )
        dispatched_buffer_tt = ttnn.reshape(dispatched_buffer_tt, per_device_shape)
        ttnn.synchronize_device(mesh_device)
        profiler.end("input_to_device")
    else:
        # Fast path: allocate directly on device (uninitialized DRAM)
        profiler.start("input_to_device")
        dispatched_buffer_tt = ttnn.empty(
            per_device_shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        ttnn.synchronize_device(mesh_device)
        profiler.end("input_to_device")

    logger.debug(f"TTNN input shape: {dispatched_buffer_tt.shape}")

    # Create TtRoutedExpert
    # When torch_weights=None, uses _create_random_weight for fast device-side DRAM allocation
    # When torch_weights provided, distributes weights to devices (device i gets experts [i*N:(i+1)*N])
    logger.debug("Creating TtRoutedExpert...")
    profiler.start("tt_routed_expert_creation")
    tt_routed_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=torch_weights_list,  # None when run_pcc_check=False
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_routed_expert_creation")

    # Run TTNN forward
    logger.debug("Running TTNN forward...")
    profiler.start("tt_forward")
    ttnn_outputs = tt_routed_expert(dispatched_buffer_tt)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.debug(f"TTNN output shape: {ttnn_outputs.shape}")

    if not run_pcc_check:
        logger.debug("run_pcc_check=False, skipping torch validation")
        for key in profiler.times:
            logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
        return

    # Convert back to torch and validate PCC
    profiler.start("pcc_validation")
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
            torch_chip = torch_outputs[dg, ds, :, :, :]  # (experts_per_chip, max_dispatched_tokens_per_expert, emb_dim)
            ttnn_chip = ttnn_outputs_torch[dg, ds, :, :, :]

            for expert_idx in range(experts_per_chip):
                global_expert_idx = ExpertMapping.get_global_expert_idx(
                    group=dg,
                    chip=ds,
                    local_expert=expert_idx,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    is_col_major=True,
                )
                _, pcc = comp_pcc(torch_chip[expert_idx], ttnn_chip[expert_idx])
                pcc_values.append(pcc)
                logger.debug(
                    f"Chip (dg={dg},ds={ds}), Local Expert {expert_idx} (Global {global_expert_idx}) PCC: {pcc:.6f}"
                )

    min_pcc = min(pcc_values)
    avg_pcc = sum(pcc_values) / len(pcc_values)
    profiler.end("pcc_validation")
    logger.debug(f"\nMin PCC: {min_pcc:.6f}, Avg PCC: {avg_pcc:.6f} (across all {len(pcc_values)} experts)")

    # Threshold for bfp8/bfp4 precision (actual PCC ~0.98)
    pcc_threshold = 0.97
    assert min_pcc >= pcc_threshold, f"PCC {min_pcc:.6f} below threshold {pcc_threshold}"

    # Verify no NaN/Inf
    assert not torch.isnan(ttnn_outputs_torch).any(), "Output contains NaN"
    assert not torch.isinf(ttnn_outputs_torch).any(), "Output contains Inf"

    # Print timing summary
    profiler.end("test_ttnn_routed_expert")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")

    logger.debug("TtRoutedExpert PCC Test PASSED!")
