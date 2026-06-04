# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_routed_expert")


@pytest.fixture(autouse=True)
def cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    report_and_clear()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="linear"),
            id="linear-2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_routed_expert_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    torch.manual_seed(42)

    # Use realistic parameters
    seq_len_per_chip = 320
    emb_dim = 1024
    hidden_dim = 512
    num_routed_experts = 64
    num_experts_per_tok = 2
    # ceil(N/2) of the most conservative integer N such that dgs*seq*N >= theoretical
    # worst-case dispatch buffer. Real traffic never approaches the worst case.
    dispatch_buffer_capacity_factor = 2

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    # Compute constants (same as PCC test)
    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    total_experts = num_devices * experts_per_chip

    # Create random weights for all experts (HF format)
    torch_weights = []
    for _ in range(total_experts):
        torch_weights.append(
            {
                "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
                "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
            }
        )

    # Create input with proper sharding (follows PCC test pattern).
    # Flat 4D layout: (num_dispatch_groups, dispatch_group_size,
    # max_dispatch_buffer_token_size, emb_dim) — each chip's experts
    # are concatenated along the token dim, matching the real dispatch kernel layout.
    dispatched_buffer_torch = torch.randn(
        num_dispatch_groups,
        dispatch_group_size,
        max_dispatch_buffer_token_size,
        emb_dim,
        dtype=torch.float32,
    )

    # Shard across devices and reshape per-device to 2D (what extract/insert require).
    per_device_shape = (max_dispatch_buffer_token_size, emb_dim)
    mesh_mapper = get_ep_mesh_mapper(mesh_device)
    dispatched_buffer_tt = ttnn.from_torch(
        dispatched_buffer_torch,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    dispatched_buffer_tt = ttnn.reshape(dispatched_buffer_tt, per_device_shape)

    # Build (group, chip, local_expert) -> global expert id table, sharded across
    # the EP mesh so each device holds (1, 1, experts_per_chip). Then squeeze to a
    # 1D (experts_per_chip,) vector (required by extract/insert validators). Shared
    # across all TtRoutedExpert instances built below.
    global_expert_idx_tt = ttnn.from_torch(
        ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        ),
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)
    global_expert_idx_tt = ttnn.squeeze(global_expert_idx_tt, 0)

    # Build synthetic token counts / region offsets from random routing indices.
    # For a cache test we only need all three expert instances to see the same
    # counts/offsets — the specific values don't matter for the PCC comparison.
    _, _, routing_indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
        skip_x_initialization=True,
    )
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    _, expert_token_counts_torch, expert_region_offsets_torch, _ = get_gate_outputs(
        routing_indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )
    expert_token_counts_tt = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_token_counts_torch)
    expert_region_offsets_tt = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_region_offsets_torch)

    # Helper to convert output back to torch (follows PCC test pattern)
    mesh_composer = get_ep_mesh_composer(mesh_device)

    def to_torch_expert(tt_tensor):
        """Convert expert output back to torch with proper mesh composer."""
        # Output shape per device: (max_dispatch_buffer_token_size, emb_dim) — 2D.
        # Unsqueeze to 4D and compose back across the EP mesh.
        tt_expanded = ttnn.unsqueeze(ttnn.unsqueeze(tt_tensor, dim=0), dim=0)
        return ttnn.to_torch(tt_expanded, mesh_composer=mesh_composer)

    # Use consistent dtype across all paths
    weights_dtype = ttnn.bfloat4_b

    # === Path 1: From Weights ===
    logger.info(f"Test params: experts_per_chip={experts_per_chip}, max_tokens={max_dispatched_tokens_per_expert}")
    logger.info(f"Dimensions: emb_dim={emb_dim}, hidden_dim={hidden_dim}")
    logger.info(f"dispatched_buffer_tt.shape={dispatched_buffer_tt.shape}")

    expert_from_weights = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=torch_weights,
        weights_dtype=weights_dtype,
        weight_cache_path=None,
    )
    output1_tt = expert_from_weights(dispatched_buffer_tt, expert_token_counts_tt, expert_region_offsets_tt)
    output1 = to_torch_expert(output1_tt)

    # === Path 2: Cold Cache ===
    init_checker(CACHE_DIR)
    assert not TtRoutedExpert.check_cache_complete(
        CACHE_DIR, "routed_expert", experts_per_chip
    ), "Cache should be empty before build"

    logger.info(f"Building cache to {CACHE_DIR}")
    profiler.clear()
    profiler.start("build_cache")
    TtRoutedExpert.build_ttnn_cache(
        torch_weights,
        experts_per_chip,
        mesh_device,
        weights_dtype,
        CACHE_DIR,
        "routed_expert",
    )
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert TtRoutedExpert.check_cache_complete(
        CACHE_DIR, "routed_expert", experts_per_chip
    ), "Cache should be complete after build"

    profiler.start("cold_load")
    expert_cold = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="routed_expert",
    )
    profiler.end("cold_load")
    output2_tt = expert_cold(dispatched_buffer_tt, expert_token_counts_tt, expert_region_offsets_tt)
    output2 = to_torch_expert(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    expert_warm = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="routed_expert",
    )
    profiler.end("warm_load")
    output3_tt = expert_warm(dispatched_buffer_tt, expert_token_counts_tt, expert_region_offsets_tt)
    output3 = to_torch_expert(output3_tt)

    # === Validation ===
    # Debug: check output stats
    logger.info(
        f"Output1 (from weights): shape={output1.shape}, min={output1.min():.4f}, max={output1.max():.4f}, mean={output1.mean():.4f}"
    )
    logger.info(
        f"Output2 (cold cache): shape={output2.shape}, min={output2.min():.4f}, max={output2.max():.4f}, mean={output2.mean():.4f}"
    )
    logger.info(
        f"Output3 (warm cache): shape={output3.shape}, min={output3.min():.4f}, max={output3.max():.4f}, mean={output3.mean():.4f}"
    )

    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"Routed Expert Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
