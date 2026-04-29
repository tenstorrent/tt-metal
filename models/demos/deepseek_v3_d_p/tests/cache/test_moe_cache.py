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
    compute_constants,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_moe")


@pytest.fixture(autouse=True)
def cleanup_cache():
    """Clean cache before each test."""
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
@pytest.mark.parametrize(
    "gate_mode",
    [GateComputeMode.DEVICE, GateComputeMode.HOST_ALL],
    ids=["device_gate", "host_gate"],
)
def test_moe_weights_cold_warm_cache(mesh_device, device_params, gate_mode):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    torch.manual_seed(42)

    logger.info(f"Testing MoE cache with gate_mode={gate_mode}")

    # Use reduced parameters for faster execution
    seq_len_per_chip = 320
    emb_dim = 1024
    hidden_dim = 512
    num_routed_experts = 256  # Required by gate kernel
    num_experts_per_tok = 8  # Required by gate kernel
    # ceil(N/2) of the most conservative integer N such that dgs*seq*N >= theoretical
    # worst-case dispatch buffer. Real traffic never approaches the worst case.
    dispatch_buffer_capacity_factor = 6

    # Compute constants
    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

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

    logger.info(f"Test params: experts_per_chip={experts_per_chip}, max_tokens={max_dispatched_tokens_per_expert}")
    logger.info(f"Dimensions: emb_dim={emb_dim}, hidden_dim={hidden_dim}, total_experts={total_experts}")

    # Create weights
    gate_weights = create_gate_weights(num_routed_experts, emb_dim)
    routed_expert_weights = create_torch_expert_weights(total_experts, emb_dim, hidden_dim)
    shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim)

    # Create input tensor (will be recreated for each path to avoid state pollution)
    x_torch = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)

    def create_input():
        """Create fresh input tensor for each path."""
        return ttnn.from_torch(
            x_torch,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)  # SP on axis 0, TP on axis 1
            ),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )

    # Helper to convert TP-sharded output to torch
    tp_composer = get_tp_mesh_composer(mesh_device)

    def to_torch_tp(tt_tensor):
        """Convert TP-sharded MoE output back to torch."""
        return ttnn.to_torch(tt_tensor, mesh_composer=tp_composer)

    # === Path 1: From Weights ===
    logger.info("Path 1: Creating TtMoe from weights...")
    moe_from_weights = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        gate_weights=gate_weights,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_links=1,
        topology=ttnn.Topology.Linear,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat16,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat16,
        gate_fallback_mode=gate_mode,
        weight_cache_path=None,
        layer_idx=0,
    )
    output1_tt, _ = moe_from_weights(create_input(), return_intermediates=False)
    output1 = to_torch_tp(output1_tt)
    logger.info(f"Output1 shape: {output1.shape}, min={output1.min():.4f}, max={output1.max():.4f}")

    # Clean up to avoid device state pollution
    del moe_from_weights
    ttnn.synchronize_device(mesh_device)

    # === Path 2: Cold Cache (build + load) ===
    init_checker(CACHE_DIR)
    assert not TtMoe.check_cache_complete(
        CACHE_DIR, layer_idx=0, experts_per_chip=experts_per_chip
    ), "Cache should be empty before build"

    logger.info(f"Building cache to {CACHE_DIR}...")
    profiler.clear()
    profiler.start("build_cache")
    TtMoe.build_ttnn_cache(
        gate_weights=gate_weights,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        experts_per_chip=experts_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        mesh_device=mesh_device,
        routed_expert_weights_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat16,
        cache_path=CACHE_DIR,
        layer_idx=0,
    )
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert TtMoe.check_cache_complete(
        CACHE_DIR, layer_idx=0, experts_per_chip=experts_per_chip
    ), "Cache should be complete after build"

    logger.info("Path 2: Creating TtMoe from cold cache...")
    profiler.start("cold_load")
    moe_cold = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        gate_weights=None,  # Cache-only mode
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_links=1,
        topology=ttnn.Topology.Linear,
        routed_expert_weights=None,
        shared_expert_weights=None,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat16,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat16,
        gate_fallback_mode=gate_mode,
        weight_cache_path=CACHE_DIR,
        layer_idx=0,
    )
    profiler.end("cold_load")
    output2_tt, _ = moe_cold(create_input(), return_intermediates=False)
    output2 = to_torch_tp(output2_tt)
    logger.info(f"Output2 shape: {output2.shape}, min={output2.min():.4f}, max={output2.max():.4f}")

    # Clean up to avoid device state pollution
    del moe_cold
    ttnn.synchronize_device(mesh_device)

    # === Path 3: Warm Cache (reuse existing cache) ===
    logger.info("Path 3: Creating TtMoe from warm cache...")
    profiler.start("warm_load")
    moe_warm = TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        gate_weights=None,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        num_links=1,
        topology=ttnn.Topology.Linear,
        routed_expert_weights=None,
        shared_expert_weights=None,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat16,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat16,
        gate_fallback_mode=gate_mode,
        weight_cache_path=CACHE_DIR,
        layer_idx=0,
    )
    profiler.end("warm_load")
    output3_tt, _ = moe_warm(create_input(), return_intermediates=False)
    output3 = to_torch_tp(output3_tt)
    logger.info(f"Output3 shape: {output3.shape}, min={output3.min():.4f}, max={output3.max():.4f}")

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"MoE Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    # MoE integration test should achieve high PCC like component tests
    # Using GateComputeMode.DEVICE ensures gate runs on device (not host fallback)
    # This avoids issues with uninitialized torch tensors in cache mode
    pcc_threshold = 0.99
    assert pcc_cold >= pcc_threshold, f"Cold cache mismatch: PCC={pcc_cold} < {pcc_threshold}"
    assert pcc_warm >= pcc_threshold, f"Warm cache mismatch: PCC={pcc_warm} < {pcc_threshold}"

    logger.info(f"✓ MoE cache test passed ({gate_mode}) with PCC cold={pcc_cold:.4f}, warm={pcc_warm:.4f}")
