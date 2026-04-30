# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, create_gate_weights, get_sp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from models.demos.deepseek_v3_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_gate")


@pytest.fixture(autouse=True)
def cleanup_cache():
    """Clean cache before each test."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    report_and_clear()


def create_gate_input(config, mesh_device):
    """Create TP+SP sharded input tensor for gate (follows PCC test pattern)."""
    n_sp_devices = mesh_device.shape[0]
    torch_input = torch.randn(config.sp_dim * n_sp_devices, config.dim, dtype=torch.bfloat16) + 1
    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)
    return ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, -1),  # SP on axis 0, TP on axis 1
            mesh_shape=mesh_device.shape,
        ),
    )


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
    [GateComputeMode.DEVICE, GateComputeMode.HOST_MATMUL, GateComputeMode.HOST_GROUPED_GATE, GateComputeMode.HOST_ALL],
    ids=["device_gate", "host_matmul", "host_grouped_gate", "host_all"],
)
def test_gate_weights_cold_warm_cache(mesh_device, device_params, gate_mode):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    torch.manual_seed(42)

    logger.info(f"Testing gate cache with gate_mode={gate_mode}")

    # Create config and adjust for mesh size
    config = TtMoEGateConfig()
    adjust_shapes_for_testing(config, mesh_device)

    # Create gate weights using helper
    gate_weights_dict = create_gate_weights(config.n_routed_experts, config.dim)
    gate_w = gate_weights_dict["weight"]
    gate_b = gate_weights_dict["e_score_correction_bias"]

    # Create dispatch table
    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]
    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=config.n_routed_experts,
        dispatch_group_size=n_sp_devices,
        num_dispatch_groups=n_tp_devices,
    )

    # Create input (SP+TP sharded)
    x = create_gate_input(config, mesh_device)

    # Helper to convert gate output to torch (use SP composer like PCC test)
    sp_composer = get_sp_mesh_composer(mesh_device)

    def to_torch_gate(tt_tensor):
        """Gate scores are SP-sharded, TP-replicated after all_reduce."""
        return ttnn.to_torch(tt_tensor, mesh_composer=sp_composer)

    # === Path 1: From Weights ===
    experts_per_chip = config.n_routed_experts // (n_sp_devices * n_tp_devices)
    gate_from_weights = TtMoEGatePrefill(
        config,
        mesh_device,
        dispatch_table,
        experts_per_chip=experts_per_chip,
        weight=gate_w,
        bias=gate_b,
        fallback_mode=gate_mode,
        weight_cache_path=None,  # No caching
    )
    scores1, indices1, logits1, offsets1, counts1, regions1 = gate_from_weights(x)
    output1 = to_torch_gate(scores1)

    # === Path 2: Cold Cache (build + load) ===
    init_checker(CACHE_DIR)
    assert not TtMoEGatePrefill.check_cache_complete(CACHE_DIR, "gate"), "Cache should be empty before build"

    # Build cache
    logger.info(f"Building cache to {CACHE_DIR}")
    profiler.clear()
    profiler.start("build_cache")
    try:
        TtMoEGatePrefill.build_ttnn_cache(
            torch_weight=gate_w,
            torch_bias=gate_b,
            config=config,
            mesh_device=mesh_device,
            cache_path=CACHE_DIR,
            cache_name_prefix="gate",
        )
        logger.info("Cache built successfully")
    except Exception as e:
        logger.error(f"Cache building failed: {e}")
        raise
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert TtMoEGatePrefill.check_cache_complete(CACHE_DIR, "gate"), "Cache should be complete after build"

    # Load from cold cache
    profiler.start("cold_load")
    gate_cold = TtMoEGatePrefill(
        config,
        mesh_device,
        dispatch_table,
        experts_per_chip=experts_per_chip,
        weight=None,
        bias=None,  # Cache-only mode
        fallback_mode=gate_mode,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="gate",
    )
    profiler.end("cold_load")
    scores2, indices2, logits2, offsets2, counts2, regions2 = gate_cold(x)
    output2 = to_torch_gate(scores2)

    # === Path 3: Warm Cache (reuse existing cache) ===
    profiler.start("warm_load")
    gate_warm = TtMoEGatePrefill(
        config,
        mesh_device,
        dispatch_table,
        experts_per_chip=experts_per_chip,
        weight=None,
        bias=None,
        fallback_mode=gate_mode,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="gate",
    )
    profiler.end("warm_load")
    scores3, indices3, logits3, offsets3, counts3, regions3 = gate_warm(x)
    output3 = to_torch_gate(scores3)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"Gate Cache Test ({gate_mode}):")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch ({gate_mode}): PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch ({gate_mode}): PCC={pcc_warm}"
