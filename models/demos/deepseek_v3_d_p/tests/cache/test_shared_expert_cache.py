# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TT_PREFILL_TRANSFORMER_L1_SMALL
from models.demos.deepseek_v3_d_p.tt.moe.tt_shared_expert import TtSharedExpert
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_shared_expert")


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
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": TT_PREFILL_TRANSFORMER_L1_SMALL},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="linear"),
            id="linear-2x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_shared_expert_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    emb_dim = 7168
    hidden_dim = 2048
    batch, seq_len = 1, 256

    # Create random weights (HF format: out_features, in_features)
    torch_weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32),
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32),
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32),
    }

    # Create input (replicated across TP axis)
    x = torch.randn(batch, seq_len, emb_dim, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Helper to convert TP-sharded output to torch
    def to_torch_concat(tt_tensor):
        """Convert TP-sharded tensor to torch with mesh composer."""
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(1, 2), mesh_shape=mesh_device.shape  # For 3D tensor: concat along last dim for TP
            ),
        )

    # Use consistent dtype across all paths
    weights_dtype = ttnn.bfloat8_b

    # === Path 1: From Weights ===
    expert_from_weights = TtSharedExpert(
        mesh_device,
        emb_dim,
        hidden_dim,
        torch_weights=torch_weights,
        weights_dtype=weights_dtype,
        weight_cache_path=None,
    )
    output1_tt = expert_from_weights(x_tt)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    init_checker(CACHE_DIR)
    assert not TtSharedExpert.check_cache_complete(CACHE_DIR, "shared_expert"), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
    TtSharedExpert.build_ttnn_cache(
        torch_weights,
        emb_dim,
        hidden_dim,
        mesh_device,
        weights_dtype,
        CACHE_DIR,
        "shared_expert",
    )
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert TtSharedExpert.check_cache_complete(CACHE_DIR, "shared_expert"), "Cache should be complete after build"

    profiler.start("cold_load")
    expert_cold = TtSharedExpert(
        mesh_device,
        emb_dim,
        hidden_dim,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="shared_expert",
    )
    profiler.end("cold_load")
    output2_tt = expert_cold(x_tt)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    expert_warm = TtSharedExpert(
        mesh_device,
        emb_dim,
        hidden_dim,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="shared_expert",
    )
    profiler.end("warm_load")
    output3_tt = expert_warm(x_tt)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"Shared Expert Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
