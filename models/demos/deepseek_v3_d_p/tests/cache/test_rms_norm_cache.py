# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_rms_norm")


@pytest.fixture(autouse=True)
def cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="linear"),
            id="linear-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_rms_norm_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    torch.manual_seed(42)

    # Use realistic parameters (from PCC test)
    emb_dim = 7168
    isl_per_chip = 320  # Sequence length per chip

    # Create random weight
    torch_weight = torch.randn(emb_dim, dtype=torch.float32)

    # Create 4D input (follows PCC test pattern)
    inp_shape_full = (1, 1, isl_per_chip, emb_dim)
    torch_input = torch.randn(inp_shape_full, dtype=torch.bfloat16).float()

    # Shard input across devices along width (dim=3, TP axis)
    x_tt = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 3)),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    # Helper to convert TP-sharded output to torch
    def to_torch_concat(tt_tensor):
        """Convert TP-sharded tensor to torch with mesh composer."""
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device,
                mesh_shape=mesh_device.shape,
                dims=(0, 3),  # Collapse dim 0, concat dim 3
            ),
        )

    # === Path 1: From Weights ===
    norm_from_weights = TtDistributedRmsNorm(
        mesh_device,
        emb_dim,
        torch_weight=torch_weight,
        weight_cache_path=None,
    )
    output1_tt = norm_from_weights(x_tt)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    assert not TtDistributedRmsNorm.check_cache_complete(CACHE_DIR, "rms_norm"), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
    TtDistributedRmsNorm.build_ttnn_cache(
        torch_weight,
        emb_dim,
        mesh_device,
        CACHE_DIR,
        "rms_norm",
    )
    profiler.end("build_cache")

    assert TtDistributedRmsNorm.check_cache_complete(CACHE_DIR, "rms_norm"), "Cache should be complete after build"

    profiler.start("cold_load")
    norm_cold = TtDistributedRmsNorm(
        mesh_device,
        emb_dim,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="rms_norm",
    )
    profiler.end("cold_load")
    output2_tt = norm_cold(x_tt)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    norm_warm = TtDistributedRmsNorm(
        mesh_device,
        emb_dim,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="rms_norm",
    )
    profiler.end("warm_load")
    output3_tt = norm_warm(x_tt)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"RMS Norm Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
