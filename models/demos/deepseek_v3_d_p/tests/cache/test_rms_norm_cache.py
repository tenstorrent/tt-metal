# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch

import ttnn
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
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_rms_norm_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    torch.manual_seed(42)

    # Use realistic parameters (from PCC test)
    emb_dim = 7168
    isl_per_chip = 320  # Sequence length per chip
    epsilon = 1e-6

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
    TtDistributedRmsNorm.build_ttnn_cache(
        torch_weight,
        emb_dim,
        mesh_device,
        CACHE_DIR,
        "rms_norm",
    )

    norm_cold = TtDistributedRmsNorm(
        mesh_device,
        emb_dim,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="rms_norm",
    )
    output2_tt = norm_cold(x_tt)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    norm_warm = TtDistributedRmsNorm(
        mesh_device,
        emb_dim,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="rms_norm",
    )
    output3_tt = norm_warm(x_tt)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    from loguru import logger

    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"RMS Norm Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
