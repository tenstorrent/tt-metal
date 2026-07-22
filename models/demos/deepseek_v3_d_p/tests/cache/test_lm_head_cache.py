# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.tt_lm_head import TtLMHead
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import report_and_clear
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_lm_head")


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
def test_lm_head_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    vocab_size = DeepSeekV3Config.VOCAB_SIZE
    emb_dim = DeepSeekV3Config.EMB_SIZE
    seq_len = 256

    # Create random weight (single tensor, not dict)
    torch.manual_seed(42)
    torch_weight = torch.randn(vocab_size, emb_dim, dtype=torch.float32)

    # Create input
    dispatch_group_size = mesh_device.shape[0]  # SP factor
    torch_input = torch.randn(dispatch_group_size, seq_len, emb_dim, dtype=torch.bfloat16)
    input_tt = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
    )
    global_token_id = seq_len * dispatch_group_size - 1

    # Helper to convert output to torch
    def to_torch_concat(tt_tensor):
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
        )

    # === Path 1: From Weights ===
    lm_head_from_weights = TtLMHead(
        mesh_device,
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        torch_weight=torch_weight,
        weight_cache_path=None,
    )
    output1, _ = lm_head_from_weights(input_tt, global_token_id)
    output1 = to_torch_concat(output1)

    # === Path 2: Cold Cache ===
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    init_checker(CACHE_DIR)
    assert not TtLMHead.check_cache_complete(CACHE_DIR), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
    TtLMHead.build_ttnn_cache(torch_weight, vocab_size, emb_dim, mesh_device, CACHE_DIR)
    profiler.end("build_cache")

    init_checker(CACHE_DIR)  # Re-init checker after cache build
    assert TtLMHead.check_cache_complete(CACHE_DIR), "Cache should be complete after build"

    profiler.start("cold_load")
    lm_head_cold = TtLMHead(
        mesh_device,
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
    )
    profiler.end("cold_load")
    output2, _ = lm_head_cold(input_tt, global_token_id)
    output2 = to_torch_concat(output2)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    lm_head_warm = TtLMHead(
        mesh_device,
        emb_dim=emb_dim,
        vocab_size=vocab_size,
        torch_weight=None,
        weight_cache_path=CACHE_DIR,
    )
    profiler.end("warm_load")
    output3, _ = lm_head_warm(input_tt, global_token_id)
    output3 = to_torch_concat(output3)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"LM Head Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")

    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
