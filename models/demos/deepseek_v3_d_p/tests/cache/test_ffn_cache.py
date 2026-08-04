# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_ffn")


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
def test_ffn_weights_cold_warm_cache(mesh_device, device_params):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    emb_dim = 7168
    hidden_dim = 18432
    batch, seq_len = 1, 256

    # Create random weights (HF format)
    torch_weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32),
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32),
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32),
    }

    # Create input (replicated full emb_dim)
    x = torch.randn(batch, 1, seq_len, emb_dim, dtype=torch.float32)
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
        # FFN output is 4D: [1, 1, seq_len, emb_dim/tp]
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape  # For 4D tensor: concat along last dim for TP
            ),
        )

    # Use consistent dtype across all paths
    weights_dtype = ttnn.bfloat8_b

    # === Path 1: From Weights ===
    ffn_from_weights = TtFfn(
        mesh_device,
        torch_weights=torch_weights,
        weights_dtype=weights_dtype,
        weight_cache_path=None,
    )
    output1_tt = ffn_from_weights(x_tt)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    init_checker(CACHE_DIR)
    assert not TtFfn.check_cache_complete(CACHE_DIR, "ffn"), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
    TtFfn.build_ttnn_cache(
        torch_weights,
        mesh_device,
        CACHE_DIR,
        "ffn",
    )
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert TtFfn.check_cache_complete(CACHE_DIR, "ffn"), "Cache should be complete after build"

    profiler.start("cold_load")
    ffn_cold = TtFfn(
        mesh_device,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="ffn",
    )
    profiler.end("cold_load")
    output2_tt = ffn_cold(x_tt)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    ffn_warm = TtFfn(
        mesh_device,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="ffn",
    )
    profiler.end("warm_load")
    output3_tt = ffn_warm(x_tt)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"FFN Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
