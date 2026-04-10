# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ffn import TtFfn
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_ffn")


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
    TtFfn.build_ttnn_cache(
        torch_weights,
        mesh_device,
        CACHE_DIR,
        "ffn",
    )

    ffn_cold = TtFfn(
        mesh_device,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="ffn",
    )
    output2_tt = ffn_cold(x_tt)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    ffn_warm = TtFfn(
        mesh_device,
        torch_weights=None,
        weights_dtype=weights_dtype,
        weight_cache_path=CACHE_DIR,
        cache_name_prefix="ffn",
    )
    output3_tt = ffn_warm(x_tt)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    from loguru import logger

    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"FFN Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
