# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_mla")


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
def test_mla_weights_cold_warm_cache(mesh_device, device_params, config_only):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    config = config_only
    layer_idx = 0
    seq_len = 1024
    sp_axis = 0
    tp_axis = 1

    # Set max_seq_len on config (required by MLA)
    config.max_seq_len = seq_len

    # Create random weights matching MLA architecture
    torch.manual_seed(42)
    std = config.initializer_range

    state_dict = {
        "q_a_proj.weight": (torch.randn(config.q_lora_rank, config.hidden_size) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_proj_with_mqa.weight": (
            torch.randn(
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "o_proj.weight": (
            torch.randn(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            )
            * std
        ).to(torch.bfloat16),
    }

    # Create input (full tensor - mesh_mapper will shard automatically)
    # Following pattern from test_mla.py: create full tensor, let TTNN shard it
    x = torch.randn(1, 1, seq_len, config.hidden_size, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(2, 3),  # Shard dim 2 (seq_len) on SP, dim 3 (hidden_size) on TP
            mesh_shape=mesh_device.shape,
        ),
    )

    # Create RoPE tensors
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Helper to convert TP-sharded output to torch
    def to_torch_concat(tt_tensor):
        """Convert TP-sharded 4D tensor to torch."""
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape  # Concat SP and TP dims
            ),
        )

    # === Path 1: From Weights ===
    mla_from_weights = ttMLA(
        config,
        state_dict,
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=None,
    )
    output1_tt = mla_from_weights.forward(x_tt, rope_tensors)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    ttMLA.build_ttnn_cache(
        state_dict,
        CACHE_DIR,
        mesh_device,
        config,
        layer_idx,
        seq_len,
        sp_axis,
        tp_axis,
    )

    mla_cold = ttMLA(
        config,
        {},
        mesh_device,  # Empty state_dict loads from cache
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=CACHE_DIR,
    )
    output2_tt = mla_cold.forward(x_tt, rope_tensors)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    mla_warm = ttMLA(
        config,
        {},
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=CACHE_DIR,
    )
    output3_tt = mla_warm.forward(x_tt, rope_tensors)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    from loguru import logger

    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"MLA Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
