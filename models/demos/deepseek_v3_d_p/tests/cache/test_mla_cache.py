# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_mla")


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
def test_mla_weights_cold_warm_cache(mesh_device, device_params, config_only):
    """Test: weights → cold cache → warm cache produce identical outputs."""
    config = config_only
    layer_idx = 0
    seq_len = 1024
    sp_axis = 0
    tp_axis = 1

    # Set max_seq_len on config (required by MLA)
    config.max_seq_len = seq_len

    mesh_shape = list(mesh_device.shape)

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

    # Initialize KVPE cache (required by MLA forward)
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

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
    output1_tt = mla_from_weights.forward(x_tt, rope_tensors, tt_kvpe_cache)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    assert not ttMLA.check_cache_complete(CACHE_DIR, f"layer_{layer_idx}.mla"), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
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
    profiler.end("build_cache")

    assert ttMLA.check_cache_complete(CACHE_DIR, f"layer_{layer_idx}.mla"), "Cache should be complete after build"

    profiler.start("cold_load")
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
    profiler.end("cold_load")
    output2_tt = mla_cold.forward(x_tt, rope_tensors, tt_kvpe_cache)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
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
    profiler.end("warm_load")
    output3_tt = mla_warm.forward(x_tt, rope_tensors, tt_kvpe_cache)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"MLA Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
