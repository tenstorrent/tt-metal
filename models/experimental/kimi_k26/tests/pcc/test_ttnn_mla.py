# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for ttMLA configured with Kimi K2.6 hyperparameters.

Kimi K2.6 reuses the DeepSeek V3 MLA architecture (DeepseekV3Attention);
only hyperparameters differ (see KimiK26Config). Versus DSv3:
- num_attention_heads = 64 (vs 128)
- max_position_embeddings = 262144 (vs 4096)
- rope_theta = 50000.0 (vs 10000.0)
- rope_scaling = YaRN factor=64, original_max=4096 (vs None)
- rms_norm_eps = 1e-5 (vs 1e-6)

q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
hidden_size are inherited from DSv3 unchanged.
"""

import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.experimental.kimi_k26.reference.configuration_deepseek import DeepseekV3Config
from models.experimental.kimi_k26.kimi_k26_config import KimiK26Config
from models.experimental.kimi_k26.reference.modeling_deepseek import (
    DeepseekV3Attention as KimiBundledAttention,
)
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.utils_for_testing import assert_with_pcc


def _build_kimi_config() -> DeepseekV3Config:
    """HF DeepseekV3Config populated with Kimi K2.6 text-config values."""
    return DeepseekV3Config(
        vocab_size=KimiK26Config.VOCAB_SIZE,
        hidden_size=KimiK26Config.EMB_SIZE,
        intermediate_size=KimiK26Config.INTERMEDIATE_SIZE,
        moe_intermediate_size=KimiK26Config.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=KimiK26Config.NUM_LAYERS,
        num_attention_heads=KimiK26Config.NUM_ATTENTION_HEADS,
        num_key_value_heads=KimiK26Config.NUM_KEY_VALUE_HEADS,
        q_lora_rank=KimiK26Config.Q_LORA_RANK,
        kv_lora_rank=KimiK26Config.KV_LORA_RANK,
        qk_nope_head_dim=KimiK26Config.QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=KimiK26Config.QK_ROPE_HEAD_DIM,
        v_head_dim=KimiK26Config.V_HEAD_DIM,
        max_position_embeddings=KimiK26Config.MAX_POSITION_EMBEDDINGS,
        rope_theta=KimiK26Config.ROPE_THETA,
        rope_scaling={
            "type": "yarn",
            "factor": KimiK26Config.ROPE_SCALING_FACTOR,
            "original_max_position_embeddings": KimiK26Config.ROPE_SCALING_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": KimiK26Config.ROPE_SCALING_BETA_FAST,
            "beta_slow": KimiK26Config.ROPE_SCALING_BETA_SLOW,
            "mscale": KimiK26Config.ROPE_SCALING_MSCALE,
            "mscale_all_dim": KimiK26Config.ROPE_SCALING_MSCALE_ALL_DIM,
        },
        rms_norm_eps=KimiK26Config.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        first_k_dense_replace=KimiK26Config.NUM_DENSE_LAYERS,
        n_routed_experts=KimiK26Config.NUM_ROUTED_EXPERTS,
        n_shared_experts=KimiK26Config.NUM_SHARED_EXPERTS,
        num_experts_per_tok=KimiK26Config.NUM_EXPERTS_PER_TOKEN,
        n_group=KimiK26Config.NUM_EXPERT_GROUPS,
        topk_group=KimiK26Config.NUM_LIMITED_GROUPS,
        routed_scaling_factor=KimiK26Config.ROUTE_SCALE,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )


def _run_kimi_mla_cpu(
    config: DeepseekV3Config,
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Forward Kimi's bundled DeepseekV3Attention on CPU.
    Returns shape (1, q_len, hidden_size).
    """
    attn = KimiBundledAttention(config, layer_idx=0)
    attn.load_state_dict(weights, strict=False)
    attn = attn.eval().to(torch.bfloat16)

    _, q_len, _ = hidden_states.shape
    causal = torch.full((q_len, q_len), float("-inf"), dtype=hidden_states.dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, q_len)

    with torch.no_grad():
        out, _, _ = attn(
            hidden_states=hidden_states,
            attention_mask=causal,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
        )
    return out


def _generate_random_mla_weights(config: DeepseekV3Config) -> dict[str, torch.Tensor]:
    """Random state_dict matching DeepseekV3Attention's expected keys."""
    h = config.hidden_size
    n = config.num_attention_heads
    qlr = config.q_lora_rank
    kvlr = config.kv_lora_rank
    qk_nope = config.qk_nope_head_dim
    qk_rope = config.qk_rope_head_dim
    vh = config.v_head_dim
    q_head_dim = qk_nope + qk_rope
    std = config.initializer_range

    return {
        "q_a_proj.weight": (torch.randn(qlr, h) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(qlr, dtype=torch.bfloat16),
        "q_b_proj.weight": (torch.randn(n * q_head_dim, qlr) * std).to(torch.bfloat16),
        "kv_a_proj_with_mqa.weight": (torch.randn(kvlr + qk_rope, h) * std).to(torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(kvlr, dtype=torch.bfloat16),
        "kv_b_proj.weight": (torch.randn(n * (qk_nope + vh), kvlr) * std).to(torch.bfloat16),
        "o_proj.weight": (torch.randn(h, n * vh) * std).to(torch.bfloat16),
    }


@pytest.mark.parametrize(
    "seq_len_per_chip",
    [
        pytest.param(128, id="kimi-1k", marks=pytest.mark.timeout(0)),
        pytest.param(640, id="kimi-5k", marks=pytest.mark.timeout(0)),
        pytest.param(3200, id="kimi-25k", marks=pytest.mark.timeout(0)),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else 1344544,
            },
            ttnn.Topology.Linear,
            marks=[
                pytest.mark.skipif(not is_blackhole(), reason="Blackhole only"),
                pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            ],
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_kimi_k26_mla(mesh_device, device_params, seq_len_per_chip, topology):
    random.seed(42)
    torch.manual_seed(42)

    sp_axis = 0
    tp_axis = 1
    sp_factor = mesh_device.shape[sp_axis]
    seq_len = sp_factor * seq_len_per_chip
    mesh_shape = list(mesh_device.shape)

    config = _build_kimi_config()
    config.max_seq_len = seq_len

    logger.info(
        f"KimiK26 MLA PCC: mesh={tuple(mesh_device.shape)}, total_seq={seq_len} "
        f"(seq_len_per_chip={seq_len_per_chip}), num_heads={config.num_attention_heads}, "
        f"q_lora={config.q_lora_rank}, kv_lora={config.kv_lora_rank}"
    )

    weights = _generate_random_mla_weights(config)

    hidden_states = torch.randn(1, seq_len, config.hidden_size).to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    tt_mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope_setup.get_rope_tensors(seq_len)

    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.qk_rope_head_dim + config.kv_lora_rank,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # Shard hidden_states: SP on seq dim (-2), TP on hidden dim (-1)
    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_input = hidden_states.unsqueeze(0)  # [1, batch, seq, hidden]
    tt_hidden_states = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    tt_output = tt_mla.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    tt_output_cpu = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)

    logger.info(f"Running Kimi MLA reference at seq_len={seq_len}")
    kimi_out = _run_kimi_mla_cpu(config, weights, hidden_states, position_ids)
    _, kimi_pcc = assert_with_pcc(kimi_out.unsqueeze(0), tt_output_cpu, 0.995)
    logger.info(f"[mla_output] PCC: {kimi_pcc}")
