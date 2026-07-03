# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.1 prefill *block* tests (one fused decoder layer: sparse-SDPA MLA + norm/residual + FFN).

Every GLM layer runs sparse DSA (lightning-indexer top-2048 + sparse SDPA); "dense"/"moe" here refers
only to the FFN — layers 0-2 have a dense FFN, layers 3-77 a 256-expert MoE. Both block types exercise
sparse SDPA (via ttMLA's DSA path); they differ only in the FFN.

GLM has no HF reference model wired (see reference/glm_5_1_config.py), so — instead of the DeepSeek/Kimi
block test's single HF decoder-layer forward — this composes the CPU references GLM already owns
(reference.glm_5_1.glm_decoder_layer_reference): x + MLA_cpu(attn_norm(x)) + FFN(ffn_norm(x+mla_out)),
which is exactly TtPrefillBlock.forward.

Weights, wired like the Kimi block test: pretrained when available (located via the variant env vars
GLM51_HF_MODEL / TT_GLM51_PREFILL_TTNN_CACHE — /mnt/MLPerf in CI, bf16 so no FP8 dequant), else random
(runs on any box). use_pretrained = supports_pretrained and weight_cache_path is not None.
"""

import json
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_1 import glm_decoder_layer_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import load_moe_weights_from_hf
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import build_weights
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import GateComputeMode, TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import assert_with_pcc

BLOCK_OUTPUT_PCC = 0.98


def _norm_weight(hidden, seed):
    return (torch.randn(hidden, generator=torch.Generator().manual_seed(seed)) * 0.1 + 1.0).to(torch.bfloat16)


def _random_dense_ffn_weights(hidden, intermediate, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate_proj": (torch.randn(intermediate, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "up_proj": (torch.randn(intermediate, hidden, generator=g) * hidden**-0.5).to(torch.bfloat16),
        "down_proj": (torch.randn(hidden, intermediate, generator=g) * intermediate**-0.5).to(torch.bfloat16),
    }


def _random_moe_weights(hidden, moe_intermediate, n_routed, seed):
    g = torch.Generator().manual_seed(seed)
    hs, ds = hidden**-0.5, moe_intermediate**-0.5

    def _expert():
        return {
            "gate_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(moe_intermediate, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, moe_intermediate, generator=g) * ds).to(torch.bfloat16),
        }

    gate_weights = {
        "weight": (torch.randn(n_routed, hidden, generator=g) * hs).to(torch.bfloat16),
        # noaux_tc selection bias — fp32 so the reference and the DEVICE_FP32 device gate agree.
        "e_score_correction_bias": (torch.randn(n_routed, generator=g) * 0.01).to(torch.float32),
    }
    return gate_weights, [_expert() for _ in range(n_routed)], _expert()


def _pretrained_weights(config, model_dir, layer_idx, is_moe):
    """Load layer-``layer_idx`` weights from a local checkpoint dir (bf16 GLM-5.1: no FP8 dequant)."""
    model_dir = str(model_dir)
    prefix = f"model.layers.{layer_idx}."
    # MLA + indexer via the sparse-MLA loader (resolve this layer's local shards; dequant is a no-op for bf16).
    weight_map = json.load(open(Path(model_dir) / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    mla_weights = pretrained_mla_weights(
        config, layer=layer_idx, checkpoint_path=[str(Path(model_dir) / s) for s in shards]
    )
    norms = load_hf_state_dict_filtered(model_dir, [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."])
    attn_norm_w = norms[f"{prefix}input_layernorm.weight"].to(torch.bfloat16)
    ffn_norm_w = norms[f"{prefix}post_attention_layernorm.weight"].to(torch.bfloat16)
    if is_moe:
        routed, shared = load_moe_weights_from_hf(model_dir, layer_idx, GLM51Config.NUM_ROUTED_EXPERTS)
        g = load_hf_state_dict_filtered(model_dir, [f"{prefix}mlp.gate."])
        gate_weights = {
            "weight": g[f"{prefix}mlp.gate.weight"].to(torch.bfloat16),
            "e_score_correction_bias": g[f"{prefix}mlp.gate.e_score_correction_bias"].float(),
        }
        return (
            mla_weights,
            attn_norm_w,
            ffn_norm_w,
            {
                "gate_weights": gate_weights,
                "routed_expert_weights": routed,
                "shared_expert_weights": shared,
            },
            None,
        )
    f = load_hf_state_dict_filtered(
        model_dir, [f"{prefix}mlp.gate_proj.", f"{prefix}mlp.up_proj.", f"{prefix}mlp.down_proj."]
    )
    ffn_weights = {p: f[f"{prefix}mlp.{p}.weight"].to(torch.bfloat16) for p in ("gate_proj", "up_proj", "down_proj")}
    return mla_weights, attn_norm_w, ffn_norm_w, None, ffn_weights


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [5120], ids=["seq5120"])
@pytest.mark.parametrize(
    "layer_type, layer_idx", [("dense", 0), ("moe", GLM51Config.NUM_DENSE_LAYERS)], ids=["dense", "moe"]
)
@pytest.mark.parametrize("variant", ["glm_5_1"], indirect=True, ids=["glm"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_block(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    topology,
    seq_len,
    layer_type,
    layer_idx,
    model_path,
    weight_cache_path,
):
    """One fused GLM decoder block (sparse-SDPA MLA + norm/residual + dense|MoE FFN) vs composed CPU ref."""
    is_moe = layer_type == "moe"
    config = config_only
    config.max_seq_len = seq_len
    hidden = config.hidden_size
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)

    # Weight source (wired like the transformer tests): use the prebuilt ttnn tensorbin cache when it is
    # present AND complete -> real weights (device LOADS from the cache; reference uses matching host
    # weights from the checkpoint). Otherwise fall back to RANDOM weights for both. We never (re)build the
    # cache here — that is slow (256-expert conversion / FP8 dequant) and is a separate staging step.
    sp_factor, tp_factor = mesh_shape[sp_axis], mesh_shape[tp_axis]
    experts_per_chip = GLM51Config.NUM_ROUTED_EXPERTS // (sp_factor * tp_factor)
    effective_cache = (weight_cache_path / f"{sp_factor}x{tp_factor}") if weight_cache_path is not None else None
    use_pretrained = False
    if effective_cache is not None:
        effective_cache.mkdir(parents=True, exist_ok=True)
        init_checker(effective_cache)  # required before check_cache_complete / pattern_exists
        use_pretrained = TtPrefillBlock.check_cache_complete(effective_cache, layer_idx, not is_moe, experts_per_chip)
    logger.info(f"[glm block {layer_type}] use_pretrained={use_pretrained} (ttnn cache={effective_cache})")

    if use_pretrained:
        # Device loads real weights from the cache; reference uses matching host weights from the checkpoint.
        mla_weights, attn_norm_w, ffn_norm_w, moe_weights, ffn_weights = _pretrained_weights(
            config, model_path, layer_idx, is_moe
        )
        device_state_dict, device_cache = {}, effective_cache
    else:
        # Random weights for both device (built from host) and reference.
        mla_weights, _ = build_weights(variant, config, seed=42)
        attn_norm_w, ffn_norm_w = _norm_weight(hidden, 1), _norm_weight(hidden, 2)
        if is_moe:
            gate_weights, routed, shared = _random_moe_weights(
                hidden, GLM51Config.MOE_INTERMEDIATE_SIZE, GLM51Config.NUM_ROUTED_EXPERTS, seed=3
            )
            moe_weights = {
                "gate_weights": gate_weights,
                "routed_expert_weights": routed,
                "shared_expert_weights": shared,
            }
            ffn_weights = None
        else:
            ffn_weights = _random_dense_ffn_weights(hidden, config.intermediate_size, seed=3)
            moe_weights = None
        device_state_dict = {"attn_norm_weight": attn_norm_w, "mla_weights": mla_weights, "ffn_norm_weight": ffn_norm_w}
        if is_moe:
            device_state_dict.update(
                gate_weights=moe_weights["gate_weights"],
                routed_expert_weights=moe_weights["routed_expert_weights"],
                shared_expert_weights=moe_weights["shared_expert_weights"],
            )
        else:
            device_state_dict["ffn_weights"] = ffn_weights
        device_cache = None

    # --- device block ---
    logger.info(
        f"[glm block {layer_type}] building TtPrefillBlock layer_idx={layer_idx} seq_len={seq_len} mesh={mesh_shape}"
    )
    block = TtPrefillBlock(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM51Config,
        state_dict=device_state_dict,
        layer_idx=layer_idx,
        seq_len=seq_len,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=device_cache,
    )
    kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors(seq_len)

    # --- input (full, host) + sharded device copy ---
    torch.manual_seed(7)
    x = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_x = ttnn.from_torch(
        x.unsqueeze(0),  # [1, 1, seq, hidden]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )

    logger.info(f"[glm block {layer_type}] running device block")
    out = block.forward(tt_x, rope_tensors=rope_tensors, kvpe_cache=kvpe_cache, actual_isl=seq_len)
    if isinstance(out, tuple):
        out = out[0]
    tt_out = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)

    # --- composed reference (reference/glm_5_1): assembles MLA + norm/residual + FFN ---
    logger.info(f"[glm block {layer_type}] composing CPU reference via reference.glm_5_1.glm_decoder_layer_reference")
    if is_moe:
        ref, _ = glm_decoder_layer_reference(
            config, mla_weights, attn_norm_w, ffn_norm_w, x, seq_len, moe_weights=moe_weights
        )
    else:
        ref, _ = glm_decoder_layer_reference(
            config, mla_weights, attn_norm_w, ffn_norm_w, x, seq_len, ffn_weights=ffn_weights
        )

    _, pcc_msg = assert_with_pcc(ref.unsqueeze(0), tt_out, BLOCK_OUTPUT_PCC)
    logger.info(f"[glm block {layer_type}] block output PCC: {pcc_msg}")
    ttnn.synchronize_device(mesh_device)
