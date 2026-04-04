# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TtPrefillTransformer — verifies composition of embed -> [block x N] -> norm.

Validates output shapes and PCC against torch reference.

Uses HF DeepseekV3Model as the reference: creates a small model with random weights,
extracts those weights into our TT state_dict format, and compares forward passes.

Parametrized over:
- tokenizer_enabled: real tokenized input vs random token IDs
- pcc_validation: per-stage PCC check (via return_intermediates) vs shape-only smoke test
- n_routed_experts / capacity_factor / gate_fallback_mode: MoE configurations
"""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model, DeepseekV3MoE
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from tests.ttnn.utils_for_testing import comp_pcc

PCC_THRESHOLD = 0.99


# --- HF model helpers ---


def _create_hf_model(config, num_layers, n_routed_experts=None):
    """Create HF DeepseekV3Model with num_layers and random weights."""
    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"
    if n_routed_experts is not None:
        test_config.n_routed_experts = n_routed_experts

    model = DeepseekV3Model(test_config)
    return model.eval().to(torch.bfloat16)


def _extract_layer_state_dict(full_sd, layer_idx, hf_layer):
    """Extract one layer's weights from HF state_dict into TtPrefillBlock format."""
    prefix = f"layers.{layer_idx}."
    is_moe = isinstance(hf_layer.mlp, DeepseekV3MoE)

    layer_sd = {
        "attn_norm_weight": full_sd[f"{prefix}input_layernorm.weight"],
        "mla_weights": {
            "q_a_proj.weight": full_sd[f"{prefix}self_attn.q_a_proj.weight"],
            "q_a_layernorm.weight": full_sd[f"{prefix}self_attn.q_a_layernorm.weight"],
            "q_b_proj.weight": full_sd[f"{prefix}self_attn.q_b_proj.weight"],
            "kv_a_proj_with_mqa.weight": full_sd[f"{prefix}self_attn.kv_a_proj_with_mqa.weight"],
            "kv_a_layernorm.weight": full_sd[f"{prefix}self_attn.kv_a_layernorm.weight"],
            "kv_b_proj.weight": full_sd[f"{prefix}self_attn.kv_b_proj.weight"],
            "o_proj.weight": full_sd[f"{prefix}self_attn.o_proj.weight"],
        },
        "ffn_norm_weight": full_sd[f"{prefix}post_attention_layernorm.weight"],
    }

    if is_moe:
        layer_sd["gate_weights"] = {
            "weight": full_sd[f"{prefix}mlp.gate.weight"],
            "e_score_correction_bias": full_sd[f"{prefix}mlp.gate.e_score_correction_bias"],
        }
        layer_sd["routed_expert_weights"] = [
            {
                "gate_proj": full_sd[f"{prefix}mlp.experts.{j}.gate_proj.weight"],
                "up_proj": full_sd[f"{prefix}mlp.experts.{j}.up_proj.weight"],
                "down_proj": full_sd[f"{prefix}mlp.experts.{j}.down_proj.weight"],
            }
            for j in range(len(hf_layer.mlp.experts))
        ]
        layer_sd["shared_expert_weights"] = {
            "gate_proj": full_sd[f"{prefix}mlp.shared_experts.gate_proj.weight"],
            "up_proj": full_sd[f"{prefix}mlp.shared_experts.up_proj.weight"],
            "down_proj": full_sd[f"{prefix}mlp.shared_experts.down_proj.weight"],
        }
    else:
        layer_sd["ffn_weights"] = {
            "gate_proj": full_sd[f"{prefix}mlp.gate_proj.weight"],
            "up_proj": full_sd[f"{prefix}mlp.up_proj.weight"],
            "down_proj": full_sd[f"{prefix}mlp.down_proj.weight"],
        }

    return layer_sd


def _extract_tt_state_dict(hf_model):
    """Extract state_dict in TtPrefillTransformer format from HF model."""
    sd = hf_model.state_dict()
    num_layers = len(hf_model.layers)

    result = {
        "embed_weight": sd["embed_tokens.weight"].float(),
        "norm_weight": sd["norm.weight"],
        "layers": [],
    }

    for i in range(num_layers):
        layer_sd = _extract_layer_state_dict(sd, i, hf_model.layers[i])
        result["layers"].append(layer_sd)

    return result


# --- Tokenization helpers ---

PROMPTS_PATH = Path("models/demos/deepseek_v3/demo/test_prompts_1024.json")


def _get_pad_id(tokenizer):
    """Resolve pad token ID, matching generator.py logic."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def _tokenize_prompts_to_isl(tokenizer, prompts_path, isl_total, sp_factor):
    """Tokenize prompts from JSON, concatenate, truncate/pad to isl_total."""
    prompts = load_prompts_from_json(str(prompts_path))
    pad_id = _get_pad_id(tokenizer)

    all_tokens = []
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        all_tokens.extend(ids)
        if len(all_tokens) >= isl_total:
            break

    # Truncate or pad
    if len(all_tokens) >= isl_total:
        all_tokens = all_tokens[:isl_total]
    else:
        all_tokens.extend([pad_id] * (isl_total - len(all_tokens)))

    # Alignment check
    isl_per_chip = isl_total // sp_factor
    assert (
        isl_per_chip % ttnn.TILE_SIZE == 0
    ), f"isl_per_chip={isl_per_chip} must be divisible by TILE_SIZE={ttnn.TILE_SIZE}"

    return torch.tensor(all_tokens, dtype=torch.int64).unsqueeze(0)  # [1, isl_total]


# --- Test ---


@pytest.mark.parametrize("tokenizer_enabled", [True, False], ids=["tokenized", "random"])
@pytest.mark.parametrize("pcc_validation", [True, False], ids=["pcc", "smoke"])
@pytest.mark.parametrize("isl_total", [1024])
@pytest.mark.parametrize("num_layers", [6])
@pytest.mark.parametrize(
    "n_routed_experts, capacity_factor, gate_fallback_mode",
    [
        (64, 4, GateComputeMode.HOST_ALL),
        (256, 32, GateComputeMode.HOST_ALL),
        (256, 32, GateComputeMode.DEVICE),
    ],
    ids=["e64_cf4_host", "e256_cf32_host", "e256_cf32_device"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_prefill_transformer(
    config_only,
    mesh_device,
    device_params,
    isl_total,
    num_layers,
    n_routed_experts,
    capacity_factor,
    gate_fallback_mode,
    num_links,
    topology,
    pcc_validation,
    tokenizer_enabled,
    request,
):
    torch.manual_seed(42)

    profiler.clear()
    profiler.start("total_test_time")
    config = config_only
    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}")
    logger.info(
        f"isl_total={isl_total}, isl_per_chip={isl_per_chip}, "
        f"num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
        f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode}, "
        f"tokenizer_enabled={tokenizer_enabled}, pcc_validation={pcc_validation}"
    )

    # --- Monkeypatch n_routed_experts ---
    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    # --- Build HF reference model and extract weights ---
    profiler.start("weights_creation")
    hf_model = _create_hf_model(config, num_layers, n_routed_experts=n_routed_experts)
    state_dict = _extract_tt_state_dict(hf_model)
    profiler.end("weights_creation")

    # --- Create input ---
    if tokenizer_enabled:
        profiler.start("tokenization")
        tok = request.getfixturevalue("tokenizer")
        token_ids = _tokenize_prompts_to_isl(tok, PROMPTS_PATH, isl_total, sp_factor)
        profiler.end("tokenization")
        logger.info(f"Tokenized input shape: {token_ids.shape}, first 10 tokens: {token_ids[0, :10].tolist()}")
    else:
        token_ids = torch.randint(0, config.vocab_size, (1, isl_total), dtype=torch.int64)

    # --- TT transformer ---
    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        capacity_factor=capacity_factor,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_transformer_creation")

    # --- Shard token_ids to device ---
    # Reshape [1, isl_total] -> [sp_factor, 1, isl_per_chip] for SP sharding
    token_ids_reshaped = token_ids.reshape(sp_factor, 1, isl_per_chip)

    tt_tokens = ttnn.from_torch(
        token_ids_reshaped,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    # --- Forward ---
    profiler.start("tt_forward")
    logger.info("Running TtPrefillTransformer forward...")
    result = transformer(tt_tokens, return_intermediates=pcc_validation)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    if pcc_validation:
        tt_output, tt_snapshots = result
    else:
        tt_output = result

    # --- Validate output shape ---
    expected_per_device_shape = [1, 1, isl_per_chip, emb_dim // tp_factor]
    output_shape = list(tt_output.shape)
    assert (
        output_shape == expected_per_device_shape
    ), f"Output shape mismatch: got {output_shape}, expected {expected_per_device_shape}"
    logger.info(f"Output shape: {output_shape} (matches expected)")

    # --- PCC check ---
    if pcc_validation:
        profiler.start("pcc_validation")

        # HF step-by-step forward for per-stage comparison
        seq_len = token_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

        with torch.no_grad():
            h_ref = hf_model.embed_tokens(token_ids).to(torch.bfloat16)
        ref_snapshots = [h_ref]

        for i in range(num_layers):
            with torch.no_grad():
                layer_out = hf_model.layers[i](h_ref, attention_mask=attention_mask, position_ids=position_ids)
                h_ref = layer_out[0]
            ref_snapshots.append(h_ref)

        with torch.no_grad():
            h_ref = hf_model.norm(h_ref)
        ref_snapshots.append(h_ref)

        # Per-stage PCC comparison
        pcc_results = []
        for (label, tt_host), ref_host in zip(tt_snapshots, ref_snapshots):
            try:
                _, pcc = comp_pcc(ref_host.float(), tt_host.float())
                logger.info(f"{label:<20s}  PCC = {pcc:.6f}")
                pcc_results.append((label, pcc))
            except Exception as e:
                logger.error(f"{label:<20s}  PCC comparison failed: {e}")
                pcc_results.append((label, -1.0))

        profiler.end("pcc_validation")

        # Summary table
        logger.info(f"\n{'='*50}")
        logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
        logger.info(f"{'-'*50}")
        failures = []
        for label, pcc in pcc_results:
            status = "PASS" if pcc > PCC_THRESHOLD else ("FAIL" if pcc >= 0 else "ERROR")
            logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
            if pcc <= PCC_THRESHOLD:
                failures.append((label, pcc))
        logger.info(f"{'='*50}")

        if failures:
            msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
            pytest.fail(f"PCC below {PCC_THRESHOLD} at: {msg}")

        logger.success(
            f"TtPrefillTransformer PCC test passed "
            f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
            f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode})"
        )
    else:
        logger.success(
            f"TtPrefillTransformer smoke test passed "
            f"(num_layers={num_layers}, n_routed_experts={n_routed_experts}, "
            f"capacity_factor={capacity_factor}, gate_fallback_mode={gate_fallback_mode})"
        )

    profiler.end("total_test_time")

    # --- Timing report ---
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    # Restore original config
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = orig_num_routed_experts
