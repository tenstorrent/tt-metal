# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for iterative PCC divergence of a single TtPrefillBlock layer.

Feeds the output of one layer back as input for N iterations (using pretrained
weights and real tokenized input), measuring PCC between torch reference and
TT hardware at each step. Generates a PCC-vs-iteration plot per layer.

This is observational/diagnostic — no PCC threshold assertions.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from loguru import logger
from transformers import DynamicCache

import ttnn
from models.demos.deepseek_v3.demo.demo import load_prompts_from_json
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    PROMPTS_PATH,
    create_hf_model_with_weights,
    get_4d_causal_mask,
    tokenize_prompt_to_isl,
)
from tests.ttnn.utils_for_testing import comp_pcc

PLOT_DIR = "models/demos/deepseek_v3_d_p/tests"


@pytest.mark.parametrize(
    "gate_fallback_mode",
    [GateComputeMode.DEVICE, GateComputeMode.HOST_ALL],
    ids=["gate_device", "gate_host"],
)
@pytest.mark.parametrize(
    "layer_idx",
    [0, -1, -4, -5, -6, -7, -2, -12, -3, -13, 3, 8],
    ids=[
        "layer0",
        "uniform_experts",
        "zero_routed",
        "zero_all_experts",
        "zero_all_plus_gate",
        "zero_shared_real_routed",
        "colvar_experts_l3",
        "colvar_experts_l8",
        "rowvar_experts_l3",
        "rowvar_experts_l8",
        "layer3",
        "layer8",
    ],
)
@pytest.mark.parametrize("num_iters", [30])
@pytest.mark.parametrize("isl_total", [1024])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (1, 1),
            {},
            1,
            ttnn.Topology.Linear,
            id="mesh-1x1",
        ),
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
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_prefill_block_loop(
    mesh_device,
    device_params,
    isl_total,
    layer_idx,
    num_iters,
    gate_fallback_mode,
    num_links,
    topology,
    model_path,
    hf_config,
    state_dict,
    tokenizer,
):
    # --- Validate fixtures ---
    if hf_config is None:
        pytest.skip("HF config not available")
    if state_dict is None:
        pytest.skip("State dict not available (no pretrained weights)")

    config = hf_config
    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    first_k_dense = config.first_k_dense_replace  # 3
    n_routed = config.n_routed_experts  # 256
    # Synthetic expert modes:
    #   -1  = uniform experts (all 1/7168), donor layer 3
    #   -2  = column-varying experts (different val per dispatch group), donor layer 3
    #   -12 = column-varying experts, donor layer 8
    #   -3  = row-varying experts (different val per row within each group), donor layer 3
    #   -13 = row-varying experts, donor layer 8
    synthetic_experts = layer_idx < 0
    donor_map = {-1: 3, -4: 3, -5: 3, -6: 3, -7: 3, -2: 3, -12: 8, -3: 3, -13: 8}
    real_layer_idx = donor_map.get(layer_idx, layer_idx)
    is_dense = not synthetic_experts and real_layer_idx < first_k_dense
    if layer_idx == -1:
        layer_type = "uniform_MoE"
    elif layer_idx == -4:
        layer_type = "zero_MoE"
    elif layer_idx == -5:
        layer_type = "zero_all_MoE"
    elif layer_idx == -6:
        layer_type = "zero_all_plus_gate"
    elif layer_idx == -7:
        layer_type = "zero_shared_real_routed"
    elif layer_idx in (-2, -12):
        layer_type = f"colvar_MoE(donor={real_layer_idx})"
    elif layer_idx in (-3, -13):
        layer_type = f"rowvar_MoE(donor={real_layer_idx})"
    else:
        layer_type = "dense" if is_dense else "MoE"
    gate_mode_name = gate_fallback_mode.name.lower()

    # gate_fallback_mode is irrelevant for dense layers — skip duplicate runs
    if is_dense and gate_fallback_mode != GateComputeMode.DEVICE:
        pytest.skip("gate_fallback_mode only applies to MoE layers")
    # For uniform/zero experts, HOST=DEVICE (proven), skip HOST to save time
    if layer_idx in (-1, -4, -5, -6) and gate_fallback_mode != GateComputeMode.DEVICE:
        pytest.skip("uniform/zero experts: DEVICE=HOST (proven), skipping HOST")

    logger.info(
        f"=== Iterative PCC test: layer {layer_idx} ({layer_type}), "
        f"{num_iters} iterations, gate_mode={gate_mode_name} ==="
    )
    logger.info(f"mesh_shape={mesh_shape}, sp_factor={sp_factor}, tp_factor={tp_factor}, isl_total={isl_total}")

    # ------------------------------------------------------------------
    # 1. Load only embed_tokens + target layer weights (bypass full dequant)
    # ------------------------------------------------------------------
    logger.info(f"Loading embed_tokens weights...")
    embed_sd = sub_state_dict(state_dict, "model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, hf_config)
    embed_weight = embed_dequant["weight"].float()

    logger.info(f"Loading layer {real_layer_idx} weights...")
    layer_raw = sub_state_dict(state_dict, f"model.layers.{real_layer_idx}.")
    layer_dequant = dequantize_state_dict(layer_raw, hf_config)

    # --- Build TT state dict for TtPrefillBlock ---
    layer_sd = {
        "attn_norm_weight": layer_dequant["input_layernorm.weight"],
        "mla_weights": {
            "q_a_proj.weight": layer_dequant["self_attn.q_a_proj.weight"],
            "q_a_layernorm.weight": layer_dequant["self_attn.q_a_layernorm.weight"],
            "q_b_proj.weight": layer_dequant["self_attn.q_b_proj.weight"],
            "kv_a_proj_with_mqa.weight": layer_dequant["self_attn.kv_a_proj_with_mqa.weight"],
            "kv_a_layernorm.weight": layer_dequant["self_attn.kv_a_layernorm.weight"],
            "kv_b_proj.weight": layer_dequant["self_attn.kv_b_proj.weight"],
            "o_proj.weight": layer_dequant["self_attn.o_proj.weight"],
        },
        "ffn_norm_weight": layer_dequant["post_attention_layernorm.weight"],
    }

    synthetic_routed = False
    if is_dense:
        layer_sd["ffn_weights"] = {
            "gate_proj": layer_dequant["mlp.gate_proj.weight"],
            "up_proj": layer_dequant["mlp.up_proj.weight"],
            "down_proj": layer_dequant["mlp.down_proj.weight"],
        }
    else:
        if layer_idx == -6:
            # Zero gate weights to produce zero routing scores
            gate_shape = layer_dequant["mlp.gate.weight"].shape
            bias_shape = layer_dequant["mlp.gate.e_score_correction_bias"].shape
            layer_sd["gate_weights"] = {
                "weight": torch.zeros(gate_shape, dtype=torch.bfloat16),
                "e_score_correction_bias": torch.zeros(bias_shape, dtype=torch.bfloat16),
            }
            logger.info(f"Zeroed gate weights: weight={gate_shape}, bias={bias_shape}")
        else:
            layer_sd["gate_weights"] = {
                "weight": layer_dequant["mlp.gate.weight"],
                "e_score_correction_bias": layer_dequant["mlp.gate.e_score_correction_bias"],
            }

        synthetic_routed = synthetic_experts and layer_idx != -7
        if synthetic_routed:
            moe_hidden = config.moe_intermediate_size  # 2048
            if layer_idx in (-1, -4, -5, -6):
                # All 256 experts identical: constant value
                val = 0.0 if layer_idx in (-4, -5, -6) else 1.0 / emb_dim
                expert_vals = [val] * n_routed
                logger.info(f"Creating {n_routed} {'zero' if val == 0 else 'uniform'} expert weights (val={val})")
            elif layer_idx in (-2, -12):
                # Column-varying: each dispatch group (64 experts) gets a different constant
                # Group 0 (experts 0-63): 1/7168, Group 1 (64-127): -1/7168
                # Group 2 (128-191): 0, Group 3 (192-255): 1/2048
                group_vals = [1.0 / emb_dim, -1.0 / emb_dim, 0.0, 1.0 / moe_hidden]
                experts_per_group = n_routed // len(group_vals)  # 64
                expert_vals = []
                for gv in group_vals:
                    expert_vals.extend([gv] * experts_per_group)
                logger.info(
                    f"Creating {n_routed} column-varying expert weights: "
                    f"group vals={group_vals}, {experts_per_group} experts/group"
                )
            else:
                # Row-varying (-3, -13): within each dispatch group (64 experts),
                # row 0 (first 32 experts) gets val_a, row 1 (next 32) gets val_b.
                # All 4 groups use the same row pattern.
                val_a = 1.0 / emb_dim
                val_b = -1.0 / emb_dim
                experts_per_chip = n_routed // mesh_device.get_num_devices()  # 32
                experts_per_group = experts_per_chip * mesh_shape[sp_axis]  # 64
                num_groups = n_routed // experts_per_group  # 4
                expert_vals = []
                for _ in range(num_groups):
                    expert_vals.extend([val_a] * experts_per_chip)  # row 0: first 32
                    expert_vals.extend([val_b] * experts_per_chip)  # row 1: next 32
                logger.info(
                    f"Creating {n_routed} row-varying expert weights: "
                    f"row0={val_a}, row1={val_b}, {experts_per_chip} experts/chip, {num_groups} groups"
                )

            layer_sd["routed_expert_weights"] = []
            for j in range(n_routed):
                v = expert_vals[j]
                layer_sd["routed_expert_weights"].append(
                    {
                        "gate_proj": torch.full((moe_hidden, emb_dim), v, dtype=torch.bfloat16),
                        "up_proj": torch.full((moe_hidden, emb_dim), v, dtype=torch.bfloat16),
                        "down_proj": torch.full((emb_dim, moe_hidden), v, dtype=torch.bfloat16),
                    }
                )
        else:
            layer_sd["routed_expert_weights"] = [
                {
                    "gate_proj": layer_dequant[f"mlp.experts.{j}.gate_proj.weight"],
                    "up_proj": layer_dequant[f"mlp.experts.{j}.up_proj.weight"],
                    "down_proj": layer_dequant[f"mlp.experts.{j}.down_proj.weight"],
                }
                for j in range(n_routed)
            ]

        if layer_idx in (-5, -6, -7):
            # Zero shared expert weights
            shared_hidden = config.moe_intermediate_size
            layer_sd["shared_expert_weights"] = {
                "gate_proj": torch.zeros(shared_hidden, emb_dim, dtype=torch.bfloat16),
                "up_proj": torch.zeros(shared_hidden, emb_dim, dtype=torch.bfloat16),
                "down_proj": torch.zeros(emb_dim, shared_hidden, dtype=torch.bfloat16),
            }
            logger.info("Zeroed shared expert weights")
        else:
            layer_sd["shared_expert_weights"] = {
                "gate_proj": layer_dequant["mlp.shared_experts.gate_proj.weight"],
                "up_proj": layer_dequant["mlp.shared_experts.up_proj.weight"],
                "down_proj": layer_dequant["mlp.shared_experts.down_proj.weight"],
            }

    # --- Build HF model with only the target layer ---
    num_layers_hf = real_layer_idx + 1
    hf_sd = {"embed_tokens.weight": embed_weight}
    for k, v in layer_dequant.items():
        hf_sd[f"layers.{real_layer_idx}.{k}"] = v

    # For synthetic experts, override HF expert weights too
    if synthetic_routed:
        moe_hidden = config.moe_intermediate_size
        for j in range(n_routed):
            v = expert_vals[j]
            hf_sd[f"layers.{real_layer_idx}.mlp.experts.{j}.gate_proj.weight"] = torch.full(
                (moe_hidden, emb_dim), v, dtype=torch.bfloat16
            )
            hf_sd[f"layers.{real_layer_idx}.mlp.experts.{j}.up_proj.weight"] = torch.full(
                (moe_hidden, emb_dim), v, dtype=torch.bfloat16
            )
            hf_sd[f"layers.{real_layer_idx}.mlp.experts.{j}.down_proj.weight"] = torch.full(
                (emb_dim, moe_hidden), v, dtype=torch.bfloat16
            )
    if layer_idx in (-5, -6, -7):
        # Zero shared expert in HF model too
        moe_hidden = config.moe_intermediate_size
        for key in ("gate_proj", "up_proj"):
            hf_sd[f"layers.{real_layer_idx}.mlp.shared_experts.{key}.weight"] = torch.zeros(
                moe_hidden, emb_dim, dtype=torch.bfloat16
            )
        hf_sd[f"layers.{real_layer_idx}.mlp.shared_experts.down_proj.weight"] = torch.zeros(
            emb_dim, moe_hidden, dtype=torch.bfloat16
        )
    if layer_idx == -6:
        # Zero gate in HF model too
        gate_shape = layer_dequant["mlp.gate.weight"].shape
        bias_shape = layer_dequant["mlp.gate.e_score_correction_bias"].shape
        hf_sd[f"layers.{real_layer_idx}.mlp.gate.weight"] = torch.zeros(gate_shape, dtype=torch.bfloat16)
        hf_sd[f"layers.{real_layer_idx}.mlp.gate.e_score_correction_bias"] = torch.zeros(
            bias_shape, dtype=torch.bfloat16
        )
    if synthetic_experts:
        logger.info(f"Overrode HF model weights with synthetic values")

    logger.info(f"Creating HF model with {num_layers_hf} layers (only layer {real_layer_idx} has real weights)...")
    hf_model = create_hf_model_with_weights(config, num_layers_hf, hf_sd)

    # ------------------------------------------------------------------
    # 2. Tokenize & embed (shared initial input)
    # ------------------------------------------------------------------
    prompts = load_prompts_from_json(str(PROMPTS_PATH))
    prompt_text = prompts[0] if isinstance(prompts, list) else prompts
    token_ids, attention_mask, tokens = tokenize_prompt_to_isl(tokenizer, max_isl=isl_total, prompt_text=prompt_text)
    attention_mask = get_4d_causal_mask(attention_mask, causal_only=True)

    logger.info(f"Token IDs shape: {token_ids.shape}, first 10: {token_ids[0, :10].tolist()}")

    with torch.no_grad():
        h0 = hf_model.embed_tokens(token_ids).to(torch.bfloat16)  # [1, 1024, 7168]
    logger.info(f"Initial embedding shape: {h0.shape}")

    # ------------------------------------------------------------------
    # 3. Create TT block & infrastructure
    # ------------------------------------------------------------------
    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        state_dict=layer_sd,
        layer_idx=real_layer_idx,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
    )
    if not is_dense:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode
        block_kwargs["dispatch_buffer_capacity_factor"] = 2
        block_kwargs["routed_expert_activations_dtype"] = ttnn.bfloat16
        block_kwargs["routed_expert_weights_dtype"] = ttnn.bfloat16
        block_kwargs["shared_expert_activations_dtype"] = ttnn.bfloat16
        block_kwargs["shared_expert_weights_dtype"] = ttnn.bfloat16

    logger.info("Creating TtPrefillBlock...")
    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope_setup.get_rope_tensors(isl_total)
    position_ids = torch.arange(isl_total, dtype=torch.long).unsqueeze(0)
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank

    # Shard initial input to device
    h_tt = ttnn.from_torch(
        h0.unsqueeze(0),  # [1, 1, 1024, 7168]
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )
    h_torch = h0.clone()

    # ------------------------------------------------------------------
    # 4. Iteration loop
    # ------------------------------------------------------------------
    pcc_values = []
    norm_torch_values = []
    norm_tt_values = []
    norm_diff_values = []
    kv_pcc_values = []
    pe_pcc_values = []
    kv_lora_rank = config.kv_lora_rank  # 512

    # --- H4: Monkey-patch to capture FFN output on iteration 1 for zero-weight variants ---
    captured_ffn_out = {}
    if layer_idx in (-5, -6) and not is_dense:
        original_moe_path = block._moe_path

        def _capturing_moe_path(ffn_norm_out):
            result = original_moe_path(ffn_norm_out)
            # Read back FFN output to host for inspection
            ffn_host = ttnn.to_torch(
                result,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape),
            ).to(torch.float32)
            captured_ffn_out["iter1_ffn_out"] = ffn_host
            captured_ffn_out["norm"] = torch.norm(ffn_host).item()
            captured_ffn_out["abs_max"] = torch.abs(ffn_host).max().item()
            captured_ffn_out["abs_mean"] = torch.abs(ffn_host).mean().item()
            logger.info(
                f"  [H4] TT MoE output with zero weights: "
                f"||ffn_out||={captured_ffn_out['norm']:.6f}  "
                f"abs_max={captured_ffn_out['abs_max']:.6e}  "
                f"abs_mean={captured_ffn_out['abs_mean']:.6e}"
            )
            # Restore original after first capture
            block._moe_path = original_moe_path
            return result

        block._moe_path = _capturing_moe_path

    for iteration in range(1, num_iters + 1):
        # Fresh KV cache each iteration (prefill writes full seq range)
        tt_kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_cache_head_dim,
            mesh_device=mesh_device,
            seq_len=isl_total,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
        )

        # --- Torch reference ---
        ref_cache = DynamicCache()
        with torch.no_grad():
            layer_out = hf_model.layers[real_layer_idx](
                h_torch,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=ref_cache,
                use_cache=True,
            )
            h_torch_next = layer_out[0]

        # --- TT forward ---
        h_tt_next, _ = block(h_tt, rope_tensors, tt_kvpe_cache)
        ttnn.synchronize_device(mesh_device)

        # --- PCC ---
        tt_out_host = ttnn.to_torch(
            h_tt_next,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        tt_out_host = tt_out_host.squeeze(0)  # [1, 1, isl, emb] -> [1, isl, emb]

        _, pcc = comp_pcc(h_torch_next.float(), tt_out_host.float())
        pcc_values.append(pcc)

        # --- Track norms (H1) ---
        h_torch_out_norm = torch.norm(h_torch_next.float()).item()
        h_tt_out_norm = torch.norm(tt_out_host.float()).item()
        diff_norm = torch.norm((h_torch_next.float() - tt_out_host.float())).item()
        norm_torch_values.append(h_torch_out_norm)
        norm_tt_values.append(h_tt_out_norm)
        norm_diff_values.append(diff_norm)

        # --- KVPE cache PCC (KV nope + PE rope) ---
        tt_kvpe_host = ttMLA.kv_cache_to_host(
            tt_kvpe_cache, mesh_device, sp_axis=sp_axis
        )  # [1, 1, seq_total, head_dim]
        tt_kvpe_host = tt_kvpe_host.squeeze(0).float()  # [1, seq_total, head_dim]
        torch_kvpe = ref_cache.key_cache[real_layer_idx].float()  # [1, 1, seq, head_dim]
        torch_kvpe = torch_kvpe.squeeze(1)  # [1, seq, head_dim]

        # Split into KV (nope) and PE (rope)
        torch_kv = torch_kvpe[:, :, :kv_lora_rank]
        torch_pe = torch_kvpe[:, :, kv_lora_rank:]
        tt_kv = tt_kvpe_host[:, :, :kv_lora_rank]
        tt_pe = tt_kvpe_host[:, :, kv_lora_rank:]

        _, kv_pcc = comp_pcc(torch_kv, tt_kv)
        _, pe_pcc = comp_pcc(torch_pe, tt_pe)
        kv_pcc_values.append(kv_pcc)
        pe_pcc_values.append(pe_pcc)

        logger.info(
            f"  Iter {iteration:>3d}/{num_iters}  PCC={pcc:.6f}  KV_PCC={kv_pcc:.6f}  PE_PCC={pe_pcc:.6f}  "
            f"||h_torch||={h_torch_out_norm:.1f}  ||h_tt||={h_tt_out_norm:.1f}  "
            f"||diff||={diff_norm:.1f}"
        )

        # --- Feed back ---
        h_torch = h_torch_next
        h_tt = h_tt_next

        # --- Cleanup KV cache ---
        ttnn.deallocate(tt_kvpe_cache)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    iterations = list(range(1, num_iters + 1))
    gate_label = f", gate={gate_mode_name}" if not is_dense else ""

    # PCC plot (hidden state)
    axes[0, 0].plot(iterations, pcc_values, marker="o", markersize=3, linewidth=1.5, color="tab:blue")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("PCC")
    axes[0, 0].set_title("Hidden State PCC")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(1, num_iters)
    axes[0, 0].set_ylim(top=1.0)
    axes[0, 0].ticklabel_format(useOffset=False, style="plain", axis="y")

    # KVPE cache PCC (KV nope + PE rope)
    axes[0, 1].plot(
        iterations, kv_pcc_values, marker="o", markersize=3, linewidth=1.5, label="KV (nope)", color="tab:green"
    )
    axes[0, 1].plot(
        iterations, pe_pcc_values, marker="s", markersize=3, linewidth=1.5, label="PE (rope)", color="tab:orange"
    )
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("PCC")
    axes[0, 1].set_title("KVPE Cache PCC")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(1, num_iters)
    axes[0, 1].set_ylim(top=1.0)
    axes[0, 1].ticklabel_format(useOffset=False, style="plain", axis="y")

    # Norm plot (H1: residual growth)
    axes[1, 0].plot(
        iterations, norm_torch_values, marker="o", markersize=3, linewidth=1.5, label="||h_torch||", color="tab:green"
    )
    axes[1, 0].plot(
        iterations, norm_tt_values, marker="s", markersize=3, linewidth=1.5, label="||h_tt||", color="tab:orange"
    )
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("||h|| (L2 norm)")
    axes[1, 0].set_title("Hidden State Norm Growth")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(1, num_iters)

    # Diff norm plot
    axes[1, 1].plot(iterations, norm_diff_values, marker="^", markersize=3, linewidth=1.5, color="tab:red")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("||h_torch - h_tt|| (L2)")
    axes[1, 1].set_title("Absolute Error Growth")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(1, num_iters)

    fig.suptitle(f"Layer {layer_idx} ({layer_type}{gate_label})", fontsize=14)
    fig.tight_layout()

    gate_suffix = f"_{gate_mode_name}" if not is_dense else ""
    plot_path = f"{PLOT_DIR}/pcc_loop_layer_{layer_idx}{gate_suffix}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    min_pcc = min(pcc_values)
    max_pcc = max(pcc_values)
    final_pcc = pcc_values[-1]

    logger.info(f"\n{'='*120}")
    logger.info(f"Layer {layer_idx} ({layer_type}{gate_label}) — {num_iters} iterations summary")
    logger.info(f"{'='*120}")
    logger.info(
        f"  {'Iter':<6s}  {'PCC':>10s}  {'KV_PCC':>10s}  {'PE_PCC':>10s}  {'||h_torch||':>14s}  {'||h_tt||':>14s}  {'||diff||':>14s}  {'rel_err':>10s}"
    )
    logger.info(f"  {'-'*100}")
    for i, (pcc, kv_p, pe_p, nt, ntt, nd) in enumerate(
        zip(pcc_values, kv_pcc_values, pe_pcc_values, norm_torch_values, norm_tt_values, norm_diff_values), 1
    ):
        rel_err = nd / max(nt, 1e-12)
        logger.info(
            f"  {i:<6d}  {pcc:>10.6f}  {kv_p:>10.6f}  {pe_p:>10.6f}  {nt:>14.1f}  {ntt:>14.1f}  {nd:>14.1f}  {rel_err:>10.6f}"
        )
    logger.info(f"  {'-'*80}")
    logger.info(f"  Min PCC:   {min_pcc:.6f}")
    logger.info(f"  Max PCC:   {max_pcc:.6f}")
    logger.info(f"  Final PCC: {final_pcc:.6f}")
    norm_growth = norm_torch_values[-1] / max(norm_torch_values[0], 1e-12)
    diff_growth = norm_diff_values[-1] / max(norm_diff_values[0], 1e-12)
    logger.info(f"  Norm growth (torch): {norm_growth:.2f}x ({norm_torch_values[0]:.1f} → {norm_torch_values[-1]:.1f})")
    logger.info(f"  Diff growth:         {diff_growth:.2f}x ({norm_diff_values[0]:.1f} → {norm_diff_values[-1]:.1f})")
    logger.info(f"{'='*100}")
