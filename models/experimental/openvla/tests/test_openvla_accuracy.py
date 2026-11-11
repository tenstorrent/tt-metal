# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Accuracy tests for OpenVLA pieces that the PCC tests with random weights do not reach.

1. Language model Q/K weights: attention scores after RoPE match HF Llama (host only)
2. SigLIP featurizer: TT features match timm, position embedding included (random weights)
3. predict_action: returns the language model's own greedy action tokens (needs OPENVLA_WEIGHTS)
"""

import os

import pytest
import torch
from safetensors import safe_open
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.common.reference_rope import apply_rotary_emb, precompute_freqs_cis
from models.common.utility_functions import comp_pcc, torch_random
from models.experimental.openvla.tt.open_vla import (
    TIMM_OVERRIDE_ACT_LAYER,
    VISION_BACKBONE_TO_RESOLUTION,
    VISION_BACKBONE_TO_TIMM_ID,
    OpenVLAConfig,
    PrismaticVisionBackbone,
    TTOpenVLAForActionPrediction,
    map_openvla_hf_to_meta_keys,
    ttnn_to_torch_safe,
)


# =============================================================================
# TEST 1: Language model Q/K layout
# =============================================================================
@pytest.mark.parametrize("n_heads, head_dim, seq_len", [(2, 128, 64)])
def test_language_model_qk_rope_pcc(n_heads, head_dim, seq_len):
    """
    tt_transformers rotates Q/K in the Meta layout. The weights map_openvla_hf_to_meta_keys produces must give the
    same attention scores under that rotation as the HF weights give under HF RoPE.
    """
    torch.manual_seed(42)
    dim = n_heads * head_dim
    hf_state_dict = {
        "language_model.model.layers.0.self_attn.q_proj.weight": torch.randn(dim, dim),
        "language_model.model.layers.0.self_attn.k_proj.weight": torch.randn(dim, dim),
    }
    x = torch.randn(1, seq_len, dim)

    # HF reference
    config = LlamaConfig(hidden_size=dim, num_attention_heads=n_heads, num_key_value_heads=n_heads)
    cos, sin = LlamaRotaryEmbedding(config)(x, torch.arange(seq_len).unsqueeze(0))
    q = (x @ hf_state_dict["language_model.model.layers.0.self_attn.q_proj.weight"].T).view(1, seq_len, n_heads, -1)
    k = (x @ hf_state_dict["language_model.model.layers.0.self_attn.k_proj.weight"].T).view(1, seq_len, n_heads, -1)
    q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
    hf_scores = q @ k.transpose(-2, -1)

    # Mapped weights with the Meta-layout rotation
    meta_state_dict = map_openvla_hf_to_meta_keys(hf_state_dict, head_dim)
    q = (x @ meta_state_dict["layers.0.attention.wq.weight"].T).view(1, seq_len, n_heads, -1)
    k = (x @ meta_state_dict["layers.0.attention.wk.weight"].T).view(1, seq_len, n_heads, -1)
    q, k = apply_rotary_emb(q, k, precompute_freqs_cis(head_dim, seq_len, theta=10000.0))
    tt_scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)

    _, pcc_value = comp_pcc(hf_scores, tt_scores, 0.9999)

    print(f"\n{'='*60}")
    print(f"LANGUAGE MODEL Q/K ROPE PCC RESULTS")
    print(f"{'='*60}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert pcc_value > 0.9999, f"Attention scores PCC {pcc_value} too low: Q/K not in the layout the RoPE expects"


# =============================================================================
# TEST 2: SigLIP featurizer
# =============================================================================
@pytest.mark.parametrize("batch_size", [1])
def test_siglip_featurizer_pcc(device, batch_size):
    """
    SigLIP features from PrismaticVisionBackbone (TT) vs its timm model (CPU), same random weights.

    Random init leaves pos_embed near zero, which hides a missing position embedding, so it is redrawn at unit scale.
    """
    torch.manual_seed(42)
    backbone_id = "dinosiglip-vit-so-224px"
    backbone_args = (
        True,
        VISION_BACKBONE_TO_RESOLUTION[backbone_id],
        VISION_BACKBONE_TO_TIMM_ID[backbone_id],
        TIMM_OVERRIDE_ACT_LAYER[backbone_id],
    )
    torch_backbone = PrismaticVisionBackbone(*backbone_args)
    torch_backbone.fused_featurizer.pos_embed.data.normal_()
    state_dict = {f"vision_backbone.{k}": v for k, v in torch_backbone.state_dict().items()}
    tt_backbone = PrismaticVisionBackbone(*backbone_args, ttnn_device=device, local_state_dict=state_dict)

    torch_pixel_values = torch_random((batch_size, 3, 224, 224), -1, 1, dtype=torch.float32)
    with torch.no_grad():
        torch_output = torch_backbone.fused_featurizer(torch_pixel_values)

    pixel_values = torch.permute(torch_pixel_values, (0, 2, 3, 1))
    pixel_values = torch.nn.functional.pad(pixel_values, (0, 1, 0, 0, 0, 0, 0, 0))
    pixel_values = ttnn.from_torch(pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = tt_backbone.ttnn_fused_featurizer(pixel_values)
    tt_output_torch = ttnn_to_torch_safe(tt_output, device)

    _, pcc_value = comp_pcc(torch_output, tt_output_torch, 0.99)

    print(f"\n{'='*60}")
    print(f"SigLIP FEATURIZER PCC RESULTS")
    print(f"{'='*60}")
    print(f"  PCC: {pcc_value}")
    print(f"{'='*60}\n")

    assert pcc_value > 0.99, f"SigLIP featurizer PCC {pcc_value} too low"


# =============================================================================
# TEST 3: Action tokens
# =============================================================================
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D if len(ttnn.get_device_ids()) > 1 else None,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_predict_action_tokens(mesh_device, openvla_weights_path):
    """
    predict_action must return the tokens the language model generated, starting with the one the prefill picked,
    and all of them must be action tokens (an end-of-sequence token here means the actions are shifted by one).
    """
    merged_tensors = {}
    for shard in [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]:
        with safe_open(os.path.join(openvla_weights_path, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_tensors[key] = f.get_tensor(key)

    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": "openvla/openvla-7b",
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, kwargs = OpenVLAConfig.from_dict(config_dict, **kwargs)
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )

    prompt = "In: What action should the robot take to pick up the red block?\nOut:"
    inputs = vla.language_model.tokenizer(prompt, return_tensors="pt")
    # What the OpenVLA processor gives for a solid orange image: DINOv2 then SigLIP normalization, stacked on channels
    color = torch.tensor([255.0, 100.0, 50.0]) / 255
    dinov2_pixel = (color - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    siglip_pixel = (color - 0.5) / 0.5
    pixel_values = torch.cat([dinov2_pixel, siglip_pixel]).view(1, 6, 1, 1).expand(1, 6, 224, 224).to(torch.bfloat16)
    _, action_tokens = vla.predict_action(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=pixel_values,
        unnorm_key="bridge_orig",
        do_sample=False,
        return_tokens=True,
    )
    generated_tokens = vla.language_model._last_generated_tokens
    first_action_token = vla.vocab_size - vla.config.n_action_bins

    print(f"\n{'='*60}")
    print(f"ACTION TOKEN RESULTS")
    print(f"{'='*60}")
    print(f"  Returned:  {action_tokens}")
    print(f"  Generated: {generated_tokens}")
    print(f"{'='*60}\n")

    assert len(action_tokens) == vla.get_action_dim("bridge_orig")
    assert action_tokens == generated_tokens, "predict_action did not return the generated tokens"
    assert all(
        first_action_token <= token < vla.vocab_size for token in action_tokens
    ), f"Non-action token in {action_tokens}"
