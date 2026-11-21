# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Accuracy tests for OpenVLA pieces that the PCC tests with random weights do not reach.

1. Language model Q/K weights: attention scores after RoPE match HF Llama (host only)
2. SigLIP featurizer: TT features match timm, position embedding included (random weights)
3. Vision backbone: DINOv2 and SigLIP features match the reference on the checkpoint (needs OPENVLA_WEIGHTS)
4. Language model prefill: logits after the first decoder layers match HF Llama on the checkpoint
   (needs OPENVLA_WEIGHTS)
5. predict_action: returns the generated action tokens in order, decoded to actions; teacher-forced on the HF
   reference tokens, each TT top-1 is in the reference top-5 (needs OPENVLA_WEIGHTS)
6. predict_action on device rejects generation options it cannot honour (host only)

Reference for 5: references/generate_reference_actions.py (HF, bfloat16 on CPU, revision 47a0ec7f).
Reference for 3 and 4: openvla/openvla-7b (revision 47a0ec7f) on CPU in float32, through the timm towers of
PrismaticVisionBackbone (the code of the checkpoint's modeling_prismatic.py) and transformers' LlamaForCausalLM.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.common.reference_rope import apply_rotary_emb, precompute_freqs_cis
from models.common.utility_functions import comp_pcc, torch_random
from models.experimental.openvla.tt.open_vla import (
    TIMM_OVERRIDE_ACT_LAYER,
    VISION_BACKBONE_TO_RESOLUTION,
    VISION_BACKBONE_TO_TIMM_ID,
    OpenVLAConfig,
    OpenVLALanguageModel,
    PrismaticProjector,
    PrismaticVisionBackbone,
    TTOpenVLAForActionPrediction,
    map_openvla_hf_to_meta_keys,
    ttnn_to_torch_safe,
)

BACKBONE_ID = "dinosiglip-vit-so-224px"
PROMPT = "In: What action should the robot take to pick up the red block?\nOut:"
OPENVLA_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D if len(ttnn.get_device_ids()) > 1 else None,
    "trace_region_size": 30000000,
    "num_command_queues": 1,
    "l1_small_size": 81920,
}
OPENVLA_MESH_DEVICE = {
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


def vision_backbone_args():
    return (
        True,
        VISION_BACKBONE_TO_RESOLUTION[BACKBONE_ID],
        VISION_BACKBONE_TO_TIMM_ID[BACKBONE_ID],
        TIMM_OVERRIDE_ACT_LAYER[BACKBONE_ID],
    )


def load_openvla_weights(weights_path, prefixes):
    """Checkpoint tensors whose names start with one of `prefixes`, read only from the shards that hold them."""
    with open(os.path.join(weights_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    keys = [key for key in weight_map if key.startswith(prefixes)]
    tensors = {}
    for shard in sorted({weight_map[key] for key in keys}):
        with safe_open(os.path.join(weights_path, shard), framework="pt", device="cpu") as f:
            for key in keys:
                if weight_map[key] == shard:
                    tensors[key] = f.get_tensor(key)
    return tensors


def normalize_image(image):
    """A 3x224x224 image in [0, 1], normalized for DINOv2 and SigLIP and stacked on channels as OpenVLA does."""
    dinov2_mean = torch.tensor([0.484375, 0.455078125, 0.40625]).view(3, 1, 1)
    dinov2_std = torch.tensor([0.228515625, 0.2236328125, 0.224609375]).view(3, 1, 1)
    return torch.cat([(image - dinov2_mean) / dinov2_std, (image - 0.5) / 0.5]).unsqueeze(0)


def fixed_pixel_values():
    y, x = torch.meshgrid(torch.linspace(0, 1, 224), torch.linspace(0, 1, 224), indexing="ij")
    return normalize_image(torch.stack([x, y, 0.5 + 0.5 * torch.sin(12 * x) * torch.cos(12 * y)]))


def to_tt_pixel_values(pixel_values, device):
    """The two NHWC inputs, channels padded to 4, that the TT vision backbone takes."""
    images = torch.split(pixel_values.permute(0, 2, 3, 1), [3, 3], dim=3)
    return [
        ttnn.from_torch(
            torch.nn.functional.pad(image, (0, 1)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        for image in images
    ]


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
    torch_backbone = PrismaticVisionBackbone(*vision_backbone_args())
    torch_backbone.fused_featurizer.pos_embed.data.normal_()
    state_dict = {f"vision_backbone.{k}": v for k, v in torch_backbone.state_dict().items()}
    tt_backbone = PrismaticVisionBackbone(*vision_backbone_args(), ttnn_device=device, local_state_dict=state_dict)

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
# TEST 3: Vision backbone on the checkpoint
# =============================================================================
def test_vision_backbone_checkpoint_pcc(device, openvla_weights_path):
    """
    DINOv2 and SigLIP patch features from the TT vision backbone vs its timm towers, both with the checkpoint's weights,
    on a fixed image.
    """
    state_dict = load_openvla_weights(openvla_weights_path, ("vision_backbone.",))
    torch_backbone = PrismaticVisionBackbone(*vision_backbone_args(), local_state_dict=state_dict)
    tt_backbone = PrismaticVisionBackbone(*vision_backbone_args(), ttnn_device=device, local_state_dict=state_dict)

    pixel_values = fixed_pixel_values()
    with torch.no_grad():
        torch_output = torch_backbone(pixel_values)
    tt_output = ttnn_to_torch_safe(tt_backbone(to_tt_pixel_values(pixel_values, device)), device)

    dinov2_dim = torch_backbone.featurizer.embed_dim
    _, dinov2_pcc = comp_pcc(torch_output[..., :dinov2_dim], tt_output[..., :dinov2_dim])
    _, siglip_pcc = comp_pcc(torch_output[..., dinov2_dim:], tt_output[..., dinov2_dim:])

    print(f"\n{'='*60}")
    print(f"VISION BACKBONE CHECKPOINT PCC RESULTS")
    print(f"{'='*60}")
    print(f"  DINOv2 PCC: {dinov2_pcc}")
    print(f"  SigLIP PCC: {siglip_pcc}")
    print(f"{'='*60}\n")

    assert dinov2_pcc > 0.97, f"DINOv2 features PCC {dinov2_pcc} too low"
    assert siglip_pcc > 0.97, f"SigLIP features PCC {siglip_pcc} too low"


# =============================================================================
# TEST 4: Language model prefill on the checkpoint
# =============================================================================
@pytest.mark.parametrize("device_params", [OPENVLA_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [OPENVLA_MESH_DEVICE], indirect=True)
@pytest.mark.parametrize("num_layers", [2])
def test_language_model_prefill_checkpoint_pcc(mesh_device, openvla_weights_path, num_layers):
    """
    Last-position logits of the language model cut to its first decoder layers vs HF LlamaForCausalLM with the
    checkpoint's weights, on the input predict_action builds: BOS, the projected patches of a fixed image, the prompt.
    """
    layers = tuple(f"language_model.model.layers.{i}." for i in range(num_layers))
    lm_state_dict = load_openvla_weights(
        openvla_weights_path,
        ("language_model.model.embed_tokens.", "language_model.model.norm.", "language_model.lm_head.") + layers,
    )
    vision_state_dict = load_openvla_weights(openvla_weights_path, ("vision_backbone.", "projector."))
    language_model = OpenVLALanguageModel(mesh_device, local_state_dict=lm_state_dict, num_layers=num_layers)

    input_ids = language_model.tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([input_ids, torch.tensor([[29871]])], dim=1)
    vision_backbone = PrismaticVisionBackbone(*vision_backbone_args(), local_state_dict=vision_state_dict)
    projector = PrismaticProjector(True, vision_backbone.embed_dim, 4096)
    projector.load_state_dict(
        {k.removeprefix("projector."): v for k, v in vision_state_dict.items() if k.startswith("projector.")}
    )
    # LlamaConfig defaults are Llama-2-7B's
    reference = LlamaForCausalLM(LlamaConfig(vocab_size=32064, num_hidden_layers=num_layers))
    reference.load_state_dict({k.removeprefix("language_model."): v.float() for k, v in lm_state_dict.items()})
    with torch.no_grad():
        token_embeds = reference.model.embed_tokens(input_ids)
        patch_embeds = projector(vision_backbone(fixed_pixel_values()))
        inputs_embeds = torch.cat([token_embeds[:, :1], patch_embeds, token_embeds[:, 1:]], dim=1)
        reference_logits = reference(inputs_embeds=inputs_embeds).logits[0, -1]

    tt_inputs_embeds = ttnn.from_torch(
        inputs_embeds.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_logits = language_model(inputs_embeds=tt_inputs_embeds)[0].reshape(-1)[: reference_logits.shape[0]].float()

    _, pcc_value = comp_pcc(reference_logits, tt_logits)
    reference_top5 = reference_logits.topk(5).indices.tolist()
    tt_top1 = tt_logits.argmax().item()

    print(f"\n{'='*60}")
    print(f"LANGUAGE MODEL PREFILL CHECKPOINT RESULTS ({num_layers} layers, {inputs_embeds.shape[1]} positions)")
    print(f"{'='*60}")
    print(f"  Logits PCC: {pcc_value}")
    print(f"  Reference top-5: {reference_top5}, TT top-1: {tt_top1}")
    print(f"{'='*60}\n")

    assert pcc_value > 0.97, f"Logits PCC {pcc_value} too low"
    assert tt_top1 in reference_top5, f"TT top-1 {tt_top1} not in the reference top-5 {reference_top5}"


# =============================================================================
# TEST 5: Action tokens
# =============================================================================
# From references/generate_reference_actions.py: the HF greedy action tokens and, at each step, the HF top-5
REFERENCE_ACTIONS = {
    "orange_red_block": {
        "rgb": (255, 100, 50),
        "instruction": "pick up the red block",
        "tokens": [31868, 31744, 31947, 31864, 31834, 31823, 31872],
        "top5": [
            [31868, 31839, 31861, 31999, 31884],
            [31744, 31798, 31885, 31794, 31776],
            [31947, 31949, 31847, 31886, 31913],
            [31864, 31870, 31827, 31917, 31896],
            [31834, 31786, 31866, 31782, 31855],
            [31823, 31848, 31850, 31812, 31827],
            [31872, 31744, 31873, 31858, 31811],
        ],
    },
    "blue_put_cup": {
        "rgb": (40, 90, 200),
        "instruction": "put the cup on the plate",
        "tokens": [31860, 31865, 31876, 31873, 31860, 31853, 31744],
        "top5": [
            [31860, 31999, 31871, 31995, 31873],
            [31865, 31855, 31832, 31843, 31831],
            [31876, 31877, 31887, 31881, 31888],
            [31873, 31872, 31881, 31882, 31883],
            [31860, 31861, 31859, 31868, 31857],
            [31853, 31854, 31852, 31855, 31856],
            [31744, 31872, 31745, 31785, 31758],
        ],
    },
}
OPENVLA_VOCAB_SIZE = 32000
OPENVLA_ACTION_BINS = 256


def reference_unnormalized_actions(tokens, stats):
    """OpenVLA's token-to-action decoding: each token's bin center, scaled from [-1, 1] to [q01, q99] where masked."""
    bins = np.linspace(-1, 1, OPENVLA_ACTION_BINS)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    normalized = bin_centers[np.clip(OPENVLA_VOCAB_SIZE - np.array(tokens) - 1, 0, bin_centers.shape[0] - 1)]
    low, high = np.array(stats["q01"]), np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(low, dtype=bool)))
    return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


@pytest.mark.parametrize("device_params", [OPENVLA_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [OPENVLA_MESH_DEVICE], indirect=True)
def test_predict_action_tokens(mesh_device, openvla_weights_path):
    """
    For each reference case, greedy predict_action must return the tokens the language model generated, starting
    with the one the prefill picked, all of them action tokens (an end-of-sequence token here means the actions are
    shifted by one), decoded to actions. Then, with the reference tokens fed back in (teacher forcing), the model's
    top-1 at every step must be in the reference top-5.
    """
    merged_tensors = load_openvla_weights(openvla_weights_path, ("",))
    kwargs = {
        "return_unused_kwargs": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": False,
        "name_or_path": "openvla/openvla-7b",
        "pretrained_model_name_or_path": openvla_weights_path,
    }
    config_dict, kwargs = OpenVLAConfig.get_config_dict(**kwargs)
    vla_config, kwargs = OpenVLAConfig.from_dict(config_dict, **kwargs)
    vla = TTOpenVLAForActionPrediction(vla_config, ttnn_device=mesh_device, local_state_dict=merged_tensors).to(
        "cpu", dtype=torch.bfloat16
    )
    first_action_token = OPENVLA_VOCAB_SIZE - OPENVLA_ACTION_BINS
    action_stats = vla_config.norm_stats["bridge_orig"]["action"]

    for case, reference in REFERENCE_ACTIONS.items():
        prompt = f"In: What action should the robot take to {reference['instruction']}?\nOut:"
        inputs = vla.language_model.tokenizer(prompt, return_tensors="pt")
        image = (torch.tensor(reference["rgb"], dtype=torch.float32) / 255).view(3, 1, 1).expand(3, 224, 224)
        predict_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": normalize_image(image).to(torch.bfloat16),
            "unnorm_key": "bridge_orig",
            "do_sample": False,
            "return_tokens": True,
        }
        actions, action_tokens = vla.predict_action(**predict_kwargs)
        generated_tokens = vla.language_model._last_generated_tokens

        vla.language_model.forced_tokens = reference["tokens"]
        try:
            vla.predict_action(**predict_kwargs)
            forced_top1 = vla.language_model._last_generated_tokens
        finally:
            vla.language_model.forced_tokens = None
        outside_top5 = [i for i, token in enumerate(forced_top1) if token not in reference["top5"][i]]

        print(f"\n{'='*60}")
        print(f"ACTION TOKEN RESULTS ({case})")
        print(f"{'='*60}")
        print(f"  Greedy:              {action_tokens}")
        print(f"  Reference:           {reference['tokens']}")
        print(f"  Teacher-forced top-1: {forced_top1}")
        print(f"  Steps outside the reference top-5: {outside_top5}")
        print(f"{'='*60}\n")

        assert len(action_tokens) == vla.get_action_dim("bridge_orig")
        assert action_tokens == generated_tokens, f"{case}: predict_action did not return the generated tokens"
        assert all(
            first_action_token <= token < OPENVLA_VOCAB_SIZE for token in action_tokens
        ), f"{case}: non-action token in {action_tokens}"
        assert np.allclose(
            actions, reference_unnormalized_actions(action_tokens, action_stats)
        ), f"{case}: actions {actions} are not the decoded tokens {action_tokens}"
        assert not outside_top5, f"{case}: top-1 outside the reference top-5 at steps {outside_top5}: {forced_top1}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"do_sample": True},
        {"num_beams": 2},
        {"logits_processor": []},
        {"attention_mask": torch.tensor([[0, 1, 1]])},
    ],
)
def test_predict_action_rejects_unsupported_generate_options(kwargs):
    """The device path decodes greedily in one forward call, so options it would ignore must raise."""
    vla = SimpleNamespace(ttnn_device=object(), cached_output=None, _skip_kv_reset=True)
    with pytest.raises(ValueError, match="decodes greedily"):
        TTOpenVLAForActionPrediction.predict_action(
            vla, input_ids=torch.tensor([[1, 2, 29871]]), unnorm_key="bridge_orig", pixel_values=None, **kwargs
        )
