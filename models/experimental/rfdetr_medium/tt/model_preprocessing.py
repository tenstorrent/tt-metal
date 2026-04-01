# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading and preprocessing for RF-DETR Medium TTNN implementation.
All weights converted to TTNN tensors on device for pure on-device inference.
"""

import torch
import ttnn

from models.experimental.rfdetr_medium.common import (
    VIT_HIDDEN_SIZE,
    VIT_NUM_LAYERS,
    HIDDEN_DIM,
    DEC_LAYERS,
)


def _to_device(tensor, device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return ttnn.from_torch(tensor.contiguous(), dtype=dtype, layout=layout, device=device)


def _get_dinov2_backbone(torch_model):
    """Navigate to WindowedDinov2WithRegistersBackbone."""
    return torch_model.backbone[0].encoder.encoder


def load_backbone_weights(torch_model, device):
    """
    Extract DINOv2-ViT-S backbone weights.
    dinov2_windowed_small: no registers, GELU MLP, norm1/norm2.
    """
    backbone_model = _get_dinov2_backbone(torch_model)
    sd = backbone_model.state_dict()

    # Patch embedding: Conv2d(3, 384, 16, 16)
    proj_w = sd["embeddings.patch_embeddings.projection.weight"]
    proj_b = sd["embeddings.patch_embeddings.projection.bias"]
    C_out, C_in, kH, kW = proj_w.shape
    pad_value = 4 - C_in
    prep_w = torch.nn.functional.pad(proj_w, (0, 0, 0, 0, 0, pad_value))
    prep_w = prep_w.permute(2, 3, 1, 0).reshape(-1, C_out)

    params = {
        "proj_weight": _to_device(prep_w, device),
        "proj_bias": _to_device(proj_b, device),
        "pos_embed": ttnn.from_torch(
            sd["embeddings.position_embeddings"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        "cls_token": ttnn.from_torch(
            sd["embeddings.cls_token"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        ),
        "layernorm_weight": _to_device(sd["layernorm.weight"], device),
        "layernorm_bias": _to_device(sd["layernorm.bias"], device),
        "layers": {},
    }

    for i in range(VIT_NUM_LAYERS):
        pfx = f"encoder.layer.{i}."
        q_w = sd[f"{pfx}attention.attention.query.weight"]
        k_w = sd[f"{pfx}attention.attention.key.weight"]
        v_w = sd[f"{pfx}attention.attention.value.weight"]
        q_b = sd[f"{pfx}attention.attention.query.bias"]
        k_b = sd[f"{pfx}attention.attention.key.bias"]
        v_b = sd[f"{pfx}attention.attention.value.bias"]

        qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)

        params["layers"][i] = {
            "norm1_weight": _to_device(sd[f"{pfx}norm1.weight"], device),
            "norm1_bias": _to_device(sd[f"{pfx}norm1.bias"], device),
            "qkv_weight": _to_device(qkv_weight.T, device),
            "qkv_bias": _to_device(qkv_bias, device),
            "proj_weight": _to_device(sd[f"{pfx}attention.output.dense.weight"].T, device),
            "proj_bias": _to_device(sd[f"{pfx}attention.output.dense.bias"], device),
            "ls1_scale": _to_device(sd.get(f"{pfx}layer_scale1.lambda1", torch.ones(VIT_HIDDEN_SIZE)), device),
            "norm2_weight": _to_device(sd[f"{pfx}norm2.weight"], device),
            "norm2_bias": _to_device(sd[f"{pfx}norm2.bias"], device),
            "fc1_weight": _to_device(sd[f"{pfx}mlp.fc1.weight"].T, device),
            "fc1_bias": _to_device(sd[f"{pfx}mlp.fc1.bias"], device),
            "fc2_weight": _to_device(sd[f"{pfx}mlp.fc2.weight"].T, device),
            "fc2_bias": _to_device(sd[f"{pfx}mlp.fc2.bias"], device),
            "ls2_scale": _to_device(sd.get(f"{pfx}layer_scale2.lambda1", torch.ones(VIT_HIDDEN_SIZE)), device),
        }

    return params


def load_projector_weights(torch_model, device):
    """
    Extract MultiScaleProjector weights for on-device conv2d.

    Projector structure (scale=1.0, C2f):
      cv1: Conv1x1(1536→256, bias=False) + LN(256) + SiLU
      3 × Bottleneck: Conv3x3(128→128) + LN + SiLU × 2, with shortcut
      cv2: Conv1x1(640→256, bias=False) + LN(256) + SiLU
      final LN(256)
    """
    projector = torch_model.backbone[0].projector
    sd = projector.state_dict()

    # Stage 0, C2f block + final LN
    pfx = "stages.0."

    c2f_c = HIDDEN_DIM // 2  # 128

    params = {
        # cv1: Conv1x1(1536→256) + LN + SiLU
        "cv1_weight": ttnn.from_torch(sd[f"{pfx}0.cv1.conv.weight"], dtype=ttnn.bfloat16),
        "cv1_bias_conv": None,  # no bias in Conv
        "cv1_ln_weight": _to_device(sd[f"{pfx}0.cv1.bn.weight"], device),
        "cv1_ln_bias": _to_device(sd[f"{pfx}0.cv1.bn.bias"], device),
        # cv2: Conv1x1(640→256) + LN + SiLU
        "cv2_weight": ttnn.from_torch(sd[f"{pfx}0.cv2.conv.weight"], dtype=ttnn.bfloat16),
        "cv2_bias_conv": None,
        "cv2_ln_weight": _to_device(sd[f"{pfx}0.cv2.bn.weight"], device),
        "cv2_ln_bias": _to_device(sd[f"{pfx}0.cv2.bn.bias"], device),
        # Final LN
        "final_ln_weight": _to_device(sd[f"{pfx}1.weight"], device),
        "final_ln_bias": _to_device(sd[f"{pfx}1.bias"], device),
        "bottlenecks": {},
    }

    # 3 Bottleneck blocks
    for bn_idx in range(3):
        bn_pfx = f"{pfx}0.m.{bn_idx}."
        params["bottlenecks"][bn_idx] = {
            "cv1_weight": ttnn.from_torch(sd[f"{bn_pfx}cv1.conv.weight"], dtype=ttnn.bfloat16),
            "cv1_bias_conv": None,
            "cv1_ln_weight": _to_device(sd[f"{bn_pfx}cv1.bn.weight"], device),
            "cv1_ln_bias": _to_device(sd[f"{bn_pfx}cv1.bn.bias"], device),
            "cv2_weight": ttnn.from_torch(sd[f"{bn_pfx}cv2.conv.weight"], dtype=ttnn.bfloat16),
            "cv2_bias_conv": None,
            "cv2_ln_weight": _to_device(sd[f"{bn_pfx}cv2.bn.weight"], device),
            "cv2_ln_bias": _to_device(sd[f"{bn_pfx}cv2.bn.bias"], device),
        }

    return params


def load_decoder_weights(torch_model, device):
    """
    Extract decoder weights including cross-attention (deformable) parameters.
    """
    decoder = torch_model.transformer.decoder
    sd = decoder.state_dict()

    # ref_point_head: MLP(512, 256, 256, 2) = Linear(512, 256) + ReLU + Linear(256, 256)
    params = {
        "ref_point_head_w0": _to_device(sd["ref_point_head.layers.0.weight"].T, device),
        "ref_point_head_b0": _to_device(sd["ref_point_head.layers.0.bias"], device),
        "ref_point_head_w1": _to_device(sd["ref_point_head.layers.1.weight"].T, device),
        "ref_point_head_b1": _to_device(sd["ref_point_head.layers.1.bias"], device),
        "layers": {},
    }

    # Final norm
    if hasattr(decoder, "norm") and decoder.norm is not None and not isinstance(decoder.norm, torch.nn.Identity):
        params["final_norm_weight"] = _to_device(sd["norm.weight"], device)
        params["final_norm_bias"] = _to_device(sd["norm.bias"], device)
    else:
        params["final_norm_weight"] = None
        params["final_norm_bias"] = None

    for i in range(DEC_LAYERS):
        pfx = f"layers.{i}."

        # Self-attention
        in_proj_w = sd[f"{pfx}self_attn.in_proj_weight"]
        in_proj_b = sd[f"{pfx}self_attn.in_proj_bias"]
        q_w, k_w, v_w = in_proj_w.chunk(3, dim=0)
        q_b, k_b, v_b = in_proj_b.chunk(3, dim=0)

        layer_params = {
            "q_proj_weight": _to_device(q_w.T, device),
            "q_proj_bias": _to_device(q_b, device),
            "k_proj_weight": _to_device(k_w.T, device),
            "k_proj_bias": _to_device(k_b, device),
            "v_proj_weight": _to_device(v_w.T, device),
            "v_proj_bias": _to_device(v_b, device),
            "out_proj_weight": _to_device(sd[f"{pfx}self_attn.out_proj.weight"].T, device),
            "out_proj_bias": _to_device(sd[f"{pfx}self_attn.out_proj.bias"], device),
            "norm1_weight": _to_device(sd[f"{pfx}norm1.weight"], device),
            "norm1_bias": _to_device(sd[f"{pfx}norm1.bias"], device),
        }

        # Cross-attention (MSDeformAttn): value_proj, sampling_offsets, attention_weights, output_proj
        layer_params.update(
            {
                "ca_value_weight": _to_device(sd[f"{pfx}cross_attn.value_proj.weight"].T, device),
                "ca_value_bias": _to_device(sd[f"{pfx}cross_attn.value_proj.bias"], device),
                "ca_offset_weight": _to_device(sd[f"{pfx}cross_attn.sampling_offsets.weight"].T, device),
                "ca_offset_bias": _to_device(sd[f"{pfx}cross_attn.sampling_offsets.bias"], device),
                "ca_attn_weight": _to_device(sd[f"{pfx}cross_attn.attention_weights.weight"].T, device),
                "ca_attn_bias": _to_device(sd[f"{pfx}cross_attn.attention_weights.bias"], device),
                "ca_output_weight": _to_device(sd[f"{pfx}cross_attn.output_proj.weight"].T, device),
                "ca_output_bias": _to_device(sd[f"{pfx}cross_attn.output_proj.bias"], device),
                "norm2_weight": _to_device(sd[f"{pfx}norm2.weight"], device),
                "norm2_bias": _to_device(sd[f"{pfx}norm2.bias"], device),
            }
        )

        # FFN
        layer_params.update(
            {
                "linear1_weight": _to_device(sd[f"{pfx}linear1.weight"].T, device),
                "linear1_bias": _to_device(sd[f"{pfx}linear1.bias"], device),
                "linear2_weight": _to_device(sd[f"{pfx}linear2.weight"].T, device),
                "linear2_bias": _to_device(sd[f"{pfx}linear2.bias"], device),
                "norm3_weight": _to_device(sd[f"{pfx}norm3.weight"], device),
                "norm3_bias": _to_device(sd[f"{pfx}norm3.bias"], device),
            }
        )

        params["layers"][i] = layer_params

    return params


def load_detection_head_weights(torch_model, device):
    """Extract detection head weights (class_embed + bbox_embed)."""
    params = {
        "cls_weight": _to_device(torch_model.class_embed.weight.T, device),
        "cls_bias": _to_device(torch_model.class_embed.bias, device),
    }
    bbox_sd = torch_model.bbox_embed.state_dict()
    for i in range(3):
        params[f"bbox_weight_{i}"] = _to_device(bbox_sd[f"layers.{i}.weight"].T, device)
        params[f"bbox_bias_{i}"] = _to_device(bbox_sd[f"layers.{i}.bias"], device)
    return params
