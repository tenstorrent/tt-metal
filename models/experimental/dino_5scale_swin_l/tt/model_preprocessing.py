# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading and preprocessing for DINO-5scale (neck, encoder, decoder).

Backbone weights and attention masks are loaded from the shared swin_l module:
  from models.experimental.swin_l.tt import load_backbone_weights, compute_attn_masks
"""

from typing import Dict, Tuple

import torch
import ttnn

from models.experimental.swin_l.tt import load_backbone_weights, compute_attn_masks  # noqa: F401


def _get(sd: dict, key: str) -> torch.Tensor:
    """Get a key from state_dict, raising a clear error if missing."""
    if key not in sd:
        raise KeyError(f"Missing key in checkpoint: {key}")
    return sd[key]


def _to_ttnn_linear_weight(weight: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Transpose [out, in] -> [in, out] and convert to TTNN tile layout."""
    return ttnn.from_torch(weight.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _to_ttnn_linear_bias(bias: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Reshape [out] -> [1, out] and convert to TTNN tile layout."""
    return ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _to_ttnn_norm_param(param: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Reshape [C] -> [1, C] and convert to TTNN tile layout."""
    return ttnn.from_torch(param.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def load_neck_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    in_channels: Tuple[int, ...] = (192, 384, 768, 1536),
    out_channels: int = 256,
    num_groups: int = 32,
) -> dict:
    """
    Load neck (ChannelMapper) weights from mmdet DINO checkpoint.

    Conv weights are stored as ttnn tensors on host.
    GN weights are stored as raw torch tensors (_torch_w, _torch_b)
    so the neck module can create per-level GN params with optimal core grids.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    params: Dict = {"convs": {}, "extra_convs": {}}

    # Main 1x1 convolutions (P2-P5)
    for i in range(len(in_channels)):
        prefix = f"neck.convs.{i}"
        conv_w = _get(sd, f"{prefix}.conv.weight")  # [out, in, 1, 1]
        conv_b = torch.zeros(1, 1, 1, out_channels)
        gn_w = _get(sd, f"{prefix}.gn.weight")  # [256]
        gn_b = _get(sd, f"{prefix}.gn.bias")  # [256]

        params["convs"][i] = {
            "conv": {
                "weight": ttnn.from_torch(conv_w, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(conv_b, dtype=ttnn.bfloat16),
            },
            "gn": {
                "_torch_w": gn_w,
                "_torch_b": gn_b,
            },
        }

    # Extra 3x3 stride-2 conv (P6)
    prefix = "neck.extra_convs.0"
    conv_w = _get(sd, f"{prefix}.conv.weight")  # [256, 1536, 3, 3]
    conv_b = torch.zeros(1, 1, 1, out_channels)
    gn_w = _get(sd, f"{prefix}.gn.weight")
    gn_b = _get(sd, f"{prefix}.gn.bias")

    params["extra_convs"][0] = {
        "conv": {
            "weight": ttnn.from_torch(conv_w, dtype=ttnn.bfloat16),
            "bias": ttnn.from_torch(conv_b, dtype=ttnn.bfloat16),
        },
        "gn": {
            "_torch_w": gn_w,
            "_torch_b": gn_b,
        },
    }

    return params


def load_encoder_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    num_layers: int = 6,
    embed_dims: int = 256,
    num_heads: int = 8,
    num_levels: int = 5,
    num_points: int = 4,
    ffn_dims: int = 2048,
) -> dict:
    """
    Load encoder (DeformableDetrTransformerEncoder) weights from mmdet DINO checkpoint.

    Checkpoint key structure per layer:
      encoder.layers.{i}.self_attn.{value_proj,sampling_offsets,attention_weights,output_proj}.{weight,bias}
      encoder.layers.{i}.ffn.layers.0.0.{weight,bias}  (fc1)
      encoder.layers.{i}.ffn.layers.1.{weight,bias}    (fc2)
      encoder.layers.{i}.norms.{0,1}.{weight,bias}

    Also loads level_embed [num_levels, embed_dims].
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    params: Dict = {"layers": {}}

    for i in range(num_layers):
        prefix = f"encoder.layers.{i}"
        layer_params = {
            "self_attn": {
                "value_proj": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.self_attn.value_proj.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.self_attn.value_proj.bias")),
                },
                "sampling_offsets": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.self_attn.sampling_offsets.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.self_attn.sampling_offsets.bias")),
                },
                "attention_weights": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.self_attn.attention_weights.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.self_attn.attention_weights.bias")),
                },
                "output_proj": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.self_attn.output_proj.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.self_attn.output_proj.bias")),
                },
            },
            "ffn": {
                "fc1": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.ffn.layers.0.0.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.ffn.layers.0.0.bias")),
                },
                "fc2": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.ffn.layers.1.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.ffn.layers.1.bias")),
                },
            },
            "norms": {
                0: {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.0.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.0.bias")),
                },
                1: {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.1.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.1.bias")),
                },
            },
        }
        params["layers"][i] = layer_params

    level_embed = _get(sd, "level_embed")  # [num_levels, embed_dims]
    params["level_embed"] = ttnn.from_torch(level_embed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    params = _to_device_recursive(params, device)
    return params


def load_decoder_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    num_layers: int = 6,
    embed_dims: int = 256,
    num_heads: int = 8,
    num_levels: int = 5,
    num_points: int = 4,
    ffn_dims: int = 2048,
    num_reg_branches: int = 7,
) -> dict:
    """
    Load decoder (DinoTransformerDecoder) weights from mmdet DINO checkpoint.

    Checkpoint key structure per layer:
      decoder.layers.{i}.self_attn.attn.in_proj_{weight,bias}
      decoder.layers.{i}.self_attn.attn.out_proj.{weight,bias}
      decoder.layers.{i}.cross_attn.{value_proj,...,output_proj}.{weight,bias}
      decoder.layers.{i}.ffn.layers.0.0.{weight,bias}
      decoder.layers.{i}.ffn.layers.1.{weight,bias}
      decoder.layers.{i}.norms.{0,1,2}.{weight,bias}

    Also loads:
      decoder.ref_point_head.layers.{0,1}.{weight,bias}
      decoder.norm.{weight,bias}
      bbox_head.reg_branches.{0..6}.{0,2,4}.{weight,bias}
      memory_trans_fc.{weight,bias}
      memory_trans_norm.{weight,bias}
      query_embedding.weight
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    params: Dict = {"layers": {}}

    for i in range(num_layers):
        prefix = f"decoder.layers.{i}"
        layer_params = {
            "self_attn": {
                "in_proj_weight": ttnn.from_torch(
                    _get(sd, f"{prefix}.self_attn.attn.in_proj_weight"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                ),
                "in_proj_bias": ttnn.from_torch(
                    _get(sd, f"{prefix}.self_attn.attn.in_proj_bias").reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                ),
                "out_proj": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.self_attn.attn.out_proj.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.self_attn.attn.out_proj.bias")),
                },
            },
            "cross_attn": {
                "value_proj": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.cross_attn.value_proj.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.cross_attn.value_proj.bias")),
                },
                "sampling_offsets": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.cross_attn.sampling_offsets.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.cross_attn.sampling_offsets.bias")),
                },
                "attention_weights": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.cross_attn.attention_weights.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.cross_attn.attention_weights.bias")),
                },
                "output_proj": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.cross_attn.output_proj.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.cross_attn.output_proj.bias")),
                },
            },
            "ffn": {
                "fc1": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.ffn.layers.0.0.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.ffn.layers.0.0.bias")),
                },
                "fc2": {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"{prefix}.ffn.layers.1.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"{prefix}.ffn.layers.1.bias")),
                },
            },
            "norms": {
                0: {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.0.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.0.bias")),
                },
                1: {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.1.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.1.bias")),
                },
                2: {
                    "weight": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.2.weight")),
                    "bias": _to_ttnn_norm_param(_get(sd, f"{prefix}.norms.2.bias")),
                },
            },
        }
        params["layers"][i] = layer_params

    # ref_point_head: MLP(512, 256, 256, 2) => 2 linear layers
    params["ref_point_head"] = {
        "layers": {
            0: {
                "weight": _to_ttnn_linear_weight(_get(sd, "decoder.ref_point_head.layers.0.weight")),
                "bias": _to_ttnn_linear_bias(_get(sd, "decoder.ref_point_head.layers.0.bias")),
            },
            1: {
                "weight": _to_ttnn_linear_weight(_get(sd, "decoder.ref_point_head.layers.1.weight")),
                "bias": _to_ttnn_linear_bias(_get(sd, "decoder.ref_point_head.layers.1.bias")),
            },
        },
    }

    # Final layer norm
    params["norm"] = {
        "weight": _to_ttnn_norm_param(_get(sd, "decoder.norm.weight")),
        "bias": _to_ttnn_norm_param(_get(sd, "decoder.norm.bias")),
    }

    # Regression branches (7 total: 6 decoder layers + 1 for encoder proposal)
    # Each branch: Linear(256,256)->ReLU->Linear(256,256)->ReLU->Linear(256,4)
    params["reg_branches"] = {}
    for i in range(num_reg_branches):
        branch = []
        for layer_idx in [0, 2, 4]:
            branch.append(
                {
                    "weight": _to_ttnn_linear_weight(_get(sd, f"bbox_head.reg_branches.{i}.{layer_idx}.weight")),
                    "bias": _to_ttnn_linear_bias(_get(sd, f"bbox_head.reg_branches.{i}.{layer_idx}.bias")),
                }
            )
        params["reg_branches"][i] = branch

    # memory_trans_fc and memory_trans_norm (used in pre_decoder / gen_encoder_output_proposals)
    params["memory_trans_fc"] = {
        "weight": _to_ttnn_linear_weight(_get(sd, "memory_trans_fc.weight")),
        "bias": _to_ttnn_linear_bias(_get(sd, "memory_trans_fc.bias")),
    }
    params["memory_trans_norm"] = {
        "weight": _to_ttnn_norm_param(_get(sd, "memory_trans_norm.weight")),
        "bias": _to_ttnn_norm_param(_get(sd, "memory_trans_norm.bias")),
    }

    # Float32 torch copies of pre_decoder weights (top-K selection is extremely
    # sensitive to precision — running it in float32 on host matches the reference).
    params["_torch_pre_decoder"] = {
        "memory_trans_fc_w": _get(sd, "memory_trans_fc.weight").float(),
        "memory_trans_fc_b": _get(sd, "memory_trans_fc.bias").float(),
        "memory_trans_norm_w": _get(sd, "memory_trans_norm.weight").float(),
        "memory_trans_norm_b": _get(sd, "memory_trans_norm.bias").float(),
        "query_embedding": _get(sd, "query_embedding.weight").float(),
        "cls_enc_w": _get(sd, f"bbox_head.cls_branches.{num_layers}.weight").float(),
        "cls_enc_b": _get(sd, f"bbox_head.cls_branches.{num_layers}.bias").float(),
        "reg_enc_layers": [
            {
                "weight": _get(sd, f"bbox_head.reg_branches.{num_layers}.{li}.weight").float(),
                "bias": _get(sd, f"bbox_head.reg_branches.{num_layers}.{li}.bias").float(),
            }
            for li in [0, 2, 4]
        ],
    }

    # query_embedding [900, 256]
    params["query_embedding"] = ttnn.from_torch(
        _get(sd, "query_embedding.weight"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # cls_branches (7 total, for encoder proposal selection)
    params["cls_branches"] = {}
    for i in range(num_reg_branches):
        params["cls_branches"][i] = {
            "weight": _to_ttnn_linear_weight(_get(sd, f"bbox_head.cls_branches.{i}.weight")),
            "bias": _to_ttnn_linear_bias(_get(sd, f"bbox_head.cls_branches.{i}.bias")),
        }

    params = _to_device_recursive(params, device)
    return params


def _to_device_recursive(d, device):
    """Recursively move all ttnn.Tensor values in a nested dict/list to device."""
    if isinstance(d, ttnn.Tensor):
        return ttnn.to_device(d, device)
    if isinstance(d, dict):
        return {k: _to_device_recursive(v, device) for k, v in d.items()}
    if isinstance(d, list):
        return [_to_device_recursive(v, device) for v in d]
    return d
