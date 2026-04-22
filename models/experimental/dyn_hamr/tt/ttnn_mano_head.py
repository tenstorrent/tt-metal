# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional tt-nn port of the HaMeR MANO cross-attention head.

Consumes the ViT feature map from ``ttnn_vit.forward`` (token-major,
``(B, 192, 1280)``) and emits a ``(B, 157)`` tensor matching the torch
reference contract: 16 flattened 3×3 rotation matrices + 10 shape +
3 weak-perspective camera.  Inference-only, random weights materialized via
``build_parameters_from_reference``.
"""
from __future__ import annotations

from typing import Any, Dict

import torch

try:
    import ttnn
except Exception:  # noqa: BLE001
    ttnn = None


def _t(weight: torch.Tensor, device: Any) -> Any:
    return ttnn.from_torch(
        weight.to(torch.bfloat16).contiguous(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def build_parameters_from_reference(ref, device: Any) -> Dict[str, Any]:
    head = ref.head
    td = head.transformer
    params: Dict[str, Any] = {
        "to_token_embedding": {
            "weight": _t(td.to_token_embedding.weight.t(), device),
            "bias": _t(td.to_token_embedding.bias, device),
        },
        "pos_embedding": _t(td.pos_embedding.squeeze(0), device),  # (1, dim)
        "layers": [],
        "decpose": {
            "weight": _t(head.decpose.weight.t(), device),
            "bias": _t(head.decpose.bias, device),
        },
        "decshape": {
            "weight": _t(head.decshape.weight.t(), device),
            "bias": _t(head.decshape.bias, device),
        },
        "deccam": {
            "weight": _t(head.deccam.weight.t(), device),
            "bias": _t(head.deccam.bias, device),
        },
        "init_hand_pose": _t(head.init_hand_pose, device),
        "init_betas": _t(head.init_betas, device),
        "init_cam": _t(head.init_cam, device),
    }
    for blk in td.layers:
        params["layers"].append({
            "norm_sa": {"weight": _t(blk.norm_sa.weight, device), "bias": _t(blk.norm_sa.bias, device)},
            "sa": {
                "qkv_w": _t(blk.sa.to_qkv.weight.t(), device),
                "out_w": _t(blk.sa.to_out.weight.t(), device),
                "out_b": _t(blk.sa.to_out.bias, device),
            },
            "norm_ca": {"weight": _t(blk.norm_ca.weight, device), "bias": _t(blk.norm_ca.bias, device)},
            "ca": {
                "q_w": _t(blk.ca.to_q.weight.t(), device),
                "kv_w": _t(blk.ca.to_kv.weight.t(), device),
                "out_w": _t(blk.ca.to_out.weight.t(), device),
                "out_b": _t(blk.ca.to_out.bias, device),
            },
            "norm_ff": {"weight": _t(blk.norm_ff.weight, device), "bias": _t(blk.norm_ff.bias, device)},
            "ff": {
                "fc1_w": _t(blk.ff.net[0].weight.t(), device),
                "fc1_b": _t(blk.ff.net[0].bias, device),
                "fc2_w": _t(blk.ff.net[3].weight.t(), device),
                "fc2_b": _t(blk.ff.net[3].bias, device),
            },
        })
    return params


def _self_attn(x, sa_p, heads: int, dim_head: int):
    qkv = x @ sa_p["qkv_w"]
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv, num_heads=heads, transpose_key=True
    )
    attn = (q @ k) * (dim_head ** -0.5)
    attn = ttnn.softmax(attn, dim=-1)
    out = attn @ v
    out = ttnn.transformer.concatenate_heads(out)
    out = out @ sa_p["out_w"]
    out = out + sa_p["out_b"]
    return out


def _cross_attn(x, context, ca_p, heads: int, dim_head: int):
    q = x @ ca_p["q_w"]
    kv = context @ ca_p["kv_w"]
    # Split K/V by halving the last dim.
    inner = kv.shape[-1] // 2
    k, v = ttnn.split(kv, [inner, inner], dim=-1)
    # Reshape into heads.
    q = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn.concat([q, k, v], dim=-1), num_heads=heads, transpose_key=True
    )
    # ``split_query_key_value_and_split_heads`` expected a fused QKV; if the
    # returned triple is (q_h, k_h, v_h), use directly.
    q_h, k_h, v_h = q
    attn = (q_h @ k_h) * (dim_head ** -0.5)
    attn = ttnn.softmax(attn, dim=-1)
    out = attn @ v_h
    out = ttnn.transformer.concatenate_heads(out)
    out = out @ ca_p["out_w"]
    out = out + ca_p["out_b"]
    return out


def _ffn(x, ff_p):
    h = x @ ff_p["fc1_w"]
    h = h + ff_p["fc1_b"]
    h = ttnn.gelu(h)
    h = h @ ff_p["fc2_w"]
    h = h + ff_p["fc2_b"]
    return h


def decoder_block(x, context, block_params: Dict[str, Any], heads: int = 8, dim_head: int = 64):
    h = ttnn.layer_norm(x, weight=block_params["norm_sa"]["weight"], bias=block_params["norm_sa"]["bias"])
    x = x + _self_attn(h, block_params["sa"], heads, dim_head)
    h = ttnn.layer_norm(x, weight=block_params["norm_ca"]["weight"], bias=block_params["norm_ca"]["bias"])
    x = x + _cross_attn(h, context, block_params["ca"], heads, dim_head)
    h = ttnn.layer_norm(x, weight=block_params["norm_ff"]["weight"], bias=block_params["norm_ff"]["bias"])
    x = x + _ffn(h, block_params["ff"])
    return x


def rot6d_to_rotmat_torch(x6: torch.Tensor) -> torch.Tensor:
    """Host-side post-processing — kept on CPU because 6D → rotmat is tiny
    and needs cross product / L2 normalization that aren't on the NPU-critical
    path.  Runs on the (B, 96) tensor copied back from the device.
    """
    import torch.nn.functional as F
    x = x6.reshape(-1, 3, 2)
    b1 = F.normalize(x[..., 0], dim=-1)
    b2 = F.normalize(x[..., 1] - (b1 * x[..., 1]).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def forward(feature_map_bchw, params: Dict[str, Any], depth: int = 6, heads: int = 8, dim_head: int = 64,
            num_hand_joints: int = 15):
    """MANO head forward on device, returning host torch tensors post head."""
    # Input is (B, 1280, Hp, Wp) feature map from ViT; token-major for cross-attn.
    b, c, hp, wp = feature_map_bchw.shape
    context = ttnn.reshape(feature_map_bchw, (b, c, hp * wp))
    context = ttnn.permute(context, (0, 2, 1))  # (B, N, C)

    # Zero query token lifted to device, then embedded.
    token_torch = torch.zeros(b, 1, 1, dtype=torch.bfloat16)
    token = ttnn.from_torch(token_torch, device=context.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    x = token @ params["to_token_embedding"]["weight"]
    x = x + params["to_token_embedding"]["bias"]
    x = x + params["pos_embedding"]

    for i in range(depth):
        x = decoder_block(x, context, params["layers"][i], heads=heads, dim_head=dim_head)

    # Regression readouts (all fused on device).
    x_sq = ttnn.squeeze(x, dim=1)  # (B, dim)
    pose = x_sq @ params["decpose"]["weight"]
    pose = pose + params["decpose"]["bias"]
    pose = pose + params["init_hand_pose"]
    betas = x_sq @ params["decshape"]["weight"]
    betas = betas + params["decshape"]["bias"]
    betas = betas + params["init_betas"]
    cam = x_sq @ params["deccam"]["weight"]
    cam = cam + params["deccam"]["bias"]
    cam = cam + params["init_cam"]

    # Copy back to host for the 6D → rotmat conversion (tiny tensor, not worth on-NPU).
    pose_host = ttnn.to_torch(pose).to(torch.float32)
    betas_host = ttnn.to_torch(betas).to(torch.float32)
    cam_host = ttnn.to_torch(cam).to(torch.float32)
    rotmats = rot6d_to_rotmat_torch(pose_host).view(b, num_hand_joints + 1, 3, 3)
    out = torch.cat([rotmats.reshape(b, -1), betas_host, cam_host], dim=-1)  # (B, 157)
    return out
