# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-nn port of the HaMeR MANO cross-attention regression head.

Consumes the ViT feature map produced on device (token-major
``(B, 192, 1280)``) and emits the same flat ``(B, 157)`` regressor output
(16 flattened 3×3 rotation matrices + 10 shape + 3 weak-perspective camera)
that the torch reference produces.

Self- and cross-attention both use ``ttnn.transformer.scaled_dot_product_
attention`` — head_dim=64 is already tile-aligned so no head padding is
needed.  The 6-D → rotmat post-processing stays on the host because it's
sub-millisecond and uses cross-products / L2 normalization that aren't on
the NPU-critical path.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

try:
    import ttnn
except Exception:  # noqa: BLE001
    ttnn = None


HEAD_DIM = 64
NUM_HEADS = 8
INNER_DIM = HEAD_DIM * NUM_HEADS  # 512


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
    # Fuse the three regression projections (pose 96, shape 10, cam 3) into a
    # single (1024, 109) matmul + (109,) bias.  One device→host transfer at
    # the end instead of three.
    dec_w = torch.cat([head.decpose.weight, head.decshape.weight, head.deccam.weight], dim=0).t()
    dec_b = torch.cat([head.decpose.bias, head.decshape.bias, head.deccam.bias], dim=0)
    params: Dict[str, Any] = {
        "to_token_embedding": {
            "weight": _t(td.to_token_embedding.weight.t(), device),  # (1, 1024)
            "bias": _t(td.to_token_embedding.bias, device),          # (1024,)
        },
        "pos_embedding": _t(td.pos_embedding.squeeze(0), device),    # (1, 1024)
        "layers": [],
        "dec_w": _t(dec_w, device),
        "dec_b": _t(dec_b, device),
        "dec_split": (head.decpose.out_features, head.decshape.out_features, head.deccam.out_features),
    }
    for blk in td.layers:
        params["layers"].append({
            "norm_sa": {"weight": _t(blk.norm_sa.weight, device), "bias": _t(blk.norm_sa.bias, device)},
            "sa_qkv_w": _t(blk.sa.to_qkv.weight.t(), device),        # (1024, 3*512)
            "sa_out_w": _t(blk.sa.to_out.weight.t(), device),        # (512, 1024)
            "sa_out_b": _t(blk.sa.to_out.bias, device),
            "norm_ca": {"weight": _t(blk.norm_ca.weight, device), "bias": _t(blk.norm_ca.bias, device)},
            "ca_q_w": _t(blk.ca.to_q.weight.t(), device),            # (1024, 512)
            "ca_kv_w": _t(blk.ca.to_kv.weight.t(), device),          # (1280, 2*512)
            "ca_out_w": _t(blk.ca.to_out.weight.t(), device),        # (512, 1024)
            "ca_out_b": _t(blk.ca.to_out.bias, device),
            "norm_ff": {"weight": _t(blk.norm_ff.weight, device), "bias": _t(blk.norm_ff.bias, device)},
            "ff_fc1_w": _t(blk.ff.net[0].weight.t(), device),        # (1024, 1024)
            "ff_fc1_b": _t(blk.ff.net[0].bias, device),
            "ff_fc2_w": _t(blk.ff.net[2].weight.t(), device),        # (1024, 1024)
            "ff_fc2_b": _t(blk.ff.net[2].bias, device),
        })
    return params


def _split_qkv_sdpa(qkv, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM):
    """(B, N, 3·h·d) → triple of (B, h, N, d) for SDPA."""
    B, N, _ = qkv.shape
    qkv = ttnn.reshape(qkv, (B, N, 3, num_heads, head_dim))
    qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))
    return qkv[0], qkv[1], qkv[2]


def _split_qk_qv_sdpa(q, kv, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM):
    """Cross-attn variant: Q is already a single token tensor; KV is fused.

    Returns (q_h, k_h, v_h) all in (B, h, N, d) layout.
    """
    Bq, Nq, _ = q.shape
    q_h = ttnn.permute(ttnn.reshape(q, (Bq, Nq, num_heads, head_dim)), (0, 2, 1, 3))
    Bk, Nk, _ = kv.shape
    kv = ttnn.reshape(kv, (Bk, Nk, 2, num_heads, head_dim))
    kv = ttnn.permute(kv, (2, 0, 3, 1, 4))
    k_h = kv[0]
    v_h = kv[1]
    return q_h, k_h, v_h


def _merge_heads(ctx, num_heads: int = NUM_HEADS, head_dim: int = HEAD_DIM):
    B = ctx.shape[0]
    N = ctx.shape[2]
    out = ttnn.permute(ctx, (0, 2, 1, 3))
    return ttnn.reshape(out, (B, N, num_heads * head_dim))


def _decoder_block(x, context, p: Dict[str, Any]):
    # --- "self-attention": with N=1 query the softmax(q·kᵀ) is a scalar 1
    # and Attn(q,k,v) = v, so the only thing self-attn contributes is
    # ``sa_out(v)``.  We compute V directly from QKV and drop Q, K, the
    # scaled-dot-product, the softmax, and the @V matmul (≈ 4 ops).
    h = ttnn.layer_norm(x, weight=p["norm_sa"]["weight"], bias=p["norm_sa"]["bias"])
    qkv = h @ p["sa_qkv_w"]                                  # (B, 1, 3·512)
    B = qkv.shape[0]
    qkv = ttnn.reshape(qkv, (B, 1, 3, NUM_HEADS, HEAD_DIM))
    qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))                 # (3, B, h, 1, d)
    v = qkv[2]
    v = ttnn.permute(v, (0, 2, 1, 3))                        # (B, 1, h, d)
    v = ttnn.reshape(v, (B, 1, NUM_HEADS * HEAD_DIM))
    sa = v @ p["sa_out_w"]
    sa = sa + p["sa_out_b"]
    x = x + sa

    # --- cross-attention via SDPA ---
    h = ttnn.layer_norm(x, weight=p["norm_ca"]["weight"], bias=p["norm_ca"]["bias"])
    q = h @ p["ca_q_w"]                                  # (B, 1, 512)
    kv = context @ p["ca_kv_w"]                          # (B, 192, 1024)
    q_h, k_h, v_h = _split_qk_qv_sdpa(q, kv)
    ca = ttnn.transformer.scaled_dot_product_attention(q_h, k_h, v_h, is_causal=False, scale=HEAD_DIM ** -0.5)
    ca = _merge_heads(ca)                                # (B, 1, 512)
    ca = ca @ p["ca_out_w"]
    ca = ca + p["ca_out_b"]
    x = x + ca

    # --- FFN ---
    h = ttnn.layer_norm(x, weight=p["norm_ff"]["weight"], bias=p["norm_ff"]["bias"])
    h = h @ p["ff_fc1_w"]
    h = h + p["ff_fc1_b"]
    h = ttnn.gelu(h)
    h = h @ p["ff_fc2_w"]
    h = h + p["ff_fc2_b"]
    return x + h


def _rot6d_to_rotmat_torch(x6: torch.Tensor) -> torch.Tensor:
    """Host-side post-processing — sub-millisecond, kept off the NPU."""
    x = x6.reshape(-1, 3, 2)
    b1 = F.normalize(x[..., 0], dim=-1)
    b2 = F.normalize(x[..., 1] - (b1 * x[..., 1]).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def forward_device(
    feature_tokens,
    params: Dict[str, Any],
    device: Any,
    depth: int = 6,
    cached_token: Tuple[Any, ...] = (),
):
    """Device-only MANO head forward.  Returns the on-device ``dec`` tensor
    of shape ``(B, 1, 109_padded)`` — the caller is responsible for copying
    it back to host and applying ``init_hand_pose / init_betas / init_cam``
    plus the rot-6d → rotmat conversion.

    Split out so the trace-capture path can record only device ops.
    """
    B = feature_tokens.shape[0]
    if cached_token and cached_token[0].shape[0] == B:
        token = cached_token[0]
    else:
        token = ttnn.from_torch(
            torch.zeros(B, 1, 1, dtype=torch.bfloat16),
            device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )
    x = token @ params["to_token_embedding"]["weight"]
    x = x + params["to_token_embedding"]["bias"]
    x = x + params["pos_embedding"]
    for i in range(depth):
        x = _decoder_block(x, feature_tokens, params["layers"][i])
    dec = x @ params["dec_w"]
    dec = dec + params["dec_b"]
    return dec


def host_finalize(
    dec_host: torch.Tensor,
    params_dec_split: Tuple[int, int, int],
    init_hand_pose: torch.Tensor,
    init_betas: torch.Tensor,
    init_cam: torch.Tensor,
    num_hand_joints: int = 15,
) -> torch.Tensor:
    """Apply the host-side post-processing (init-param add, 6D→rotmat
    conversion, flatten) to a ``dec`` tensor that was just copied back."""
    B = dec_host.shape[0]
    np_pose, np_shape, np_cam = params_dec_split
    pose_h = dec_host[:, :np_pose] + init_hand_pose
    betas_h = dec_host[:, np_pose : np_pose + np_shape] + init_betas
    cam_h = dec_host[:, np_pose + np_shape : np_pose + np_shape + np_cam] + init_cam
    rotmats = _rot6d_to_rotmat_torch(pose_h).view(B, num_hand_joints + 1, 3, 3)
    return torch.cat([rotmats.reshape(B, -1), betas_h, cam_h], dim=-1)


def forward(
    feature_tokens,
    params: Dict[str, Any],
    device: Any,
    init_hand_pose: torch.Tensor,
    init_betas: torch.Tensor,
    init_cam: torch.Tensor,
    num_hand_joints: int = 15,
    depth: int = 6,
    cached_token: Tuple[Any, ...] = (),
) -> torch.Tensor:
    """MANO head forward on device → returns (B, 157) flat regression result.

    ``feature_tokens`` is ``(B, 192, 1280)`` already on device.  ``cached_token``
    can be a length-1 tuple holding a previously-uploaded zero query token
    (per-batch reuse) — a fresh upload is allocated when empty.
    """
    B = feature_tokens.shape[0]

    # Zero query token (B, 1, 1).  Caller can cache one for reuse.
    if cached_token and cached_token[0].shape[0] == B:
        token = cached_token[0]
    else:
        token = ttnn.from_torch(
            torch.zeros(B, 1, 1, dtype=torch.bfloat16),
            device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        )

    x = token @ params["to_token_embedding"]["weight"]
    x = x + params["to_token_embedding"]["bias"]
    x = x + params["pos_embedding"]

    for i in range(depth):
        x = _decoder_block(x, feature_tokens, params["layers"][i])

    # Single fused regression matmul → one device→host transfer.
    dec = x @ params["dec_w"]                              # (B, 1, 109_padded)
    dec = dec + params["dec_b"]
    dec_h = ttnn.to_torch(dec).to(torch.float32).reshape(B, -1)
    np_pose, np_shape, np_cam = params["dec_split"]
    pose_h = dec_h[:, :np_pose] + init_hand_pose
    betas_h = dec_h[:, np_pose : np_pose + np_shape] + init_betas
    cam_h = dec_h[:, np_pose + np_shape : np_pose + np_shape + np_cam] + init_cam

    rotmats = _rot6d_to_rotmat_torch(pose_h).view(B, num_hand_joints + 1, 3, 3)
    return torch.cat([rotmats.reshape(B, -1), betas_h, cam_h], dim=-1)  # (B, 157)
