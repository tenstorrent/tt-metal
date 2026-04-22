# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional tt-nn port of the HaMeR ViT-H backbone.

Follows the pattern from ``models/demos/vision/classification/vit/common/tt/
ttnn_functional_vit.py`` — each sub-op is a free function that takes tt-nn
tensors plus a ``parameters`` dict and returns tt-nn tensors.  HaMeR-specific
adjustments vs. the HF-ViT demo:

- A single fused ``qkv`` linear (HaMeR uses ``nn.Linear(dim, dim*3)`` not
  three separate Q/K/V projections).
- No CLS token — positional embedding adds ``pos_embed[:, 1:]`` and
  ``pos_embed[:, :1]`` separately to every patch token.
- PatchEmbed uses ``Conv2d(kernel=16, stride=16, pad=4)`` producing a
  ``16×12`` grid for ``256×192`` input.  On-device we use a ``ttnn.fold``
  → matmul decomposition equivalent to the HF-ViT path but with the
  HaMeR-specific padding pre-baked into the torch-side weight layout.

Inference only; weights are converted on the host via
``build_parameters_from_reference``.  Runs on a single Blackhole p150.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

try:
    import ttnn
except Exception:  # noqa: BLE001
    ttnn = None  # allow import without the runtime; callers guard on ``ttnn is not None``


# --------------------------------------------------------------------------- #
# Weight preparation (host side)
# --------------------------------------------------------------------------- #
def _t(weight: torch.Tensor, device: Any) -> Any:
    """Convert a torch Tensor → tt-nn bfloat16 TILE_LAYOUT tensor on device."""
    return ttnn.from_torch(
        weight.to(torch.bfloat16).contiguous(),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def _patch_embed_weight(conv: torch.nn.Conv2d) -> torch.Tensor:
    """Flatten a 2-D patch-embedding conv to a matmul weight.

    HaMeR's PatchEmbed conv is (embed_dim=1280, in=3, k=16, k=16).  After
    padding+unfold, each 16×16×3 patch becomes a 768-d input vector; the
    equivalent matmul weight is (768, 1280).
    """
    w = conv.weight  # (embed_dim, in_chans, k, k)
    embed_dim, in_chans, kh, kw = w.shape
    # (in, kh, kw, embed) → (in·kh·kw, embed)
    w = w.permute(1, 2, 3, 0).reshape(in_chans * kh * kw, embed_dim)
    return w.contiguous()


def build_parameters_from_reference(ref, device: Any) -> Dict[str, Any]:
    """Map the torch reference ``Hamer`` module's state into tt-nn parameters.

    ``ref`` is the torch reference defined in
    ``models.experimental.dyn_hamr.reference.hamer``.  This converts every
    backbone weight to bfloat16 tiles and returns a nested dict matching the
    structure the functional ops below consume.

    Patch embedding stays CPU-resident — it's one Conv2d that contributes
    sub-percent latency but has awkward asymmetric padding (256×192 with
    pad=4) that doesn't decompose cleanly into ``ttnn.fold``.  We push the
    already-flattened patch tokens to the device instead.
    """
    assert ttnn is not None, "ttnn runtime not available"
    backbone = ref.backbone
    params: Dict[str, Any] = {
        "pos_embed": _t(backbone.pos_embed.squeeze(0), device),  # (N+1, C)
        "last_norm": {
            "weight": _t(backbone.last_norm.weight, device),
            "bias": _t(backbone.last_norm.bias, device),
        },
        "blocks": [],
    }
    for blk in backbone.blocks:
        params["blocks"].append({
            "norm1": {"weight": _t(blk.norm1.weight, device), "bias": _t(blk.norm1.bias, device)},
            "attn": {
                "qkv_w": _t(blk.attn.qkv.weight.t(), device),
                "qkv_b": _t(blk.attn.qkv.bias, device),
                "proj_w": _t(blk.attn.proj.weight.t(), device),
                "proj_b": _t(blk.attn.proj.bias, device),
            },
            "norm2": {"weight": _t(blk.norm2.weight, device), "bias": _t(blk.norm2.bias, device)},
            "mlp": {
                "fc1_w": _t(blk.mlp.fc1.weight.t(), device),
                "fc1_b": _t(blk.mlp.fc1.bias, device),
                "fc2_w": _t(blk.mlp.fc2.weight.t(), device),
                "fc2_b": _t(blk.mlp.fc2.bias, device),
            },
        })
    return params


# --------------------------------------------------------------------------- #
# Functional ops
# --------------------------------------------------------------------------- #
def patch_embed(pixel_values, params: Dict[str, Any], pad: int = 4, patch: int = 16):
    """HaMeR patch embedding on device.  Expects NHWC input already on device."""
    # Host-side padding keeps this op decomposition trivial on-device: we
    # unfold 16×16×3 patches and project with a matmul.  Upstream uses a
    # Conv2d(pad=4) which is the same as padding the image by 4 on each side
    # first and then striding by 16 with zero pad.
    x = pixel_values
    # fold: (B, Hp, Wp, 16·16·3)
    x = ttnn.fold(x, stride_h=patch, stride_w=1)
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = x @ params["weight"]
    x = x + params["bias"]
    return x


def _split_qkv(qkv, num_heads: int, head_dim: int):
    """Split fused (B, N, 3·h·d) into ((B, h, N, d), (B, h, d, N), (B, h, N, d)).

    Avoids ``ttnn.transformer.split_query_key_value_and_split_heads`` because
    that helper asserts head_dim is a multiple of TILE_WIDTH (32), which
    HaMeR's 80-d heads are not.  Manual reshape+permute lets the tile-padded
    matmul handle the misaligned head dim transparently.
    """
    B, N, _ = qkv.shape
    qkv = ttnn.reshape(qkv, (B, N, 3, num_heads, head_dim))
    qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))  # (3, B, h, N, d)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]
    k = ttnn.permute(k, (0, 1, 3, 2))  # (B, h, d, N) for q @ k
    return q, k, v


def _merge_heads(ctx, num_heads: int, head_dim: int):
    B = ctx.shape[0]
    N = ctx.shape[2]
    out = ttnn.permute(ctx, (0, 2, 1, 3))  # (B, N, h, d)
    return ttnn.reshape(out, (B, N, num_heads * head_dim))


def block(hidden, block_params: Dict[str, Any], num_heads: int, head_dim: int):
    """One ViT-H transformer block (pre-norm, fused qkv, MLP with GELU)."""
    # --- self-attention ---
    h = ttnn.layer_norm(hidden, weight=block_params["norm1"]["weight"], bias=block_params["norm1"]["bias"])
    qkv = h @ block_params["attn"]["qkv_w"]
    qkv = qkv + block_params["attn"]["qkv_b"]

    q, k, v = _split_qkv(qkv, num_heads, head_dim)
    scale = head_dim ** -0.5
    attn = q @ k
    attn = attn * scale
    attn = ttnn.softmax(attn, dim=-1)
    ctx = attn @ v
    ctx = _merge_heads(ctx, num_heads, head_dim)

    attn_out = ctx @ block_params["attn"]["proj_w"]
    attn_out = attn_out + block_params["attn"]["proj_b"]
    hidden = hidden + attn_out

    # --- MLP ---
    h = ttnn.layer_norm(hidden, weight=block_params["norm2"]["weight"], bias=block_params["norm2"]["bias"])
    h = h @ block_params["mlp"]["fc1_w"]
    h = h + block_params["mlp"]["fc1_b"]
    h = ttnn.gelu(h)
    h = h @ block_params["mlp"]["fc2_w"]
    h = h + block_params["mlp"]["fc2_b"]
    hidden = hidden + h
    return hidden


def forward(
    patch_tokens,
    params: Dict[str, Any],
    num_heads: int = 16,
    head_dim: int = 80,
    depth: int = 32,
):
    """ViT-H body forward.

    ``patch_tokens`` is a tt-nn tensor of shape ``(B, 192, 1280)`` carrying the
    *already-projected* patch embeddings (the CPU does the asymmetric Conv2d
    patch projection for us — that's a one-shot, sub-percent op that doesn't
    decompose cleanly to ``ttnn.fold``).  Output is the post-LN feature map at
    ``(B, 192, 1280)`` token-major.
    """
    x = patch_tokens
    pos = params["pos_embed"]
    # HaMeR adds pos_embed[:, 1:] + pos_embed[:, :1] to every token; we
    # collapsed pos_embed to (193, C) at upload, so reuse the same split.
    pos_patches = pos[1:]
    pos_cls = pos[:1]
    x = x + pos_patches
    x = x + pos_cls

    for i in range(depth):
        x = block(x, params["blocks"][i], num_heads=num_heads, head_dim=head_dim)

    x = ttnn.layer_norm(
        x,
        weight=params["last_norm"]["weight"],
        bias=params["last_norm"]["bias"],
    )
    return x  # (B, 192, 1280), token-major
