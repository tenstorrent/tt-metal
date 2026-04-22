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


# --- head-dim padding (80 → 96) so SDPA / FlashAttention is usable ---------- #
HEAD_DIM_RAW = 80     # HaMeR ViT-H native head dim
HEAD_DIM_PAD = 96     # next multiple of TILE_WIDTH=32 — required by SDPA


def _pad_qkv_weight(qkv_w_t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Pad fused QKV weight so each head has ``HEAD_DIM_PAD`` cols.

    Input layout: ``(in_dim, 3*h*d_raw)``.
    Output layout: ``(in_dim, 3*h*d_pad)`` with the new cols set to zero so
    they contribute nothing to attention.
    """
    in_dim = qkv_w_t.shape[0]
    w = qkv_w_t.view(in_dim, 3, num_heads, HEAD_DIM_RAW)
    pad = torch.zeros(in_dim, 3, num_heads, HEAD_DIM_PAD - HEAD_DIM_RAW, dtype=w.dtype)
    return torch.cat([w, pad], dim=-1).reshape(in_dim, 3 * num_heads * HEAD_DIM_PAD).contiguous()


def _pad_qkv_bias(qkv_b: torch.Tensor, num_heads: int) -> torch.Tensor:
    b = qkv_b.view(3, num_heads, HEAD_DIM_RAW)
    pad = torch.zeros(3, num_heads, HEAD_DIM_PAD - HEAD_DIM_RAW, dtype=b.dtype)
    return torch.cat([b, pad], dim=-1).reshape(3 * num_heads * HEAD_DIM_PAD).contiguous()


def _pad_proj_weight(proj_w_t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Pad attention output-projection weight rows so the merged padded heads
    feed straight into ``proj`` without an explicit slice — the new rows are
    zero, so they don't contribute to the dot product.

    Input layout: ``(h*d_raw, out_dim)``.
    Output layout: ``(h*d_pad, out_dim)``.
    """
    out_dim = proj_w_t.shape[1]
    w = proj_w_t.view(num_heads, HEAD_DIM_RAW, out_dim)
    pad = torch.zeros(num_heads, HEAD_DIM_PAD - HEAD_DIM_RAW, out_dim, dtype=w.dtype)
    return torch.cat([w, pad], dim=1).reshape(num_heads * HEAD_DIM_PAD, out_dim).contiguous()


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
    num_heads = backbone.blocks[0].attn.num_heads
    for blk in backbone.blocks:
        qkv_w_padded = _pad_qkv_weight(blk.attn.qkv.weight.t(), num_heads)
        qkv_b_padded = _pad_qkv_bias(blk.attn.qkv.bias, num_heads)
        proj_w_padded = _pad_proj_weight(blk.attn.proj.weight.t(), num_heads)
        params["blocks"].append({
            "norm1": {"weight": _t(blk.norm1.weight, device), "bias": _t(blk.norm1.bias, device)},
            "attn": {
                "qkv_w": _t(qkv_w_padded, device),
                "qkv_b": _t(qkv_b_padded, device),
                "proj_w": _t(proj_w_padded, device),
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


def _split_qkv_sdpa(qkv, num_heads: int):
    """Split padded (B, N, 3·h·HEAD_DIM_PAD) → triple of (B, h, N, HEAD_DIM_PAD).

    Layout consumed by ``ttnn.transformer.scaled_dot_product_attention``.
    """
    B, N, _ = qkv.shape
    qkv = ttnn.reshape(qkv, (B, N, 3, num_heads, HEAD_DIM_PAD))
    qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))  # (3, B, h, N, d_pad)
    return qkv[0], qkv[1], qkv[2]


def _merge_heads(ctx, num_heads: int):
    B = ctx.shape[0]
    N = ctx.shape[2]
    out = ttnn.permute(ctx, (0, 2, 1, 3))  # (B, N, h, d_pad)
    return ttnn.reshape(out, (B, N, num_heads * HEAD_DIM_PAD))


def block(hidden, block_params: Dict[str, Any], num_heads: int, head_dim: int):
    """One ViT-H transformer block (pre-norm, fused qkv, FlashAttention-2, MLP).

    Attention runs through ``ttnn.transformer.scaled_dot_product_attention``
    on tile-aligned heads (``HEAD_DIM_PAD = 96``).  Padded cols are zeroed in
    the upload-time weight transform so the math is exact.  ``head_dim`` arg
    kept in the signature for backward compatibility but is unused.
    """
    del head_dim  # padded head dim handled internally
    # --- self-attention via FlashAttention-2 ---
    h = ttnn.layer_norm(hidden, weight=block_params["norm1"]["weight"], bias=block_params["norm1"]["bias"])
    qkv = h @ block_params["attn"]["qkv_w"]
    qkv = qkv + block_params["attn"]["qkv_b"]

    q, k, v = _split_qkv_sdpa(qkv, num_heads)
    ctx = ttnn.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=HEAD_DIM_RAW ** -0.5,
    )
    ctx = _merge_heads(ctx, num_heads)

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
