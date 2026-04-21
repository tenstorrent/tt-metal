# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-stage (relative) PCC test for the Gaze-LLE TT-NN forward.

Runs the reference torch model in a "shadow forward" that mirrors the exact op
sequence of :class:`TtGazeLLE.__call__`, and compares intermediate tensors to
the tt output at each of the following stages. Random weights are used; since
both paths consume the same weights, matching torch values prove the tt
implementation is algebraically equivalent up to bf16 / bfp8 / LoFi precision.

Stages captured:

    patch_embed                 — after the (588,768) patch matmul + bias
    after_prefix                — after pos_patches add + [CLS+pos_cls,REG] prepend
    after_block_0,5,11          — DINOv2 encoder depth samples (first, mid, last)
    after_final_norm            — DINOv2 final LayerNorm
    after_slice                 — CLS+REG dropped, patch tokens only
    after_gaze_proj_pos         — 768→256 proj + gaze pos_embed
    head_map                    — bbox-derived binary mask built on device
    after_head_conditioning     — head_map * head_token added to patch tokens
    after_gaze_blocks           — output of the 3 gaze decoder blocks
    heatmap_compact             — fused ConvT + Conv + sigmoid (pre-view-reshape)
    inout_scalar                — in/out MLP output after sigmoid
    heatmap                     — final (B, 64, 64) heatmap after host reshape
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.experimental.gaze_lle.reference.torch_gaze_lle import build_gaze_lle
from models.experimental.gaze_lle.tt.tt_gaze_lle import TtGazeLLE


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient of two tensors (flattened).

    PCC is undefined for single-element tensors (variance 0); for those we
    fall back to 1 - |a - b| / max(|a|, |b|, 1e-6) so we still get a
    [0, 1]-like score that's 1.0 for equal scalars and 0.0 for opposite signs.
    """
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() <= 1:
        if a.numel() == 0:
            return 1.0
        denom = max(abs(a.item()), abs(b.item()), 1e-6)
        return float(max(0.0, 1.0 - abs(a.item() - b.item()) / denom))
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-12
    return float((a @ b).item() / denom)


def _torch_dinov2_block(x: torch.Tensor, blk) -> torch.Tensor:
    # Mirrors _dinov2_attention + _dinov2_mlp (LayerScale folded into proj/fc2).
    h = blk.norm1(x)
    h = blk.attn(h)
    h = blk.ls1(h)  # torch reference still has explicit LayerScale; tt folds it
    x = x + h
    h = blk.norm2(x)
    h = blk.mlp(h)
    h = blk.ls2(h)
    return x + h


def _torch_gaze_block(x: torch.Tensor, blk) -> torch.Tensor:
    h = blk.attn(blk.norm1(x))
    x = x + h
    h = blk.mlp(blk.norm2(x))
    return x + h


def _torch_shadow_forward(ref_model, images: torch.Tensor, bboxes, patch_size: int,
                          num_patches_side: int, num_reg_tokens: int):
    """Torch forward that matches the tt op sequence exactly."""
    b = images.shape[0]
    backbone = ref_model.backbone
    stages = {}

    # ---- Patch embed via matmul (same weight arrangement as tt path).
    patches = (
        images.view(b, 3, num_patches_side, patch_size, num_patches_side, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(b, num_patches_side * num_patches_side, patch_size * patch_size * 3)
    )
    w_flat = backbone.patch_embed_proj.weight.permute(2, 3, 1, 0).reshape(-1, backbone.embed_dim)
    patches_proj = patches @ w_flat + backbone.patch_embed_proj.bias
    stages["patch_embed"] = patches_proj.clone()

    # ---- Add patch pos_embed, prepend [CLS+pos_cls, REG].
    patches_proj = patches_proj + backbone.pos_embed[:, 1:]
    cls_plus_pos = backbone.cls_token + backbone.pos_embed[:, :1]
    prefix = torch.cat([cls_plus_pos, backbone.reg_token], dim=1).expand(b, -1, -1)
    x = torch.cat([prefix, patches_proj], dim=1)
    stages["after_prefix"] = x.clone()

    # ---- DINOv2 backbone.
    for i, blk in enumerate(backbone.blocks):
        x = _torch_dinov2_block(x, blk)
        if i in (0, 5, 11):
            stages[f"after_block_{i}"] = x.clone()
    x = backbone.norm(x)
    stages["after_final_norm"] = x.clone()

    # ---- Slice CLS + REG.
    x = x[:, 1 + num_reg_tokens :]
    stages["after_slice"] = x.clone()

    # ---- Gaze decoder proj (1x1 conv as matmul).
    lin_w = ref_model.linear.weight.squeeze(-1).squeeze(-1).T  # (embed_dim, dim)
    x = x @ lin_w + ref_model.linear.bias
    # Gaze pos_embed (dim, H, W) → (1, H*W, dim)
    pe = ref_model.pos_embed.permute(1, 2, 0).reshape(1, -1, ref_model.dim)
    x = x + pe
    stages["after_gaze_proj_pos"] = x.clone()

    # ---- Head-map from bbox (matching the ge/lt/mul cascade).
    fh, fw = ref_model.featmap_h, ref_model.featmap_w
    bbox = bboxes[0]
    xmin_pix = round(bbox[0] * fw)
    ymin_pix = round(bbox[1] * fh)
    xmax_pix = round(bbox[2] * fw)
    ymax_pix = round(bbox[3] * fh)
    idx_h = torch.arange(fh).view(1, fh, 1).float()
    idx_w = torch.arange(fw).view(1, 1, fw).float()
    h_mask = ((idx_h >= ymin_pix).float() * (idx_h < ymax_pix).float())  # (1, fh, 1)
    w_mask = ((idx_w >= xmin_pix).float() * (idx_w < xmax_pix).float())  # (1, 1, fw)
    head_map = (h_mask * w_mask).view(b, fh * fw, 1)
    stages["head_map"] = head_map.clone()

    head_contrib = head_map * ref_model.head_token.weight.unsqueeze(0)
    x = x + head_contrib
    stages["after_head_conditioning"] = x.clone()

    # ---- In/out token + 3 gaze blocks.
    inout_tok = ref_model.inout_token.weight.unsqueeze(0).expand(b, -1, -1)
    x = torch.cat([inout_tok, x], dim=1)
    for blk in ref_model.transformer:
        x = _torch_gaze_block(x, blk)
    stages["after_gaze_blocks"] = x.clone()

    # ---- In/out head.
    inout_logits = ref_model.inout_head[2](F.relu(ref_model.inout_head[0](x[:, 0, :])))
    inout_pred = torch.sigmoid(inout_logits).reshape(b)
    stages["inout_scalar"] = inout_pred.clone()

    # ---- Fused heatmap head.
    patch_out = x[:, 1:, :]
    ct_w = ref_model.heatmap_head[0].weight  # (in, out, 2, 2)
    ct_b = ref_model.heatmap_head[0].bias
    c1_w = ref_model.heatmap_head[1].weight.squeeze(-1).squeeze(-1)  # (1, 256)
    w_fused = torch.einsum("ko,ioab->ikab", c1_w, ct_w).squeeze(1).reshape(ref_model.dim, 4)
    b_fused = (c1_w @ ct_b).squeeze()
    hm_compact = torch.sigmoid(patch_out @ w_fused + b_fused)
    stages["heatmap_compact"] = hm_compact.clone()

    heatmap = (
        hm_compact.view(b, ref_model.featmap_h, ref_model.featmap_w, 2, 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, ref_model.featmap_h * 2, ref_model.featmap_w * 2)
    )
    stages["heatmap"] = heatmap

    return stages


# PCC thresholds per stage — loose where bfp8/LoFi is known to bite the hardest
# (attention & MLP inside long backbones), tight where no lossy math happens.
_STAGE_THRESHOLDS = {
    "patch_embed":              0.999,
    "after_prefix":             0.999,
    "after_block_0":            0.99,
    "after_block_5":            0.98,
    "after_block_11":           0.95,   # cumulative bf16/bfp8/LoFi drift worst here
    "after_final_norm":         0.95,
    "after_slice":              0.95,
    "after_gaze_proj_pos":      0.95,
    "head_map":                 0.9999, # binary mask, identical
    "after_head_conditioning":  0.95,
    "after_gaze_blocks":        0.95,
    "heatmap_compact":          0.95,
    "inout_scalar":             0.90,   # single scalar, very sensitive
    "heatmap":                  0.95,
}


@pytest.mark.parametrize("variant", ["vitb14"])
def test_relative_pcc(device, variant, capsys):
    """Assert per-stage PCC ≥ stage-specific thresholds."""
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    ref = build_gaze_lle(variant=variant, inout=True).eval()
    tt_model = TtGazeLLE(ref, device, inout=True)

    image = torch.randn(1, 3, 448, 448)
    bboxes = [(0.3, 0.2, 0.6, 0.5)]

    # Run torch shadow forward.
    torch_stages = _torch_shadow_forward(
        ref, image, bboxes,
        patch_size=ref.backbone.patch_size,
        num_patches_side=tt_model.num_patches_side,
        num_reg_tokens=tt_model.num_reg_tokens,
    )

    # Warm up tt model (populate program cache) then capture.
    _ = tt_model(image, bboxes)
    tt_stages = {}
    _ = tt_model(image, bboxes, captures=tt_stages)

    # Verify shapes match and PCCs pass.
    with capsys.disabled():
        print("\n{:30s} {:>7s} {:>20s} {:>20s}".format("stage", "pcc", "shape (torch)", "shape (tt)"))
    failures = []
    for key, thresh in _STAGE_THRESHOLDS.items():
        assert key in torch_stages, f"torch shadow missing stage {key!r}"
        assert key in tt_stages, f"tt captures missing stage {key!r}"
        t = torch_stages[key]
        d = tt_stages[key]
        # Flatten for PCC; shapes can differ in tile padding but element count must match
        # up to trailing zeros.
        n = t.numel()
        d_flat = d.flatten()
        if d_flat.numel() < n:
            d_flat = F.pad(d_flat, (0, n - d_flat.numel()))
        else:
            d_flat = d_flat[:n]
        pcc = _pcc(t.flatten(), d_flat)
        mark = "OK " if pcc >= thresh else "BAD"
        with capsys.disabled():
            print(f"  {mark} {key:26s} {pcc:>7.4f} {str(tuple(t.shape)):>20s} {str(tuple(d.shape)):>20s}")
        if pcc < thresh:
            failures.append(f"{key}: PCC {pcc:.4f} < threshold {thresh:.4f}")

    assert not failures, "Stage PCC failures:\n  " + "\n  ".join(failures)
