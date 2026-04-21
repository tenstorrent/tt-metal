# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load pretrained Gaze-LLE weights into the reference model.

The official Gaze-LLE checkpoint at
  https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt
contains only the gaze decoder weights. The DINOv2 backbone comes from
facebookresearch/dinov2 torch.hub and is available as a direct download at
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

Notes:
  * DINOv2 pretrained pos_embed is for 518x518 inputs (1 CLS + 37*37=1370 total);
    we interpolate to 32*32+1=1025 for our 448x448 runtime.
  * LayerScale attribute is named ``gamma`` in both our reference and the
    checkpoint (``blocks.*.ls{1,2}.gamma``).
  * ``mask_token`` and position-related keys that don't map are ignored.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _interpolate_pos_embed_2d(
    pretrained_pos_embed: torch.Tensor,  # (1, 1 + src*src, D)
    target_side: int,                     # e.g. 32 for 448/14
) -> torch.Tensor:
    """Resize a DINOv2 1-CLS + patch-grid pos_embed to a new patch-grid side length."""
    cls_pe = pretrained_pos_embed[:, :1]
    patch_pe = pretrained_pos_embed[:, 1:]
    src_side = int(math.isqrt(patch_pe.shape[1]))
    assert src_side * src_side == patch_pe.shape[1], "patch pos_embed must be square"
    embed_dim = patch_pe.shape[-1]
    patch_pe = patch_pe.reshape(1, src_side, src_side, embed_dim).permute(0, 3, 1, 2)  # (1, D, s, s)
    patch_pe = F.interpolate(patch_pe, size=(target_side, target_side), mode="bicubic", align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, target_side * target_side, embed_dim)
    return torch.cat([cls_pe, patch_pe], dim=1)


def load_dinov2_into_backbone(backbone, dinov2_state_dict: dict, verbose: bool = False):
    """Load the stripped DINOv2 state dict into our reference ``DinoV2Backbone``.

    Keys in the official pretrained checkpoint:
        cls_token, pos_embed, mask_token, patch_embed.proj.{weight,bias},
        blocks.{i}.{norm{1,2}.{weight,bias}, attn.{qkv,proj}.{weight,bias},
                     ls{1,2}.gamma, mlp.fc{1,2}.{weight,bias}}, norm.{weight,bias}
    """
    own = dict(backbone.named_parameters())
    own_buf = dict(backbone.named_buffers())
    target_side = backbone.img_size // backbone.patch_size

    load_report = {"loaded": 0, "interp": 0, "skipped": [], "missing": []}
    remapped = {}

    for k, v in dinov2_state_dict.items():
        if k == "mask_token":
            load_report["skipped"].append(k)
            continue
        if k == "pos_embed":
            expected = 1 + target_side * target_side
            if v.shape[1] != expected:
                v = _interpolate_pos_embed_2d(v, target_side)
                load_report["interp"] += 1
        # Our conv weight is registered as `patch_embed_proj.weight`; checkpoint is `patch_embed.proj.weight`.
        if k.startswith("patch_embed.proj."):
            k = k.replace("patch_embed.proj.", "patch_embed_proj.")
        remapped[k] = v

    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    load_report["missing"] = list(missing)
    load_report["unexpected"] = list(unexpected)
    load_report["loaded"] = len(remapped) - len(unexpected)
    if verbose:
        print(f"DINOv2 backbone: loaded {load_report['loaded']} params, "
              f"interpolated pos_embed: {load_report['interp']}, "
              f"missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    return load_report


def load_gaze_lle_into_reference(
    ref_model,
    dinov2_state_dict: dict,
    gaze_state_dict: dict,
    verbose: bool = False,
) -> dict:
    """Populate ref_model.backbone from DINOv2 and the remaining params from the gaze ckpt.

    Gaze checkpoint keys (45 total, prefixless — matching ref_model attributes):
        pos_embed, linear.{weight,bias}, head_token.weight, inout_token.weight,
        transformer.{i}.norm{1,2}.{weight,bias},
        transformer.{i}.attn.{qkv.{weight,bias},proj.{weight,bias}},
        transformer.{i}.mlp.fc{1,2}.{weight,bias},
        heatmap_head.{0.{weight,bias}, 1.weight},
        inout_head.{0.{weight,bias}, 3.{weight,bias}}
    """
    bb_report = load_dinov2_into_backbone(ref_model.backbone, dinov2_state_dict, verbose=verbose)

    # Gaze decoder weights map directly onto ref_model state_dict.
    target_state = ref_model.state_dict()
    remapped = {}
    for k, v in gaze_state_dict.items():
        if k in target_state:
            remapped[k] = v
        else:
            # Gaze checkpoint may use keys without backbone prefix
            if f"backbone.{k}" in target_state:
                remapped[f"backbone.{k}"] = v

    missing, unexpected = ref_model.load_state_dict(remapped, strict=False)
    decoder_report = {
        "decoder_loaded": len(remapped) - len(unexpected),
        "decoder_missing": [m for m in missing if not m.startswith("backbone.")],
        "decoder_unexpected": list(unexpected),
    }
    if verbose:
        dm = decoder_report["decoder_missing"]
        print(f"Gaze decoder: loaded {decoder_report['decoder_loaded']} params, "
              f"decoder-side missing: {len(dm)}, unexpected: {len(decoder_report['decoder_unexpected'])}")
        if dm:
            print("  missing:", dm[:10], "..." if len(dm) > 10 else "")
    return {"backbone": bb_report, "decoder": decoder_report}


def load_pretrained(
    ref_model,
    dinov2_path: str = "/home/ttuser/experiments/gaze-lle/weights/dinov2_vitb14_pretrain.pth",
    gaze_path: Optional[str] = "/home/ttuser/experiments/gaze-lle/weights/gazelle_dinov2_vitb14_inout.pt",
    verbose: bool = True,
):
    """Convenience wrapper: loads both checkpoints into ``ref_model`` in place."""
    dinov2_sd = torch.load(dinov2_path, map_location="cpu", weights_only=False)
    gaze_sd = torch.load(gaze_path, map_location="cpu", weights_only=False) if gaze_path else {}
    return load_gaze_lle_into_reference(ref_model, dinov2_sd, gaze_sd, verbose=verbose)
