# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test ED-Pose backbone (Swin-L + input_proj + position encoding) on CPU.
Builds a reference Swin + input_proj from the checkpoint directly (no full ED-Pose model)
and compares against our wrapper.

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_backbone_test.py
"""

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import (
    EDPoseBackbone,
    PositionEmbeddingSineHW,
)


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


def _import_swin_transformer():
    """Import swin_transformer module directly to avoid ED-Pose __init__.py chain."""
    if EDPOSE_ROOT not in sys.path:
        sys.path.insert(0, EDPOSE_ROOT)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "swin_transformer",
        os.path.join(EDPOSE_ROOT, "models", "edpose", "backbones", "swin_transformer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_reference_swin_and_input_proj(full_sd):
    """Build Swin-L + input_proj from checkpoint without full ED-Pose import chain."""
    swin_mod = _import_swin_transformer()
    build_swin_transformer = swin_mod.build_swin_transformer

    swin = build_swin_transformer(
        "swin_L_384_22k",
        pretrain_img_size=384,
        out_indices=(0, 1, 2, 3),
        dilation=False,
        use_checkpoint=False,
    )
    swin_sd = {k[len("backbone.0."):]: v for k, v in full_sd.items() if k.startswith("backbone.0.")}
    swin.load_state_dict(swin_sd, strict=False)
    swin.eval()

    backbone_channels = swin.num_features  # [192, 384, 768, 1536]
    d_model = 256
    num_feature_levels = 5

    input_proj = nn.ModuleList()
    for i in range(len(backbone_channels)):
        input_proj.append(nn.Sequential(
            nn.Conv2d(backbone_channels[i], d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
        ))
    for i in range(num_feature_levels - len(backbone_channels)):
        in_ch = backbone_channels[-1] if i == 0 else d_model
        input_proj.append(nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model),
        ))

    proj_sd = {k[len("input_proj."):]: v for k, v in full_sd.items() if k.startswith("input_proj.")}
    input_proj.load_state_dict(proj_sd, strict=True)
    input_proj.eval()

    level_embed = full_sd.get("transformer.level_embed", None)

    pos_embed = PositionEmbeddingSineHW(num_pos_feats=128, temperatureH=20, temperatureW=20, normalize=True)

    return swin, input_proj, pos_embed, level_embed


def run_reference_pipeline(swin, input_proj, pos_embed, level_embed, image_tensor, mask):
    """Run Swin + input_proj + pos + level_embed to produce flattened encoder inputs."""
    sys.path.insert(0, EDPOSE_ROOT) if EDPOSE_ROOT not in sys.path else None
    from util.misc import NestedTensor

    nested = NestedTensor(image_tensor, mask)
    features_dict = swin(nested)

    srcs = []
    masks_list = []
    poss = []

    for lvl in range(len(features_dict)):
        feat = features_dict[lvl]
        src_l, mask_l = feat.tensors, feat.mask
        srcs.append(input_proj[lvl](src_l))
        masks_list.append(mask_l)
        poss.append(pos_embed(src_l, mask_l).to(src_l.dtype))

    num_feature_levels = 5
    if num_feature_levels > len(features_dict):
        _len_srcs = len(features_dict)
        for lvl in range(_len_srcs, num_feature_levels):
            if lvl == _len_srcs:
                src_extra = input_proj[lvl](features_dict[_len_srcs - 1].tensors)
            else:
                src_extra = input_proj[lvl](srcs[-1])

            m = mask
            mask_extra = F.interpolate(m[None].float(), size=src_extra.shape[-2:]).to(torch.bool)[0]
            pos_extra = pos_embed(src_extra, mask_extra).to(src_extra.dtype)

            srcs.append(src_extra)
            masks_list.append(mask_extra)
            poss.append(pos_extra)

    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for lvl, (src_l, mask_l, pos_l) in enumerate(zip(srcs, masks_list, poss)):
        bs, c, h, w = src_l.shape
        spatial_shapes.append((h, w))
        src_flatten.append(src_l.flatten(2).transpose(1, 2))
        mask_flatten.append(mask_l.flatten(1))
        if level_embed is not None:
            lvl_pos = pos_l.flatten(2).transpose(1, 2) + level_embed[lvl].view(1, 1, -1)
        else:
            lvl_pos = pos_l.flatten(2).transpose(1, 2)
        lvl_pos_embed_flatten.append(lvl_pos)

    return (
        torch.cat(src_flatten, 1),
        torch.cat(lvl_pos_embed_flatten, 1),
        torch.cat(mask_flatten, 1),
        spatial_shapes,
    )


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    full_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    print(f"  {len(full_sd)} total parameters")

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    print("\n=== Building reference Swin + input_proj ===")
    t0 = time.time()
    swin, input_proj, pos_embed, level_embed = build_reference_swin_and_input_proj(full_sd)
    print(f"  Built in {time.time() - t0:.1f}s")

    print("\n=== Running reference pipeline ===")
    t0 = time.time()
    with torch.no_grad():
        ref_src, ref_pos, ref_mask, ref_shapes = run_reference_pipeline(
            swin, input_proj, pos_embed, level_embed, image_tensor, mask
        )
    ref_time = time.time() - t0
    print(f"  Reference: {ref_time*1000:.0f}ms")
    print(f"  src: {ref_src.shape}, pos: {ref_pos.shape}")
    print(f"  shapes: {ref_shapes}")

    print("\n=== Building TTnn backbone wrapper ===")
    t0 = time.time()
    tt_backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    print(f"  Built in {time.time() - t0:.1f}s")

    print("\n=== Running TTnn backbone wrapper ===")
    t0 = time.time()
    with torch.no_grad():
        result = tt_backbone(image_tensor, mask)
    tt_time = time.time() - t0
    tt_src = result["src_flatten"]
    tt_pos = result["pos_flatten"]
    tt_shapes = result["spatial_shapes"]
    print(f"  Wrapper: {tt_time*1000:.0f}ms")
    print(f"  src: {tt_src.shape}, pos: {tt_pos.shape}")
    print(f"  shapes: {tt_shapes.tolist()}")

    print("\n=== Comparison ===")
    pcc_src = compute_pcc(ref_src, tt_src)
    pcc_pos = compute_pcc(ref_pos, tt_pos)
    shapes_match = all(
        ref_shapes[i][0] == tt_shapes[i][0].item() and ref_shapes[i][1] == tt_shapes[i][1].item()
        for i in range(len(ref_shapes))
    )
    mask_match = torch.equal(ref_mask, result["mask_flatten"])

    status_src = "PASS" if pcc_src > 0.999 else "FAIL"
    status_pos = "PASS" if pcc_pos > 0.999 else "FAIL"
    status_shapes = "PASS" if shapes_match else "FAIL"
    status_mask = "PASS" if mask_match else "FAIL"

    print(f"  src_flatten PCC: {pcc_src:.6f} | {status_src}")
    print(f"  pos_flatten PCC: {pcc_pos:.6f} | {status_pos}")
    print(f"  spatial_shapes match: {shapes_match} | {status_shapes}")
    print(f"  mask_flatten match: {mask_match} | {status_mask}")

    total_tokens = tt_src.shape[1]
    print(f"\n  Total tokens: {total_tokens}")
    for i, (h, w) in enumerate(tt_shapes.tolist()):
        print(f"    Level {i}: {h}x{w} = {h*w}")


if __name__ == "__main__":
    main()
