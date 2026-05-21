# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug script to trace bbox prediction through the pipeline."""

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import (
    EDPoseDecoder,
    MLP,
    inverse_sigmoid,
)

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_ENC_LAYERS = 6
N_DEC_LAYERS = 6
NUM_QUERIES = 900
NUM_CLASSES = 2
NUM_BODY_POINTS = 17
NUM_BOX_DEC_LAYERS = 2
NUM_GROUP = 100


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = 800 / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > 1333:
        scale = 1333 / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = normalize(img)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    mask = torch.ones(tensor.shape[1], tensor.shape[2], dtype=torch.bool)
    mask[:new_h, :new_w] = False
    return tensor.unsqueeze(0), mask.unsqueeze(0), torch.tensor([[orig_h, orig_w]])


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def prepare_attn_mask2():
    total_q = NUM_GROUP * (NUM_BODY_POINTS + 1)
    group_size = NUM_BODY_POINTS + 1
    kpt_index = [x for x in range(total_q) if x % group_size == 0]
    attn_mask = torch.zeros(1, N_HEADS, total_q, total_q, dtype=torch.bool)
    for matchj in range(total_q):
        sj = (matchj // group_size) * group_size
        ej = (matchj // group_size + 1) * group_size
        if sj > 0:
            attn_mask[:, :, matchj, :sj] = True
        if ej < total_q:
            attn_mask[:, :, matchj, ej:] = True
    for match_x in range(total_q):
        if match_x % group_size == 0:
            attn_mask[:, :, match_x, kpt_index] = False
    return attn_mask.flatten(0, 1)


def main():
    image_path = "/home/yito/datasets/coco/val2017/000000000139.jpg"

    device = ttnn.open_device(device_id=0)
    full_sd = load_state_dict()

    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")

    enc_sd = {}
    for k, v in full_sd.items():
        if k.startswith("transformer.encoder.layers."):
            enc_sd[k[len("transformer.encoder.") :]] = v.float()
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
    )

    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)

    decoder = EDPoseDecoder(
        full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES,
        NUM_BODY_POINTS, NUM_BOX_DEC_LAYERS, NUM_GROUP,
    )

    attn_mask2 = prepare_attn_mask2()

    # Preprocess
    tensor, mask, orig_size = preprocess_image(image_path)
    print(f"Image: {tensor.shape}, orig_size: {orig_size}")

    # Backbone
    with torch.no_grad():
        bb_out = backbone(tensor, mask)

    # Encoder (ttnn)
    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_tt = ttnn.from_torch(
        bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with torch.no_grad():
        enc_out_tt = encoder(
            src_tt, pos_tt, bb_out["reference_points"],
            bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"],
        )
    memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(enc_out_tt)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)

    # Two-stage
    with torch.no_grad():
        query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])

    # Print two-stage output
    refpoint_embed = query_out["refpoint_embed"]  # (N, 900, 4) unsigmoided
    ref_sig = refpoint_embed.sigmoid()
    print(f"\n=== Two-stage refpoint_embed ===")
    print(f"  Shape: {refpoint_embed.shape}")
    print(f"  Sigmoided stats (cx, cy, w, h):")
    for dim, name in enumerate(["cx", "cy", "w", "h"]):
        vals = ref_sig[0, :, dim]
        print(f"    {name}: mean={vals.mean():.3f}, min={vals.min():.3f}, max={vals.max():.3f}, std={vals.std():.3f}")

    # Decoder with debug output
    with torch.no_grad():
        hs, references = decoder(
            tgt=query_out["tgt"], memory=memory,
            refpoint_embed=query_out["refpoint_embed"],
            spatial_shapes=bb_out["spatial_shapes"],
            level_start_index=bb_out["level_start_index"],
            valid_ratios=bb_out["valid_ratios"],
            memory_key_padding_mask=bb_out["mask_flatten"],
            self_attn_mask=None, self_attn_mask2=attn_mask2,
        )

    print(f"\n=== Reference points at each stage ===")
    stride = NUM_BODY_POINTS + 1
    for i, ref in enumerate(references):
        label = f"ref[{i}]"
        if i == 0:
            label += " (initial)"
        elif i <= 2:
            label += f" (after layer {i-1}, nq={ref.shape[1]})"
        else:
            label += f" (after layer {i-1})"

        if ref.shape[1] <= 900:
            # Pre-expansion: just show some query stats
            for dim, name in enumerate(["cx", "cy", "w", "h"]):
                vals = ref[0, :, dim]
                print(f"  {label} {name}: mean={vals.mean():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")
        else:
            # Post-expansion: show box queries at stride 18
            box_ref = ref[0, 0::stride, :]
            for dim, name in enumerate(["cx", "cy", "w", "h"]):
                vals = box_ref[:, dim]
                print(f"  {label} box_{name}: mean={vals.mean():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")

    # Apply prediction heads for last layer
    print(f"\n=== Final prediction heads ===")
    last_hs = hs[-1]
    last_ref = references[-2]

    hs_bbox = last_hs[:, 0::stride, :]
    ref_bbox = last_ref[:, 0::stride, :]

    # Load bbox_embed for last layer
    last = N_DEC_LAYERS - 1
    bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
    be_sd = {k.replace(f"bbox_embed.{last}.", ""): v
             for k, v in full_sd.items() if k.startswith(f"bbox_embed.{last}.")}
    bbox_embed.load_state_dict(be_sd)
    bbox_embed.eval()

    class_embed = nn.Linear(D_MODEL, NUM_CLASSES)
    ce_sd = {k.replace(f"class_embed.{last}.", ""): v
             for k, v in full_sd.items() if k.startswith(f"class_embed.{last}.")}
    class_embed.load_state_dict(ce_sd)
    class_embed.eval()

    with torch.no_grad():
        delta = bbox_embed(hs_bbox)
        pred_boxes = (delta + inverse_sigmoid(ref_bbox)).sigmoid()
        pred_logits = class_embed(hs_bbox)

    # Find top-scoring detection
    prob = pred_logits.sigmoid()
    scores = prob[0].max(-1)[0]
    top_idx = scores.argmax().item()
    top_score = scores[top_idx].item()

    print(f"  Top box query: idx={top_idx}, score={top_score:.3f}")
    print(f"  ref_bbox[{top_idx}] (sigmoided): {ref_bbox[0, top_idx].tolist()}")
    print(f"  delta[{top_idx}]:                {delta[0, top_idx].tolist()}")
    print(f"  pred_boxes[{top_idx}] (cxcywh):  {pred_boxes[0, top_idx].tolist()}")

    # Compare with expected (from CPU reference):
    # CPU ref normalized: cx=0.684, cy=0.532, w=0.090, h=0.326
    print(f"\n  Expected (CPU ref): cx=0.684, cy=0.532, w=0.090, h=0.326")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
