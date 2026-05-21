# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single-image inference demo for ED-Pose on TT P150.

Loads a COCO image, runs the full E2E pipeline
(Backbone→Encoder→TwoStage→Decoder→Heads→PostProcess),
and prints detected persons with bounding boxes and keypoints.

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_inference_demo.py \
    --image /home/yito/datasets/coco/val2017/000000000139.jpg

Or with a default COCO image:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_inference_demo.py
"""

import argparse
import glob
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
    TTEDPoseDecoder,
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

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def preprocess_image(image_path, max_size=1333, target_size=800):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = target_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)
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


def build_prediction_heads(state_dict):
    last = N_DEC_LAYERS - 1

    class_embed = nn.Linear(D_MODEL, NUM_CLASSES)
    ce_sd = {
        k.replace(f"class_embed.{last}.", ""): v
        for k, v in state_dict.items()
        if k.startswith(f"class_embed.{last}.")
    }
    class_embed.load_state_dict(ce_sd)
    class_embed.eval()

    bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
    be_sd = {
        k.replace(f"bbox_embed.{last}.", ""): v
        for k, v in state_dict.items()
        if k.startswith(f"bbox_embed.{last}.")
    }
    bbox_embed.load_state_dict(be_sd)
    bbox_embed.eval()

    pose_idx = last - NUM_BOX_DEC_LAYERS
    pose_embed = MLP(D_MODEL, D_MODEL, 2, 3)
    pe_sd = {
        k.replace(f"pose_embed.{pose_idx}.", ""): v
        for k, v in state_dict.items()
        if k.startswith(f"pose_embed.{pose_idx}.")
    }
    pose_embed.load_state_dict(pe_sd)
    pose_embed.eval()

    kpt_index = [
        x for x in range(NUM_GROUP * (NUM_BODY_POINTS + 1)) if x % (NUM_BODY_POINTS + 1) != 0
    ]

    return class_embed, bbox_embed, pose_embed, kpt_index


@torch.no_grad()
def apply_prediction_heads(hs, references, class_embed, bbox_embed, pose_embed, kpt_index):
    last_hs = hs[-1]
    last_ref = references[-2]

    stride = NUM_BODY_POINTS + 1
    hs_bbox = last_hs[:, 0::stride, :]
    ref_bbox = last_ref[:, 0::stride, :]

    pred_logits = class_embed(hs_bbox)
    delta = bbox_embed(hs_bbox)
    pred_boxes = (delta + inverse_sigmoid(ref_bbox)).sigmoid()

    hs_kpt = last_hs.index_select(1, torch.tensor(kpt_index))
    ref_kpt = last_ref.index_select(1, torch.tensor(kpt_index))

    delta_xy = pose_embed(hs_kpt)
    outputs_unsig = delta_xy + inverse_sigmoid(ref_kpt[..., :2])
    vis = torch.ones_like(outputs_unsig)
    xyv = torch.cat((outputs_unsig, vis[:, :, 0:1]), dim=-1).sigmoid()

    bs = last_hs.shape[0]
    kpt_res = xyv.reshape(bs, NUM_GROUP, NUM_BODY_POINTS, 3).flatten(2, 3)

    pred_keypoints = torch.zeros_like(kpt_res)
    pred_keypoints[..., 0 : 2 * NUM_BODY_POINTS : 2] = kpt_res[..., 0::3]
    pred_keypoints[..., 1 : 2 * NUM_BODY_POINTS : 2] = kpt_res[..., 1::3]
    pred_keypoints[..., 2 * NUM_BODY_POINTS :] = kpt_res[..., 2::3]

    return pred_logits, pred_boxes, pred_keypoints


@torch.no_grad()
def postprocess_results(pred_logits, pred_boxes, pred_keypoints, target_sizes, num_select=100):
    prob = pred_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(pred_logits.shape[0], -1), num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // pred_logits.shape[2]
    labels = topk_indexes % pred_logits.shape[2]

    x_c, y_c, w, h = pred_boxes.unbind(-1)
    boxes = torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    keypoints = torch.gather(
        pred_keypoints, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, NUM_BODY_POINTS * 3)
    )
    Z_pred = keypoints[:, :, : NUM_BODY_POINTS * 2]
    V_pred = keypoints[:, :, NUM_BODY_POINTS * 2 :]
    Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, NUM_BODY_POINTS)[:, None, :]

    keypoints_res = torch.zeros_like(keypoints)
    keypoints_res[..., 0::3] = Z_pred[..., 0::2]
    keypoints_res[..., 1::3] = Z_pred[..., 1::2]
    keypoints_res[..., 2::3] = V_pred

    return scores[0], labels[0], boxes[0], keypoints_res[0]


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
    parser = argparse.ArgumentParser(description="ED-Pose single-image inference demo on TT P150")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to input image. Defaults to first COCO val2017 image.",
    )
    parser.add_argument("--coco-dir", default="/home/yito/datasets/coco")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    args = parser.parse_args()

    if args.image is None:
        candidates = sorted(glob.glob(os.path.join(args.coco_dir, "val2017", "*.jpg")))
        if not candidates:
            print(f"No images found in {args.coco_dir}/val2017/. Specify --image.")
            sys.exit(1)
        args.image = candidates[0]

    assert os.path.exists(args.image), f"Image not found: {args.image}"
    print(f"Image: {args.image}")

    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    full_sd = load_state_dict()
    print(f"Loaded checkpoint: {len(full_sd)} parameters\n")

    print("Building pipeline...")
    t0 = time.time()

    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")

    enc_sd = {}
    for k, v in full_sd.items():
        if k.startswith("transformer.encoder.layers."):
            enc_sd[k[len("transformer.encoder.") :]] = v.float()
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
    )

    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)

    decoder = TTEDPoseDecoder(
        device,
        full_sd,
        D_MODEL,
        D_FFN,
        N_LEVELS,
        N_HEADS,
        N_POINTS,
        N_DEC_LAYERS,
        NUM_QUERIES,
        NUM_CLASSES,
        NUM_BODY_POINTS,
        NUM_BOX_DEC_LAYERS,
        NUM_GROUP,
    )

    class_embed, bbox_embed, pose_embed, kpt_index = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()

    print(f"Pipeline built in {time.time() - t0:.1f}s\n")

    # Preprocess
    print("Preprocessing image...")
    img = Image.open(args.image)
    orig_w, orig_h = img.size
    print(f"  Original size: {orig_w} x {orig_h}")
    tensor, mask, orig_size = preprocess_image(args.image)
    print(f"  Preprocessed: {tensor.shape[2]}x{tensor.shape[3]} (padded)")
    print(f"  Target size for output: {orig_h} x {orig_w}\n")

    # Backbone
    print("=== Phase 1: Backbone (CPU) ===")
    t = time.time()
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    t_bb = time.time() - t
    print(f"  Time: {t_bb*1000:.0f}ms, tokens: {bb_out['src_flatten'].shape[1]}")

    # Encoder
    print("\n=== Phase 2: Encoder (ttnn) ===")
    t = time.time()
    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_tt = ttnn.from_torch(
        bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with torch.no_grad():
        enc_out_tt = encoder(
            src_tt,
            pos_tt,
            bb_out["reference_points"],
            bb_out["spatial_shapes"],
            bb_out["level_start_index"],
            bb_out["mask_flatten"],
        )
    memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(enc_out_tt)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    t_enc = time.time() - t
    print(f"  Time: {t_enc*1000:.0f}ms (includes JIT compilation on first run)")

    # Two-stage
    print("\n=== Phase 3: Two-stage query generation (CPU) ===")
    t = time.time()
    with torch.no_grad():
        query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
    t_ts = time.time() - t
    print(f"  Time: {t_ts*1000:.0f}ms")
    print(f"  Queries: {query_out['tgt'].shape[1]}")

    # Decoder
    print("\n=== Phase 4: Decoder (ttnn layers) ===")
    t = time.time()
    memory_tt = ttnn.from_torch(
        memory.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with torch.no_grad():
        hs, references = decoder(
            tgt=query_out["tgt"],
            memory_tt=memory_tt,
            refpoint_embed=query_out["refpoint_embed"],
            spatial_shapes=bb_out["spatial_shapes"],
            level_start_index=bb_out["level_start_index"],
            valid_ratios=bb_out["valid_ratios"],
            memory_key_padding_mask=bb_out["mask_flatten"],
            self_attn_mask=None,
            self_attn_mask2=attn_mask2,
        )
    ttnn.deallocate(memory_tt)
    t_dec = time.time() - t
    print(f"  Time: {t_dec*1000:.0f}ms")

    # Prediction heads + Postprocess
    print("\n=== Phase 5: Prediction heads + PostProcess ===")
    t = time.time()
    with torch.no_grad():
        pred_logits, pred_boxes, pred_keypoints = apply_prediction_heads(
            hs, references, class_embed, bbox_embed, pose_embed, kpt_index
        )
        scores, labels, boxes, keypoints = postprocess_results(
            pred_logits, pred_boxes, pred_keypoints, orig_size
        )
    t_heads = time.time() - t
    print(f"  Time: {t_heads*1000:.0f}ms")

    # Print timing
    t_total = t_bb + t_enc + t_ts + t_dec + t_heads
    print(f"\n=== Timing Summary ===")
    print(f"  Backbone (CPU):   {t_bb*1000:>7.0f}ms")
    print(f"  Encoder  (ttnn):  {t_enc*1000:>7.0f}ms")
    print(f"  Two-stage (CPU):  {t_ts*1000:>7.0f}ms")
    print(f"  Decoder  (CPU):   {t_dec*1000:>7.0f}ms")
    print(f"  Heads    (CPU):   {t_heads*1000:>7.0f}ms")
    print(f"  Total:            {t_total*1000:>7.0f}ms")

    # Print detections
    high_conf = scores > args.score_threshold
    person_mask = (labels == 1) & high_conf
    n_persons = person_mask.sum().item()

    print(f"\n=== Detection Results ===")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Total detections (score > {args.score_threshold}): {high_conf.sum().item()}")
    print(f"  Person detections: {n_persons}")

    person_indices = torch.where(person_mask)[0]
    for rank, idx in enumerate(person_indices):
        score = scores[idx].item()
        box = boxes[idx]
        kpts = keypoints[idx].reshape(NUM_BODY_POINTS, 3)

        print(f"\n  --- Person {rank+1} (score={score:.3f}) ---")
        print(
            f"  BBox [x1,y1,x2,y2]: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
        )
        print(
            f"  BBox [x,y,w,h]:     [{box[0]:.1f}, {box[1]:.1f}, "
            f"{(box[2]-box[0]):.1f}, {(box[3]-box[1]):.1f}]"
        )
        print(f"  Keypoints:")
        for kp_idx, name in enumerate(COCO_KEYPOINT_NAMES):
            x, y, v = kpts[kp_idx]
            print(f"    {name:>16}: ({x:.1f}, {y:.1f})  conf={v:.3f}")

    if n_persons == 0:
        print("\n  No person detections above threshold.")
        print("  Top-5 scores (all classes):")
        top5_scores, top5_idx = scores[:5], labels[:5]
        for i in range(min(5, len(top5_scores))):
            print(f"    score={top5_scores[i]:.4f}, label={top5_idx[i].item()}")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
