# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ED-Pose demo: run inference on a COCO image and save pose overlay.

Runs the full TT P150 pipeline (Backbone→Encoder→TwoStage→Decoder→Heads),
then draws bounding boxes, keypoint skeletons, and confidence scores
on the original image.

Usage inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_demo_with_overlay.py
"""

import math
import os
import sys
import time

import cv2
import numpy as np
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

INPUT_IMAGE = "/home/yito/datasets/coco/val2017/000000053626.jpg"
OUTPUT_PATH = "/home/yito/edpose_demo_output.jpg"
SCORE_THRESHOLD = 0.3

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SKELETON_COLORS = [
    (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
    (0, 255, 0), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
    (255, 0, 0), (0, 0, 255), (0, 255, 0),
    (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255),
]

PERSON_COLORS = [
    (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 255, 0), (128, 255, 0),
]


def preprocess_image(image_path, max_size=1333, target_size=800):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = target_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
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
    ce_sd = {k.replace(f"class_embed.{last}.", ""): v
             for k, v in state_dict.items() if k.startswith(f"class_embed.{last}.")}
    class_embed.load_state_dict(ce_sd)
    class_embed.eval()

    bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
    be_sd = {k.replace(f"bbox_embed.{last}.", ""): v
             for k, v in state_dict.items() if k.startswith(f"bbox_embed.{last}.")}
    bbox_embed.load_state_dict(be_sd)
    bbox_embed.eval()

    pose_idx = last - NUM_BOX_DEC_LAYERS
    pose_embed = MLP(D_MODEL, D_MODEL, 2, 3)
    pe_sd = {k.replace(f"pose_embed.{pose_idx}.", ""): v
             for k, v in state_dict.items() if k.startswith(f"pose_embed.{pose_idx}.")}
    pose_embed.load_state_dict(pe_sd)
    pose_embed.eval()

    kpt_index = [x for x in range(NUM_GROUP * (NUM_BODY_POINTS + 1))
                 if x % (NUM_BODY_POINTS + 1) != 0]
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
    pred_keypoints[..., 0:2*NUM_BODY_POINTS:2] = kpt_res[..., 0::3]
    pred_keypoints[..., 1:2*NUM_BODY_POINTS:2] = kpt_res[..., 1::3]
    pred_keypoints[..., 2*NUM_BODY_POINTS:] = kpt_res[..., 2::3]
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
        pred_keypoints, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, NUM_BODY_POINTS * 3))
    Z_pred = keypoints[:, :, :NUM_BODY_POINTS * 2]
    V_pred = keypoints[:, :, NUM_BODY_POINTS * 2:]
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


def draw_pose_overlay(image_path, scores, labels, boxes, keypoints, output_path,
                      score_threshold=0.3):
    img = cv2.imread(image_path)
    overlay = img.copy()
    h, w = img.shape[:2]

    person_mask = (labels == 1) & (scores > score_threshold)
    person_indices = torch.where(person_mask)[0]

    print(f"\nDrawing {len(person_indices)} person detections...")

    for rank, idx in enumerate(person_indices):
        score = scores[idx].item()
        box = boxes[idx].numpy()
        kpts = keypoints[idx].reshape(NUM_BODY_POINTS, 3).numpy()
        color = PERSON_COLORS[rank % len(PERSON_COLORS)]

        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label_text = f"person {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        for si, (j1, j2) in enumerate(COCO_SKELETON):
            x_a, y_a, v_a = kpts[j1]
            x_b, y_b, v_b = kpts[j2]
            if v_a > 0.3 and v_b > 0.3:
                pt1 = (int(x_a), int(y_a))
                pt2 = (int(x_b), int(y_b))
                skel_color = SKELETON_COLORS[si % len(SKELETON_COLORS)]
                cv2.line(overlay, pt1, pt2, skel_color, 2, cv2.LINE_AA)

        for ki in range(NUM_BODY_POINTS):
            x_k, y_k, v_k = kpts[ki]
            if v_k > 0.3:
                cv2.circle(overlay, (int(x_k), int(y_k)), 4, color, -1)
                cv2.circle(overlay, (int(x_k), int(y_k)), 4, (0, 0, 0), 1)

        print(f"  Person {rank+1}: score={score:.3f}, "
              f"bbox=[{x1},{y1},{x2},{y2}], "
              f"visible_kpts={sum(1 for k in kpts if k[2] > 0.3)}/{NUM_BODY_POINTS}")

    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    info_y = 30
    cv2.putText(result, f"ED-Pose on TT P150 | {len(person_indices)} persons detected",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"ED-Pose on TT P150 | {len(person_indices)} persons detected",
                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved overlay to: {output_path}")
    return len(person_indices)


def main():
    print(f"Input image: {INPUT_IMAGE}")
    print(f"Output path: {OUTPUT_PATH}")
    assert os.path.exists(INPUT_IMAGE), f"Image not found: {INPUT_IMAGE}"

    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    full_sd = load_state_dict()
    print(f"Loaded checkpoint: {len(full_sd)} parameters\n")

    print("Building pipeline...")
    t0 = time.time()
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN,
                                   N_LEVELS, N_HEADS, N_POINTS)
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = TTEDPoseDecoder(device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
                               N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS,
                               NUM_BOX_DEC_LAYERS, NUM_GROUP)
    class_embed, bbox_embed, pose_embed, kpt_index = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()
    print(f"Pipeline built in {time.time() - t0:.1f}s\n")

    print("Preprocessing image...")
    tensor, mask, orig_size = preprocess_image(INPUT_IMAGE)
    orig_h, orig_w = orig_size[0].tolist()
    print(f"  Original: {orig_w}x{orig_h}, preprocessed: {tensor.shape[2]}x{tensor.shape[3]}")

    timings = {}

    # Phase 1: Backbone
    print("\nRunning Backbone (CPU)...")
    t = time.time()
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    timings["backbone"] = time.time() - t
    print(f"  {timings['backbone']*1000:.0f}ms, tokens={bb_out['src_flatten'].shape[1]}")

    # Phase 2: Encoder
    print("Running Encoder (ttnn)...")
    t = time.time()
    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with torch.no_grad():
        enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                              bb_out["spatial_shapes"], bb_out["level_start_index"],
                              bb_out["mask_flatten"])
    memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    timings["encoder"] = time.time() - t
    print(f"  {timings['encoder']*1000:.0f}ms")

    # Phase 3: Two-stage
    print("Running Two-stage (CPU)...")
    t = time.time()
    with torch.no_grad():
        query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
    timings["two_stage"] = time.time() - t
    print(f"  {timings['two_stage']*1000:.0f}ms, queries={query_out['tgt'].shape[1]}")

    # Phase 4: Decoder
    print("Running Decoder (ttnn)...")
    t = time.time()
    with torch.no_grad():
        hs, references = decoder(
            tgt=query_out["tgt"],
            memory_tt=enc_out_tt,
            refpoint_embed=query_out["refpoint_embed"],
            spatial_shapes=bb_out["spatial_shapes"],
            level_start_index=bb_out["level_start_index"],
            valid_ratios=bb_out["valid_ratios"],
            memory_key_padding_mask=bb_out["mask_flatten"],
            self_attn_mask=None,
            self_attn_mask2=attn_mask2,
        )
    ttnn.deallocate(enc_out_tt)
    timings["decoder"] = time.time() - t
    print(f"  {timings['decoder']*1000:.0f}ms")

    # Phase 5: Heads + PostProcess
    print("Running Heads + PostProcess...")
    t = time.time()
    with torch.no_grad():
        pred_logits, pred_boxes, pred_keypoints = apply_prediction_heads(
            hs, references, class_embed, bbox_embed, pose_embed, kpt_index)
        scores, labels, boxes, keypoints_out = postprocess_results(
            pred_logits, pred_boxes, pred_keypoints, orig_size)
    timings["heads"] = time.time() - t
    print(f"  {timings['heads']*1000:.0f}ms")

    # Timing summary
    total = sum(timings.values())
    print(f"\n{'='*50}")
    print(f"  Backbone (CPU):   {timings['backbone']*1000:>7.0f}ms")
    print(f"  Encoder  (ttnn):  {timings['encoder']*1000:>7.0f}ms")
    print(f"  Two-stage (CPU):  {timings['two_stage']*1000:>7.0f}ms")
    print(f"  Decoder  (ttnn):  {timings['decoder']*1000:>7.0f}ms")
    print(f"  Heads    (CPU):   {timings['heads']*1000:>7.0f}ms")
    print(f"  Total:            {total*1000:>7.0f}ms")
    print(f"{'='*50}")

    # Draw and save overlay
    n_det = draw_pose_overlay(INPUT_IMAGE, scores, labels, boxes, keypoints_out,
                               OUTPUT_PATH, SCORE_THRESHOLD)

    if n_det == 0:
        print("\nNo detections above threshold. Top-5 scores:")
        for i in range(min(5, len(scores))):
            print(f"  score={scores[i]:.4f}, label={labels[i].item()}")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
