# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
COCO val2017 keypoint AP evaluation for ED-Pose on TT P150.

Pipeline: Backbone(CPU) -> Encoder(ttnn) -> TwoStage(CPU) -> Decoder(CPU) -> Heads(CPU) -> PostProcess

Official ED-Pose score: 75.8 AP (keypoints) on COCO val2017.

Prerequisites:
  pip install pycocotools

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_coco_eval.py \
    --coco-dir /home/yito/datasets/coco --max-images 100

Full evaluation (slow, ~5h):
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_coco_eval.py \
    --coco-dir /home/yito/datasets/coco
"""

import argparse
import json
import math
import os
import sys
import time

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
    EDPoseDecoder,
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


def compute_padded_size(orig_w, orig_h, max_size=1333, target_size=800):
    scale = target_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    return (new_h + pad_h, new_w + pad_w)


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
def postprocess_single(pred_logits, pred_boxes, pred_keypoints, target_sizes, num_select=100):
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

    return {
        "scores": scores[0],
        "labels": labels[0],
        "boxes": boxes[0],
        "keypoints": keypoints_res[0],
    }


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


def results_to_coco_format(result, image_id):
    kpt_results = []
    bbox_results = []
    scores = result["scores"].cpu()
    labels = result["labels"].cpu()
    boxes = result["boxes"].cpu()
    keypoints = result["keypoints"].cpu()

    for k in range(len(scores)):
        score = float(scores[k].item())
        label = int(labels[k].item())
        box = boxes[k]

        kpt_results.append(
            {
                "image_id": image_id,
                "category_id": label,
                "keypoints": keypoints[k].tolist(),
                "score": score,
            }
        )

        bbox_results.append(
            {
                "image_id": image_id,
                "category_id": label,
                "bbox": [
                    box[0].item(),
                    box[1].item(),
                    (box[2] - box[0]).item(),
                    (box[3] - box[1]).item(),
                ],
                "score": score,
            }
        )

    return kpt_results, bbox_results


def main():
    parser = argparse.ArgumentParser(description="COCO val2017 keypoint AP evaluation for ED-Pose on TT P150")
    parser.add_argument("--coco-dir", default="/home/yito/datasets/coco")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to process (0=all). Use 100 for quick test.",
    )
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--output", default=None, help="Output JSON path for raw results")
    parser.add_argument(
        "--sort-by-size",
        action="store_true",
        default=True,
        help="Sort images by padded size to minimize JIT recompilations (default: True)",
    )
    parser.add_argument("--no-sort-by-size", dest="sort_by_size", action="store_false")
    args = parser.parse_args()

    ann_file = os.path.join(args.coco_dir, "annotations", "person_keypoints_val2017.json")
    img_dir = os.path.join(args.coco_dir, "val2017")

    assert os.path.exists(ann_file), f"Annotation file not found: {ann_file}"
    assert os.path.isdir(img_dir), f"Image directory not found: {img_dir}"

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("pycocotools not found. Install with: pip install pycocotools")
        sys.exit(1)

    # Open device
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    # Load checkpoint
    full_sd = load_state_dict()
    print(f"Loaded checkpoint: {len(full_sd)} parameters\n")

    # Build pipeline
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

    class_embed, bbox_embed_head, pose_embed, kpt_index = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()

    print(f"Pipeline built in {time.time() - t0:.1f}s\n")

    # Load COCO annotations
    print("Loading COCO annotations...")
    coco_gt = COCO(ann_file)
    image_ids = sorted(coco_gt.getImgIds())
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]

    # Sort by padded size to minimize JIT recompilations
    if args.sort_by_size:
        size_map = {}
        for img_id in image_ids:
            info = coco_gt.loadImgs(img_id)[0]
            size_map[img_id] = compute_padded_size(info["width"], info["height"])
        image_ids.sort(key=lambda x: size_map[x])
        unique_sizes = len(set(size_map.values()))
        print(f"  Sorted by size: {unique_sizes} unique padded sizes")

    print(f"  Processing {len(image_ids)} images\n")

    # Process images
    all_kpt_results = []
    all_bbox_results = []
    timings = {
        "backbone": [],
        "encoder": [],
        "two_stage": [],
        "decoder": [],
        "heads": [],
        "total": [],
    }
    errors = []
    prev_padded_size = None
    jit_compilations = 0

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            errors.append((img_id, "file not found"))
            continue

        try:
            t_total = time.time()

            # Preprocess
            tensor, mask, orig_size = preprocess_image(img_path)
            padded_size = (tensor.shape[2], tensor.shape[3])
            if padded_size != prev_padded_size:
                jit_compilations += 1
                prev_padded_size = padded_size

            # Backbone (CPU)
            t = time.time()
            with torch.no_grad():
                bb_out = backbone(tensor, mask)
            t_bb = time.time() - t

            # Encoder (ttnn)
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

            # Two-stage (CPU)
            t = time.time()
            with torch.no_grad():
                query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
            t_ts = time.time() - t

            # Decoder (ttnn layers)
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

            # Prediction heads + PostProcess (CPU)
            t = time.time()
            with torch.no_grad():
                pred_logits, pred_boxes, pred_keypoints = apply_prediction_heads(
                    hs, references, class_embed, bbox_embed_head, pose_embed, kpt_index
                )
                result = postprocess_single(
                    pred_logits, pred_boxes, pred_keypoints, orig_size, args.num_select
                )
            t_heads = time.time() - t

            t_total = time.time() - t_total

            # Collect COCO-format results
            kpt_res, bbox_res = results_to_coco_format(result, img_id)
            all_kpt_results.extend(kpt_res)
            all_bbox_results.extend(bbox_res)

            # Track timing (skip first 2 images per unique size for JIT warmup)
            timings["backbone"].append(t_bb)
            timings["encoder"].append(t_enc)
            timings["two_stage"].append(t_ts)
            timings["decoder"].append(t_dec)
            timings["heads"].append(t_heads)
            timings["total"].append(t_total)

            # Progress
            n_det = (result["scores"] > 0.3).sum().item()
            if idx < 3 or (idx + 1) % 50 == 0 or idx == len(image_ids) - 1:
                print(
                    f"  [{idx+1:>5}/{len(image_ids)}] {img_info['file_name']}: "
                    f"{n_det:>2} det(>0.3), {t_total:.1f}s "
                    f"[bb={t_bb:.1f} enc={t_enc:.1f} ts={t_ts:.2f} dec={t_dec:.2f}] "
                    f"pad={padded_size[0]}x{padded_size[1]}"
                )

        except Exception as e:
            errors.append((img_id, str(e)))
            print(f"  [{idx+1:>5}/{len(image_ids)}] {img_info['file_name']}: ERROR - {e}")
            continue

    # Timing summary
    print(f"\n{'='*60}")
    print(f"Timing Summary ({len(timings['total'])} images)")
    print(f"{'='*60}")
    for key in ["backbone", "encoder", "two_stage", "decoder", "heads", "total"]:
        vals = timings[key]
        if vals:
            avg = np.mean(vals)
            med = np.median(vals)
            p90 = np.percentile(vals, 90)
            print(f"  {key:>12}: avg={avg*1000:>7.0f}ms  med={med*1000:>7.0f}ms  p90={p90*1000:>7.0f}ms")
    print(f"  JIT compilations (unique sizes): {jit_compilations}")
    if errors:
        print(f"  Errors: {len(errors)}")

    # Save raw results
    if args.output:
        print(f"\nSaving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump({"keypoints": all_kpt_results, "bbox": all_bbox_results}, f)
        print(f"  Saved {len(all_kpt_results)} keypoint results, {len(all_bbox_results)} bbox results")

    # COCO Keypoint Evaluation
    print(f"\n{'='*60}")
    print(f"COCO Keypoint Evaluation")
    print(f"{'='*60}")
    print(f"Total predictions: {len(all_kpt_results)}")

    if all_kpt_results:
        try:
            coco_dt = coco_gt.loadRes(all_kpt_results)
            coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
            coco_eval.params.imgIds = [
                img_id for img_id in image_ids if img_id not in {e[0] for e in errors}
            ]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats
            print(f"\n  AP       = {stats[0]:.3f}  (official: 0.758)")
            print(f"  AP50     = {stats[1]:.3f}")
            print(f"  AP75     = {stats[2]:.3f}")
            print(f"  AP_M     = {stats[3]:.3f}")
            print(f"  AP_L     = {stats[4]:.3f}")
            print(f"  AR       = {stats[5]:.3f}")
        except Exception as e:
            print(f"  Keypoint eval error: {e}")
    else:
        print("  No predictions to evaluate.")

    # COCO Bbox Evaluation
    print(f"\n{'='*60}")
    print(f"COCO Bbox Evaluation")
    print(f"{'='*60}")
    print(f"Total predictions: {len(all_bbox_results)}")

    if all_bbox_results:
        try:
            coco_dt_bbox = coco_gt.loadRes(all_bbox_results)
            coco_eval_bbox = COCOeval(coco_gt, coco_dt_bbox, "bbox")
            coco_eval_bbox.params.imgIds = [
                img_id for img_id in image_ids if img_id not in {e[0] for e in errors}
            ]
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
        except Exception as e:
            print(f"  Bbox eval error: {e}")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
