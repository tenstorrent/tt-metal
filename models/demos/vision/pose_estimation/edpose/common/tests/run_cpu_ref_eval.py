# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only reference eval to validate the pipeline gives correct AP without bf16."""

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

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import (
    EDPoseDecoder,
    MLP,
    CPUMSDeformAttn,
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


class CPUEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = CPUMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + src2
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src


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
    ce_sd = {k.replace(f"class_embed.{last}.", ""): v for k, v in state_dict.items() if k.startswith(f"class_embed.{last}.")}
    class_embed.load_state_dict(ce_sd)
    class_embed.eval()

    bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
    be_sd = {k.replace(f"bbox_embed.{last}.", ""): v for k, v in state_dict.items() if k.startswith(f"bbox_embed.{last}.")}
    bbox_embed.load_state_dict(be_sd)
    bbox_embed.eval()

    pose_idx = last - NUM_BOX_DEC_LAYERS
    pose_embed = MLP(D_MODEL, D_MODEL, 2, 3)
    pe_sd = {k.replace(f"pose_embed.{pose_idx}.", ""): v for k, v in state_dict.items() if k.startswith(f"pose_embed.{pose_idx}.")}
    pose_embed.load_state_dict(pe_sd)
    pose_embed.eval()

    kpt_index = [x for x in range(NUM_GROUP * (NUM_BODY_POINTS + 1)) if x % (NUM_BODY_POINTS + 1) != 0]
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

    keypoints = torch.gather(pred_keypoints, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, NUM_BODY_POINTS * 3))
    Z_pred = keypoints[:, :, :NUM_BODY_POINTS * 2]
    V_pred = keypoints[:, :, NUM_BODY_POINTS * 2:]
    Z_pred = Z_pred * torch.stack([img_w, img_h], dim=1).repeat(1, NUM_BODY_POINTS)[:, None, :]

    keypoints_res = torch.zeros_like(keypoints)
    keypoints_res[..., 0::3] = Z_pred[..., 0::2]
    keypoints_res[..., 1::3] = Z_pred[..., 1::2]
    keypoints_res[..., 2::3] = V_pred

    return {"scores": scores[0], "labels": labels[0], "boxes": boxes[0], "keypoints": keypoints_res[0]}


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
        kpt_results.append({
            "image_id": image_id,
            "category_id": label,
            "keypoints": keypoints[k].tolist(),
            "score": score,
        })
        bbox_results.append({
            "image_id": image_id,
            "category_id": label,
            "bbox": [box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()],
            "score": score,
        })
    return kpt_results, bbox_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="/home/yito/datasets/coco")
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ann_file = os.path.join(args.coco_dir, "annotations", "person_keypoints_val2017.json")
    img_dir = os.path.join(args.coco_dir, "val2017")

    full_sd = load_state_dict()
    print("Loaded checkpoint")

    # Build CPU-only pipeline
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")

    encoder_layers = nn.ModuleList()
    for i in range(N_ENC_LAYERS):
        layer = CPUEncoderLayer(D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
        layer_sd = {}
        prefix = f"transformer.encoder.layers.{i}."
        for k, v in full_sd.items():
            if k.startswith(prefix):
                layer_sd[k[len(prefix):]] = v
        layer.load_state_dict(layer_sd, strict=True)
        layer.eval()
        encoder_layers.append(layer)

    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = EDPoseDecoder(
        full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES,
        NUM_BODY_POINTS, NUM_BOX_DEC_LAYERS, NUM_GROUP,
    )
    class_embed, bbox_embed_head, pose_embed, kpt_index = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()
    print("Pipeline built\n")

    coco_gt = COCO(ann_file)
    image_ids = sorted(coco_gt.getImgIds())
    if args.max_images > 0:
        image_ids = image_ids[:args.max_images]
    print(f"Processing {len(image_ids)} images\n")

    all_kpt_results = []
    all_bbox_results = []

    for idx, img_id in enumerate(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])

        try:
            t = time.time()
            tensor, mask, orig_size = preprocess_image(img_path)

            with torch.no_grad():
                bb_out = backbone(tensor, mask)

                # CPU encoder (fp32)
                memory = bb_out["src_flatten"].clone()
                for layer in encoder_layers:
                    memory = layer(
                        memory, bb_out["pos_flatten"],
                        bb_out["reference_points"], bb_out["spatial_shapes"],
                        bb_out["level_start_index"], bb_out["mask_flatten"],
                    )

                query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])

                hs, references = decoder(
                    tgt=query_out["tgt"], memory=memory,
                    refpoint_embed=query_out["refpoint_embed"],
                    spatial_shapes=bb_out["spatial_shapes"],
                    level_start_index=bb_out["level_start_index"],
                    valid_ratios=bb_out["valid_ratios"],
                    memory_key_padding_mask=bb_out["mask_flatten"],
                    self_attn_mask=None, self_attn_mask2=attn_mask2,
                )

                pred_logits, pred_boxes, pred_keypoints = apply_prediction_heads(
                    hs, references, class_embed, bbox_embed_head, pose_embed, kpt_index
                )
                result = postprocess_single(pred_logits, pred_boxes, pred_keypoints, orig_size)

            kpt_res, bbox_res = results_to_coco_format(result, img_id)
            all_kpt_results.extend(kpt_res)
            all_bbox_results.extend(bbox_res)

            n_det = (result["scores"] > 0.3).sum().item()
            elapsed = time.time() - t
            print(f"  [{idx+1:>3}/{len(image_ids)}] {img_info['file_name']}: {n_det} det, {elapsed:.1f}s")

        except Exception as e:
            print(f"  [{idx+1:>3}/{len(image_ids)}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n=== COCO Keypoint Evaluation (CPU reference) ===")
    print(f"Total predictions: {len(all_kpt_results)}")
    if all_kpt_results:
        coco_dt = coco_gt.loadRes(all_kpt_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap = coco_eval.stats[0]
        print(f"\n  AP = {ap:.3f}  (official: 0.758)")

    print(f"\n=== COCO Bbox Evaluation (CPU reference) ===")
    if all_bbox_results:
        coco_dt = coco_gt.loadRes(all_bbox_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    print("\nDone.")


if __name__ == "__main__":
    main()
