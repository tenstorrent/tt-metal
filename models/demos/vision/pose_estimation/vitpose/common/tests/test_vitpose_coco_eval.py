# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
COCO val2017 keypoint evaluation for ViTPose-B.

Reproduces the published AP 75.8 benchmark using GT bounding boxes.
Compares TT device (bfloat16) against HuggingFace fp32 baseline.

Prerequisites:
  - COCO val2017 data at COCO_DATA_DIR (annotations + images)
  - uv pip install pycocotools (installed at runtime)

Usage:
  pytest models/demos/vision/pose_estimation/vitpose/common/tests/test_vitpose_coco_eval.py -v -s --timeout=3600
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import VitPoseImageProcessor

import ttnn
from models.demos.vision.pose_estimation.vitpose.common.common import load_torch_model
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose import VitPose

COCO_DATA_DIR = os.environ.get("COCO_DATA_DIR", "/home/yito/datasets/coco")
MAX_IMAGES = int(os.environ.get("VITPOSE_EVAL_MAX_IMAGES", "0"))


@dataclass
class HeatmapOutput:
    heatmaps: torch.Tensor


def load_coco_person_annotations(data_dir):
    ann_file = os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")
    coco = COCO(ann_file)
    person_cat_id = coco.getCatIds(catNms=["person"])[0]
    img_ids = sorted(coco.getImgIds(catIds=[person_cat_id]))
    return coco, img_ids, person_cat_id


def format_keypoint_results(pose_results, img_id, category_id):
    """Convert post_process output to COCO result format."""
    results = []
    for img_results in pose_results:
        for person in img_results:
            keypoints = person["keypoints"].numpy()
            scores = person["scores"].numpy()
            kp_flat = []
            for k in range(17):
                x, y = keypoints[k]
                v = 2 if scores[k] > 0.2 else 0
                kp_flat.extend([float(x), float(y), v])
            results.append(
                {
                    "image_id": img_id,
                    "category_id": category_id,
                    "keypoints": kp_flat,
                    "score": float(np.mean(scores)),
                }
            )
    return results


def run_coco_eval(coco_gt, results, ann_type="keypoints"):
    if not results:
        logger.warning("No results to evaluate")
        return {}

    result_file = "/tmp/vitpose_coco_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = sorted(set(r["image_id"] for r in results))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_medium": coco_eval.stats[3],
        "AP_large": coco_eval.stats[4],
        "AR": coco_eval.stats[5],
    }


def evaluate_vitpose_tt(device, max_images=0):
    coco, img_ids, person_cat_id = load_coco_person_annotations(COCO_DATA_DIR)
    processor = VitPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")

    if max_images > 0:
        img_ids = img_ids[:max_images]

    model_hf = load_torch_model()
    state_dict = model_hf.state_dict()
    tt_model = VitPose(state_dict, device, batch_size=1)

    results = []
    total_anns = 0
    img_dir = os.path.join(COCO_DATA_DIR, "val2017")

    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")

        anns = [
            a
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False))
            if a.get("num_keypoints", 0) > 0
        ]
        if not anns:
            continue

        for ann in anns:
            bbox = ann["bbox"]
            inputs = processor(image, boxes=[[bbox]], return_tensors="pt")
            pixel_values_bf16 = inputs["pixel_values"].to(torch.bfloat16)
            tt_input = VitPose.prepare_input(pixel_values_bf16, device)
            tt_output = tt_model(tt_input)
            tt_heatmaps = ttnn.to_torch(tt_output)
            tt_heatmaps = tt_heatmaps.reshape(1, 64, 48, 17).permute(0, 3, 1, 2).float()

            output_wrapper = HeatmapOutput(heatmaps=tt_heatmaps)
            pose_results = processor.post_process_pose_estimation(output_wrapper, boxes=[[[*bbox]]])
            results.extend(format_keypoint_results(pose_results, img_id, person_cat_id))
            total_anns += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"[TT] {idx + 1}/{len(img_ids)} images, {total_anns} annotations")

    logger.info(f"[TT] Done: {len(img_ids)} images, {total_anns} annotations, {len(results)} predictions")
    return run_coco_eval(coco, results)


def evaluate_vitpose_hf(max_images=0):
    coco, img_ids, person_cat_id = load_coco_person_annotations(COCO_DATA_DIR)
    processor = VitPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")

    if max_images > 0:
        img_ids = img_ids[:max_images]

    model = load_torch_model()

    results = []
    total_anns = 0
    img_dir = os.path.join(COCO_DATA_DIR, "val2017")

    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(img_dir, img_info["file_name"])).convert("RGB")

        anns = [
            a
            for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[person_cat_id], iscrowd=False))
            if a.get("num_keypoints", 0) > 0
        ]
        if not anns:
            continue

        for ann in anns:
            bbox = ann["bbox"]
            inputs = processor(image, boxes=[[bbox]], return_tensors="pt")

            with torch.no_grad():
                hf_output = model(inputs["pixel_values"])

            pose_results = processor.post_process_pose_estimation(hf_output, boxes=[[[*bbox]]])
            results.extend(format_keypoint_results(pose_results, img_id, person_cat_id))
            total_anns += 1

        if (idx + 1) % 100 == 0:
            logger.info(f"[HF] {idx + 1}/{len(img_ids)} images, {total_anns} annotations")

    logger.info(f"[HF] Done: {len(img_ids)} images, {total_anns} annotations, {len(results)} predictions")
    return run_coco_eval(coco, results)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vitpose_coco_eval(device):
    metrics = evaluate_vitpose_tt(device, max_images=MAX_IMAGES)

    logger.info("=== ViTPose-B TT Device COCO Evaluation ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    assert metrics.get("AP", 0) > 0.70, f"AP {metrics.get('AP', 0):.4f} below 0.70 threshold"


def test_vitpose_coco_eval_hf():
    metrics = evaluate_vitpose_hf(max_images=MAX_IMAGES)

    logger.info("=== ViTPose-B HuggingFace fp32 COCO Evaluation ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    assert metrics.get("AP", 0) > 0.70, f"AP {metrics.get('AP', 0):.4f} below 0.70 threshold"
