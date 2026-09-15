# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from PIL import Image
from transformers import RTDetrImageProcessor
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vision.detection.rtdetr.common.preprocessing import custom_preprocessor
from models.demos.vision.detection.rtdetr.demo import MODEL_CONFIGS
from models.demos.vision.detection.rtdetr.tt.model import TtRTDetrModel

CONFIDENCE_THRESHOLD = 0.5
MATCH_IOU_THRESHOLD = 0.5
MIN_MATCHED_RATIO = 0.8
MIN_LABEL_AGREEMENT = 0.9
MIN_MEAN_IOU = 0.9
MAX_MEAN_SCORE_ERROR = 0.1
COCO_IMAGES_ENV = "COCO_VAL2017_IMAGES"
DEFAULT_COCO_IMAGES_PATH = Path(__file__).resolve().parents[1] / "datasets" / "coco" / "val2017"
NUM_VALIDATION_IMAGES = 500


def calculate_box_iou(boxes_1, boxes_2):
    areas_1 = (boxes_1[:, 2] - boxes_1[:, 0]).clamp(min=0) * (boxes_1[:, 3] - boxes_1[:, 1]).clamp(min=0)
    areas_2 = (boxes_2[:, 2] - boxes_2[:, 0]).clamp(min=0) * (boxes_2[:, 3] - boxes_2[:, 1]).clamp(min=0)

    intersections_min = torch.maximum(boxes_1[:, None, :2], boxes_2[None, :, :2])
    intersections_max = torch.minimum(boxes_1[:, None, 2:], boxes_2[None, :, 2:])
    intersection_sizes = (intersections_max - intersections_min).clamp(min=0)
    intersections = intersection_sizes[..., 0] * intersection_sizes[..., 1]
    unions = areas_1[:, None] + areas_2[None, :] - intersections

    return intersections / unions.clamp(min=torch.finfo(intersections.dtype).eps)


def match_detections(torch_boxes, tt_boxes):
    if len(torch_boxes) == 0 or len(tt_boxes) == 0:
        return []

    ious = calculate_box_iou(torch_boxes, tt_boxes)
    sorted_pairs = torch.argsort(ious.flatten(), descending=True)
    used_torch_indices = set()
    used_tt_indices = set()
    matches = []

    for flat_index in sorted_pairs.tolist():
        torch_index = flat_index // tt_boxes.shape[0]
        tt_index = flat_index % tt_boxes.shape[0]
        iou = float(ious[torch_index, tt_index])

        if iou < MATCH_IOU_THRESHOLD:
            break
        if torch_index in used_torch_indices or tt_index in used_tt_indices:
            continue

        used_torch_indices.add(torch_index)
        used_tt_indices.add(tt_index)
        matches.append((torch_index, tt_index, iou))

    return matches


def get_coco_images():
    images_path = Path(os.environ.get(COCO_IMAGES_ENV, DEFAULT_COCO_IMAGES_PATH))
    image_paths = sorted(images_path.glob("*.jpg"))[:NUM_VALIDATION_IMAGES]

    if not images_path.is_dir() or not image_paths:
        pytest.skip(f"Set {COCO_IMAGES_ENV} to the downloaded COCO val2017 directory")

    return image_paths


def run_rtdetr_detection_test(device, model_version):
    image_paths = get_coco_images()
    model_name, model_class = MODEL_CONFIGS[model_version]

    image_processor = RTDetrImageProcessor.from_pretrained(model_name)
    torch_model = model_class.from_pretrained(model_name).eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model.model,
        custom_preprocessor=custom_preprocessor,
    )
    tt_model = TtRTDetrModel(
        config=torch_model.config,
        parameters=parameters,
        device=device,
        dtype=ttnn.bfloat16,
        input_height=640,
        input_width=640,
    )
    device.enable_program_cache()

    total_torch_detections = 0
    total_tt_detections = 0
    matched_ious = []
    label_matches = []
    score_errors = []

    logger.info(f"Evaluating RT-DETR v{model_version} on {len(image_paths)} COCO val2017 images")

    with Image.open(image_paths[0]) as image_file:
        first_image = image_file.convert("RGB")
    first_pixel_values = image_processor(images=first_image, return_tensors="pt").pixel_values
    tt_pixel_values = ttnn.from_torch(
        first_pixel_values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    warmup_outputs = tt_model(tt_pixel_values)
    ttnn.synchronize_device(device)
    del warmup_outputs

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_outputs = tt_model(tt_pixel_values)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    try:
        for image_index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")

            pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
            target_sizes = torch.tensor([(image.height, image.width)])

            with torch.no_grad():
                torch_outputs = torch_model(pixel_values=pixel_values)

            torch_detections = image_processor.post_process_object_detection(
                torch_outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes,
            )[0]

            host_input = ttnn.from_torch(
                pixel_values,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(host_input, tt_pixel_values, cq_id=0)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            tt_outputs = SimpleNamespace(
                logits=ttnn.to_torch(trace_outputs[4]).float(),
                pred_boxes=ttnn.to_torch(trace_outputs[5]).float(),
            )
            tt_detections = image_processor.post_process_object_detection(
                tt_outputs,
                threshold=CONFIDENCE_THRESHOLD,
                target_sizes=target_sizes,
            )[0]

            total_torch_detections += len(torch_detections["boxes"])
            total_tt_detections += len(tt_detections["boxes"])

            for torch_index, tt_index, iou in match_detections(torch_detections["boxes"], tt_detections["boxes"]):
                matched_ious.append(iou)
                label_matches.append(
                    int(torch_detections["labels"][torch_index]) == int(tt_detections["labels"][tt_index])
                )
                score_errors.append(
                    abs(float(torch_detections["scores"][torch_index]) - float(tt_detections["scores"][tt_index]))
                )

            if image_index % 100 == 0:
                logger.info(f"Processed {image_index}/{len(image_paths)} images")
    finally:
        ttnn.release_trace(device, trace_id)
        del trace_outputs
        ttnn.deallocate(tt_pixel_values)

    assert total_torch_detections > 0, "Torch postprocessing produced no detections"
    assert total_tt_detections > 0, "TTNN postprocessing produced no detections"
    assert matched_ious, "No Torch and TTNN detections matched at IoU >= 0.5"

    matched_count = len(matched_ious)
    matched_ratio = matched_count / max(total_torch_detections, total_tt_detections)
    label_agreement = sum(label_matches) / matched_count
    mean_score_error = sum(score_errors) / matched_count
    mean_iou = sum(matched_ious) / matched_count
    minimum_iou = min(matched_ious)

    logger.info(f"RT-DETR v{model_version} images evaluated: {len(image_paths)}")
    logger.info(f"RT-DETR v{model_version} Torch detections: {total_torch_detections}")
    logger.info(f"RT-DETR v{model_version} TTNN detections: {total_tt_detections}")
    logger.info(f"RT-DETR v{model_version} matched detections: {matched_count}")
    logger.info(f"RT-DETR v{model_version} matched ratio: {matched_ratio:.4f}")
    logger.info(f"RT-DETR v{model_version} label agreement: {label_agreement:.4f}")
    logger.info(f"RT-DETR v{model_version} mean matched IoU: {mean_iou:.4f}")
    logger.info(f"RT-DETR v{model_version} minimum matched IoU: {minimum_iou:.4f}")
    logger.info(f"RT-DETR v{model_version} mean confidence-score error: {mean_score_error:.4f}")

    assert matched_ratio >= MIN_MATCHED_RATIO
    assert label_agreement >= MIN_LABEL_AGREEMENT
    assert mean_iou >= MIN_MEAN_IOU
    assert mean_score_error <= MAX_MEAN_SCORE_ERROR


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1 << 26}],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_rtdetr_detection(device):
    run_rtdetr_detection_test(device, model_version=1)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "trace_region_size": 1 << 26}],
    indirect=True,
)
@pytest.mark.timeout(0)
def test_rtdetr_v2_detection(device):
    run_rtdetr_detection_test(device, model_version=2)
