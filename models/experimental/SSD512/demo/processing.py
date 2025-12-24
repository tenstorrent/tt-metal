# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
import ttnn
from PIL import Image, ImageDraw, ImageFont

from models.experimental.SSD512.common import SSD512_NUM_CLASSES, reshape_prediction_tensors
from models.experimental.SSD512.reference.voc0712 import VOC_CLASSES
from models.experimental.SSD512.reference.detection import Detect


def load_image(image_path, size=512):
    """Load and preprocess image for SSD512 inference."""
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")

    original_height, original_width = image_bgr.shape[:2]
    original_img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(original_img_rgb)

    x = cv2.resize(image_bgr, (size, size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    img_tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    original_img.original_size = (original_width, original_height)
    return img_tensor, original_img


def filter_top_detections(output_tensor: torch.Tensor, max_detections=5, min_score=0.01):
    """
    Filter reference Detect output format [batch, num_classes, top_k, 5]
    to keep only top N by confidence score.
    Returns Dict format for compatibility with draw_detections.
    """
    batch_size = output_tensor.size(0)
    num_classes = output_tensor.size(1)

    result = []
    for batch_idx in range(batch_size):
        box_list = []
        score_list = []
        label_list = []

        # Process each class
        for class_idx in range(1, num_classes):
            class_output = output_tensor[batch_idx, class_idx]
            scores = class_output[:, 0]
            boxes = class_output[:, 1:5]

            # Filter by min_score and valid detections
            valid_mask = (scores > 0) & (scores >= min_score)
            if valid_mask.any():
                valid_scores = scores[valid_mask]
                valid_boxes = boxes[valid_mask]
                valid_count = valid_mask.sum().item()

                box_list.append(valid_boxes)
                score_list.append(valid_scores)
                label_list.extend([class_idx] * valid_count)

        # If any detections were found, concatenate and sort; otherwise return empty
        if len(box_list) > 0:
            all_boxes = torch.cat(box_list, 0)
            all_scores = torch.cat(score_list, 0)
            all_labels = torch.tensor(label_list, dtype=torch.long)

            # Sort by confidence score (descending) and take top N
            sorted_indices = torch.argsort(all_scores, descending=True)
            top_indices = sorted_indices[:max_detections]

            result.append(
                {
                    "boxes": all_boxes[top_indices],
                    "scores": all_scores[top_indices],
                    "labels": all_labels[top_indices],
                }
            )
        else:
            result.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                }
            )

    return result


def draw_detections(image, detections, output_path, model_name):
    """Draw bounding boxes and labels on image."""
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    if len(detections) > 0:
        det = detections[0]
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if hasattr(image, "original_size"):
            img_width, img_height = image.original_size
        else:
            img_width, img_height = image.size
        scale = torch.tensor([img_width, img_height, img_width, img_height], dtype=torch.float32)

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            box_scaled = (box * scale).cpu()
            x1, y1, x2, y2 = box_scaled[0].item(), box_scaled[1].item(), box_scaled[2].item(), box_scaled[3].item()

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            if x1 >= x2 or y1 >= y2:
                continue

            class_idx = label.item()
            voc_class_idx = class_idx - 1
            if 0 <= voc_class_idx < len(VOC_CLASSES):
                class_name = VOC_CLASSES[voc_class_idx]
            else:
                class_name = f"Class {class_idx}"
            color = colors[voc_class_idx % len(colors)] if voc_class_idx >= 0 else colors[0]

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label_text = f"{class_name}: {score.item():.2f}"
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)

    image.save(output_path)


def run_ssd_inference(model, image_tensor, priors, device, conf_thresh=0.01, nms_thresh=0.45, top_k=200):
    """Run SSD512 model forward pass and detection on TTNN device."""
    ttnn.synchronize_device(device)

    image_tensor_permuted = image_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(
        image_tensor_permuted, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)

    tt_loc_preds, tt_conf_preds = model(device, ttnn_input)

    memory_config = ttnn.DRAM_MEMORY_CONFIG
    loc = reshape_prediction_tensors(tt_loc_preds, memory_config)
    conf = reshape_prediction_tensors(tt_conf_preds, memory_config)

    batch_size = 1
    num_priors = loc.shape[1] // 4

    loc = ttnn.experimental.view(loc, (batch_size, num_priors, 4))
    conf = ttnn.experimental.view(conf, (batch_size, num_priors, SSD512_NUM_CLASSES))

    conf = ttnn.to_layout(conf, ttnn.TILE_LAYOUT, memory_config=memory_config)
    conf = ttnn.softmax(conf, dim=-1, memory_config=memory_config)
    conf = ttnn.to_layout(conf, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)

    # Convert TTNN tensors to torch for post-processing
    loc_torch = ttnn.to_torch(loc)
    conf_torch = ttnn.to_torch(conf)

    detect = Detect(
        num_classes=SSD512_NUM_CLASSES,
        size=512,
        bkg_label=0,
        top_k=top_k,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh,
    )

    # Reference Detect returns [batch, num_classes, top_k, 5] tensor
    output_tensor = detect(loc_torch, conf_torch, priors)

    ttnn.deallocate(loc)
    ttnn.deallocate(conf)
    ttnn.synchronize_device(device)

    return output_tensor
