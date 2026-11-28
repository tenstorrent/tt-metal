# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np


def load_ground_truth_labels(label_file):
    with open(label_file, "r") as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        line = line.strip().split()
        labels.append({"class": int(line[0]), "bbox": [float(coord) for coord in line[1:]]})
    return labels


# Function to compute IoU (Intersection over Union) between two bounding boxes
def calculate_iou(box1, box2):
    # Extract coordinates of bounding boxes
    b1_x, b1_y, b1_w, b1_h = box1
    b2_x, b2_y, b2_w, b2_h = box2

    # Calculate coordinates of intersection area
    x1 = max(b1_x - b1_w / 2, b2_x - b2_w / 2)
    y1 = max(b1_y - b1_h / 2, b2_y - b2_h / 2)
    x2 = min(b1_x + b1_w / 2, b2_x + b2_w / 2)
    y2 = min(b1_y + b1_h / 2, b2_y + b2_h / 2)

    # Calculate area of intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate area of union
    union = b1_w * b1_h + b2_w * b2_h - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-16)
    return iou


# Compute Average Precision (AP) for a single class
def calculate_ap(predictions, ground_truth, iou_threshold=0.5):
    """VOC 11-point AP. ground_truth is a list of per-image GT lists; a prediction may carry
    "image_id" (index into ground_truth) to restrict matching to its own image. Each GT object
    matches at most one detection; later detections on it count as false positives."""
    # Sort predictions by confidence score in descending order
    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    true_positives = np.zeros(len(predictions))
    false_positives = np.zeros(len(predictions))
    num_ground_truth = sum(len(gt_list) for gt_list in ground_truth)
    matched = set()

    for i, prediction in enumerate(predictions):
        iou_max = 0
        gt_match = None

        image_id = prediction.get("image_id")
        images = [image_id] if image_id is not None else range(len(ground_truth))

        # Match prediction with ground truth with highest IoU
        for j in images:
            for k, gt in enumerate(ground_truth[j]):
                iou = calculate_iou(prediction["bbox"], gt["bbox"])
                if iou > iou_max:
                    iou_max = iou
                    gt_match = (j, k)

        # True positive if IoU exceeds threshold, class matches and that GT is not already matched
        if (
            gt_match is not None
            and iou_max >= iou_threshold
            and prediction["class"] == ground_truth[gt_match[0]][gt_match[1]]["class"]
            and gt_match not in matched
        ):
            true_positives[i] = 1
            matched.add(gt_match)
        else:
            false_positives[i] = 1

    if num_ground_truth == 0:
        return 0.0

    # Compute precision and recall
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / num_ground_truth
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-16)

    # Compute Average Precision (AP) using precision-recall curve
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap = ap + p / 11

    return ap
