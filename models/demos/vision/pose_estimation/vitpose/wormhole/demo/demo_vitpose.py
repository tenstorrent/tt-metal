# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
ViTPose-B demo: human pose estimation on a sample image.

Requires:
  pip install supervision

Usage (inside Docker container):
  pytest models/demos/vision/pose_estimation/vitpose/wormhole/demo/demo_vitpose.py -v
"""

import numpy as np
import pytest
import requests
import torch
from PIL import Image

import ttnn
from models.demos.vision.pose_estimation.vitpose.common.common import load_torch_model
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose import VitPose


COCO_KEYPOINT_EDGES = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

COCO_KEYPOINT_NAMES = [
    "Nose",
    "L_Eye",
    "R_Eye",
    "L_Ear",
    "R_Ear",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
]


def extract_keypoints_from_heatmaps(heatmaps, bbox, image_size):
    """
    Extract keypoint coordinates from heatmaps and map back to image coordinates.

    Args:
        heatmaps: (17, 64, 48) numpy array
        bbox: (x, y, w, h) bounding box in COCO format
        image_size: (height, width) of original image

    Returns:
        keypoints: (17, 2) array of (x, y) in image coordinates
        scores: (17,) confidence scores
    """
    num_keypoints, hm_h, hm_w = heatmaps.shape
    keypoints = np.zeros((num_keypoints, 2))
    scores = np.zeros(num_keypoints)

    bx, by, bw, bh = bbox

    for k in range(num_keypoints):
        hm = heatmaps[k]
        flat_idx = np.argmax(hm)
        y, x = divmod(flat_idx, hm_w)
        scores[k] = hm[y, x]

        x_img = bx + (x / hm_w) * bw
        y_img = by + (y / hm_h) * bh
        keypoints[k] = [x_img, y_img]

    return keypoints, scores


def preprocess_person_crop(image, bbox, target_size=(256, 192)):
    """
    Crop and resize a person bounding box from the image.
    Simplified preprocessing (no affine transform).

    Args:
        image: PIL Image
        bbox: (x, y, w, h) COCO format
        target_size: (height, width)

    Returns:
        pixel_values: (1, 3, 256, 192) torch tensor normalized
    """
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    crop = image.crop((x1, y1, x2, y2))
    crop = crop.resize((target_size[1], target_size[0]), Image.BILINEAR)

    pixel_values = np.array(crop, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pixel_values = (pixel_values - mean) / std
    pixel_values = torch.from_numpy(pixel_values).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)

    return pixel_values


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_vitpose_demo(device, batch_size):
    model = load_torch_model()
    state_dict = model.state_dict()

    url = "http://images.cocodataset.org/val2017/000000000785.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    bbox = [0, 0, image.width, image.height]
    pixel_values = preprocess_person_crop(image, bbox)

    tt_model = VitPose(state_dict, device, batch_size=batch_size)
    tt_input = VitPose.prepare_input(pixel_values, device)
    tt_output = tt_model(tt_input)
    tt_output = ttnn.to_torch(tt_output)

    heatmaps = tt_output.reshape(batch_size, 64, 48, 17).permute(0, 3, 1, 2)
    heatmaps_np = heatmaps[0].float().numpy()

    keypoints, scores = extract_keypoints_from_heatmaps(heatmaps_np, bbox, (image.height, image.width))

    print("\n=== ViTPose-B Inference Results ===")
    print(f"Image size: {image.width}x{image.height}")
    for i, (name, (x, y), score) in enumerate(zip(COCO_KEYPOINT_NAMES, keypoints, scores)):
        print(f"  {name:>12s}: ({x:6.1f}, {y:6.1f})  score={score:.3f}")

    detected = np.sum(scores > 0.3)
    print(f"\nDetected {detected}/17 keypoints (score > 0.3)")

    assert heatmaps.shape == (batch_size, 17, 64, 48), f"Unexpected heatmap shape: {heatmaps.shape}"
