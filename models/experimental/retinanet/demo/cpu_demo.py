# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from loguru import logger


def test_cpu_demo(imagenet_sample_input, reset_seeds):
    image = imagenet_sample_input

    model = retinanet_resnet50_fpn_v2(pretrained=True)
    model.eval()

    with torch.no_grad():
        predictions = model(image)

    # Post-process the predictions
    boxes = predictions[0]["boxes"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()

    # Print the detected objects
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score > 0.9:  # Adjust this threshold as needed
            logger.info(f"Object {i}: ")
            logger.info(f"Label: {label}")
            logger.info(f"Score: {score}")
            logger.info(f"Box: {box}")
