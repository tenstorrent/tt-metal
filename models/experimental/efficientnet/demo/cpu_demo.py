# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import cv2
import torch
import torchvision
from loguru import logger
from models.experimental.efficientnet.demo.demo_utils import (
    load_imagenet_labels,
    download_images,
    preprocess,
)


def load_efficientnet_model():
    """
    Load the pre-trained EfficientNetB0 model.
    """
    model = torchvision.models.efficientnet_v2_s(pretrained=True)
    model.eval()
    return model


def test_cpu_demo_v2_s():
    img_path = ROOT / "input_image.jpg"
    download_images(img_path)

    model = load_efficientnet_model()
    categories = load_imagenet_labels()
    transform = preprocess()

    image = cv2.imread(str(img_path))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Check the top 5 categories that are predicted.
    top5_prob, top5_catid = torch.topk(probabilities, 3)

    for i in range(top5_prob.size(0)):
        cv2.putText(
            image,
            f"{top5_prob[i].item()*100:.3f}%",
            (15, (i + 1) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{categories[top5_catid[i]]}",
            (160, (i + 1) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        logger.info(categories[top5_catid[i]], top5_prob[i].item())

    out_path = str(ROOT / "out_image.jpg")
    cv2.imwrite(out_path, image)
    logger.info(f"Output image saved to {out_path}")
