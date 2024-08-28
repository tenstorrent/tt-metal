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

import ast
import cv2
import torch
import ttnn
import torchvision
from loguru import logger
from datasets import load_dataset

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)


def preprocess():
    """
    Define the transform for the input image/frames.
    Resize, crop, convert to tensor, and apply ImageNet normalization stats.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def download_images(img_path):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image.save(img_path)
    logger.info(f"Input image saved to {img_path}")


def load_imagenet_labels():
    imagenet_class_labels_path = ROOT / "imagenet_class_labels.txt"

    with open(ROOT / "imagenet_class_labels.txt", "r") as file:
        class_labels = ast.literal_eval(file.read())

        categories = [class_labels[key] for key in sorted(class_labels.keys(), reverse=False)]

    return categories


def run_gs_demo(efficientnet_model_constructor):
    device = ttnn.open_device(0)

    ttnn.SetDefaultDevice(device)

    img_path = ROOT / "input_image.jpg"
    download_images(img_path)

    model = efficientnet_model_constructor(device)
    categories = load_imagenet_labels()
    transform = preprocess()

    image = cv2.imread(str(img_path))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        input_batch = torch2tt_tensor(input_batch, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        output = model(input_batch)
        output = tt2torch_tensor(output)
        output = output.squeeze(0).squeeze(0)

    ttnn.close_device(device)
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
