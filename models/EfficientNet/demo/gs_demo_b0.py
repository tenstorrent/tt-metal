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
import tt_lib
import torchvision
from loguru import logger
from datasets import load_dataset

from models.EfficientNet.tt.efficientnet_model import efficientnet_b0
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
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return transform


def download_images(img_path):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    image.save(img_path)


def run_gs_demo(efficientnet_model_constructor, imagenet_label_dict):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    img_path = ROOT / "input_image.jpg"
    download_images(img_path)

    model = efficientnet_model_constructor(device)
    categories = [
        imagenet_label_dict[key]
        for key in sorted(imagenet_label_dict.keys(), reverse=False)
    ]
    transform = preprocess()

    image = cv2.imread(str(img_path))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        input_batch = torch2tt_tensor(
            input_batch, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        output = model(input_batch)
        output = tt2torch_tensor(output)
        output = output.squeeze(0).squeeze(0)

    tt_lib.device.CloseDevice(device)
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

    cv2.imwrite(str(ROOT / "out_image.jpg"), image)


def test_gs_demo_b0(imagenet_label_dict):
    run_gs_demo(efficientnet_b0, imagenet_label_dict)
