# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch

from torchvision import transforms, datasets
from loguru import logger
import ttnn

from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.experimental.mnist.tt.mnist_model import mnist_model


def test_mnist_inference(model_location_generator):
    device = ttnn.open_device(0)

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)

    # Load model
    tt_model, _ = mnist_model(device, model_location_generator)

    with torch.no_grad():
        input_img, _ = test_dataset[0]

        input_img_path = str(ROOT / "input_img.jpg")
        input_img.save(input_img_path)
        logger.info(f"Input image saved to {input_img_path}")

        test_input = transforms.ToTensor()(input_img)
        test_input = torch2tt_tensor(test_input, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        tt_output = tt_model(test_input)
        tt_output = tt2torch_tensor(tt_output)
        tt_output = tt_output.squeeze()
        ttnn.close_device(device)

    logger.info(f"Tt prediction: {tt_output.topk(10).indices[0]}")
