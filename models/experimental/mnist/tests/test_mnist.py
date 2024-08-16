# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
from numpy import argmax

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
    skip_for_wormhole_b0,
)
from models.experimental.mnist.tt.mnist_model import mnist_model


@skip_for_wormhole_b0()
def test_mnist_inference(device, model_location_generator):
    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=1)

    # Load model
    tt_model, pt_model = mnist_model(device, model_location_generator)

    with torch.no_grad():
        test_input, _ = next(iter(dataloader))
        tt_input = torch2tt_tensor(test_input, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        pt_output = pt_model(test_input)
        tt_output = tt_model(tt_input)
        tt_output = tt2torch_tensor(tt_output)

    pcc_passing, pcc_output = comp_pcc(pt_output, tt_output, 0.99)
    logger.info(f"Output {pcc_output}")

    assert pcc_passing, f"Model output does not meet PCC requirement {0.99}."

    assert (
        tt_output.topk(10).indices == pt_output.topk(10).indices
    ).all(), "The outputs from device and pytorch must have the same topk indices"

    # Check that the scale of each output is the same
    tt_out_largest_val_position = argmax(tt_output.view(-1).tolist())
    pytorch_out_largest_val_position = argmax(pt_output.tolist())

    assert (
        tt_out_largest_val_position == pytorch_out_largest_val_position
    ), "The largest value in both TT and PT outputs must be in the same position"
