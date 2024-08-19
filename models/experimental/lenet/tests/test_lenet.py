# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from models.experimental.lenet.lenet_utils import load_torch_lenet, prepare_image
from models.experimental.lenet.tt.lenet import lenet5
from models.utility_functions import comp_pcc, torch2tt_tensor


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_lenet_inference(device, pcc, mnist_sample_input, model_location_generator, reset_seeds):
    num_classes = 10
    batch_size = 1
    with torch.no_grad():
        # Initialize Torch model
        pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
        torch_LeNet, _ = load_torch_lenet(pt_model_path, num_classes)

        # Initialize TT model
        tt_lenet = lenet5(num_classes, device, model_location_generator)

        image = prepare_image(mnist_sample_input)

        torch_output = torch_LeNet(image).unsqueeze(1).unsqueeze(1)
        _, torch_predicted = torch.max(torch_output.data, -1)

        tt_image = torch2tt_tensor(image, device, ttnn.ROW_MAJOR_LAYOUT)

        tt_output = tt_lenet(tt_image)
        tt_output = tt_output.cpu()
        tt_output = tt_output.to_torch()

        _, tt_predicted = torch.max(tt_output.data, -1)

        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."
