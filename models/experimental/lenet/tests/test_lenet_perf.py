# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import pytest
import ttnn

from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    comp_pcc,
    torch2tt_tensor,
)

from models.experimental.lenet.tt.lenet import lenet5
from models.experimental.lenet.lenet_utils import load_torch_lenet, prepare_image


@pytest.mark.parametrize(
    "pcc, PERF_CNT",
    ((0.99, 2),),
)
def test_lenet_perf_inference(device, pcc, PERF_CNT, mnist_sample_input, model_location_generator, reset_seeds):
    disable_persistent_kernel_cache()
    image = prepare_image(mnist_sample_input)
    num_classes = 10
    batch_size = 1

    with torch.no_grad():
        # Initialize Torch model
        pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
        torch_LeNet, _ = load_torch_lenet(pt_model_path, num_classes)

        # Initialize TT model
        tt_lenet = lenet5(num_classes, device, model_location_generator)

        profiler.enable()

        profiler.start("\nExec time of reference model")
        torch_output = torch_LeNet(image).unsqueeze(1).unsqueeze(1)
        _, torch_predicted = torch.max(torch_output.data, -1)
        profiler.end("\nExec time of reference model")

        tt_image = torch2tt_tensor(image, device, ttnn.ROW_MAJOR_LAYOUT)

        profiler.start("\nExecution time of tt_vgg first run")
        tt_output = tt_lenet(tt_image)
        profiler.end("\nExecution time of tt_vgg first run")

        enable_persistent_kernel_cache()

        logger.info(f"\nRunning the tt_vgg model for {PERF_CNT} iterations . . . ")
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_vgg model")
            tt_output = tt_lenet(tt_image)
            profiler.end("\nAverage execution time of tt_vgg model")

        tt_output = tt_output.cpu()
        tt_output = tt_output.to_torch()
        _, tt_predicted = torch.max(tt_output.data, -1)
        logger.info(f"Correct Output: {torch_predicted[0][0][0]}")
        logger.info(f"Predicted Output: {tt_predicted[0][0][0]}\n")
        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        profiler.print()
