from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from loguru import logger
import pytest
import tt_lib

from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
)
from tt.lenet import lenet5
from lenet_utils import load_torch_lenet, prepare_image
from utility_functions_new import torch2tt_tensor


@pytest.mark.parametrize(
    "pcc, PERF_CNT",
    ((0.99, 2),),
)
def test_lenet_perf_inference(
    pcc, PERF_CNT, mnist_sample_input, model_location_generator, reset_seeds
):
    disable_compile_cache()
    image = prepare_image(mnist_sample_input)
    num_classes = 10
    batch_size = 1

    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        # Initialize Torch model
        pt_model_path = model_location_generator("tt_dnn-models/LeNet/model.pt")
        torch_LeNet, _ = load_torch_lenet(pt_model_path, num_classes)

        # Initialize TT model
        tt_lenet = lenet5(num_classes, device, host, model_location_generator)

        profiler.enable()

        profiler.start("\nExec time of reference model")
        torch_output = torch_LeNet(image).unsqueeze(1).unsqueeze(1)
        _, torch_predicted = torch.max(torch_output.data, -1)
        profiler.end("\nExec time of reference model")

        tt_image = torch2tt_tensor(image, device, tt_lib.tensor.Layout.ROW_MAJOR)

        profiler.start("\nExecution time of tt_vgg first run")
        tt_output = tt_lenet(tt_image)
        profiler.end("\nExecution time of tt_vgg first run")

        enable_compile_cache()

        logger.info(f"\nRunning the tt_vgg model for {PERF_CNT} iterations . . . ")
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_vgg model")
            tt_output = tt_lenet(tt_image)
            profiler.end("\nAverage execution time of tt_vgg model")

        tt_output = tt_output.to(host)
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
        _, tt_predicted = torch.max(tt_output.data, -1)
        logger.info(f"Correct Output: {torch_predicted[0][0][0]}")
        logger.info(f"Predicted Output: {tt_predicted[0][0][0]}\n")
        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        tt_lib.device.CloseDevice(device)

        profiler.print()
