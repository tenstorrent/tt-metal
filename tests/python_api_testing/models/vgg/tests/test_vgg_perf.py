from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from torchvision import models
from loguru import logger
import pytest

import tt_lib
from tt_models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    comp_pcc,
)
from tt.vgg import *


_batch_size = 1

@pytest.mark.parametrize("pcc, PERF_CNT", ((0.99, 2),),)
def test_vgg_inference(pcc, PERF_CNT, imagenet_sample_input, imagenet_label_dict):
    disable_persistent_kernel_cache()
    image = imagenet_sample_input
    class_labels = imagenet_label_dict
    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)


        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        torch_vgg.eval()

        state_dict = torch_vgg.state_dict()
        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device=device, host=host, disable_conv_on_tt_device=True)

        profiler.enable()

        profiler.start("\nExec time of reference model")
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        profiler.end("\nExec time of reference model")

        tt_image = tt_lib.tensor.Tensor(
            image.reshape(-1).tolist(),
            get_shape(image.shape),
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.Layout.ROW_MAJOR,
        )

        profiler.start("\nExecution time of tt_vgg first run")
        tt_output = tt_vgg(tt_image)
        profiler.end("\nExecution time of tt_vgg first run")

        enable_persistent_kernel_cache()

        logger.info(f"\nRunning the tt_vgg model for {PERF_CNT} iterations . . . ")
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_vgg model")
            tt_output = tt_vgg(tt_image)
            profiler.end("\nAverage execution time of tt_vgg model")

        tt_output = tt_output.cpu()
        tt_output = torch.Tensor(tt_output.to_torch()).reshape(tt_output.shape())

        logger.info(
            f"Correct Output: {class_labels[torch.argmax(torch_output).item()]}"
        )
        logger.info(
            f"Predicted Output: {class_labels[torch.argmax(tt_output).item()]}\n"
        )
        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        tt_lib.device.CloseDevice(device)

        profiler.print()
        assert profiler.get("tt_vgg model") < 30.0
