from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import numpy as np
import ast
from torchvision import models
from loguru import logger
from PIL import Image
import pytest

import tt_lib
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
)
from vgg import *


_batch_size = 1


def run_vgg_inference(image_path, pcc, PERF_CNT=1):
    im = Image.open(image_path)
    im = im.resize((224, 224))

    # Apply the transformation to the random image and Add an extra dimension at the beginning
    # to match the desired shape of 3x224x224
    image = transforms.ToTensor()(im).unsqueeze(0)

    batch_size = _batch_size
    with torch.no_grad():
        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        host = tt_lib.device.GetHost()

        torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        torch_vgg.eval()

        state_dict = torch_vgg.state_dict()
        # TODO: enable conv on tt device after adding fast dtx transform
        tt_vgg = vgg16(device, host, state_dict, disable_conv_on_tt_device=True)

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

        enable_compile_cache()

        logger.info(f"\nRunning the tt_vgg model for {PERF_CNT} iterations . . . ")
        for i in range(PERF_CNT):
            profiler.start("\nAverage execution time of tt_vgg model")
            tt_output = tt_vgg(tt_image)
            profiler.end("\nAverage execution time of tt_vgg model")

        with open("imagenet_class_labels.txt", "r") as file:
            class_labels = ast.literal_eval(file.read())

        tt_output = tt_output.to(host)
        tt_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())

        logger.info(
            f"Correct Output: {class_labels[torch.argmax(torch_output).item()]}"
        )
        logger.info(
            f"Predicted Output: {class_labels[torch.argmax(tt_output).item()]}\n"
        )
        file.close()
        pcc_passing, pcc_output = comp_pcc(torch_output, tt_output, pcc)
        logger.info(f"Output {pcc_output}")
        assert pcc_passing, f"Model output does not meet PCC requirement {pcc}."

        tt_lib.device.CloseDevice(device)

        profiler.print()
        assert profiler.get("tt_vgg model") < 30.0


@pytest.mark.parametrize(
    "path_to_image, pcc, iter",
    (("sample_image.JPEG", 0.99, 2),),
)
def test_vgg_inference(path_to_image, pcc, iter):
    disable_compile_cache()
    run_vgg_inference(path_to_image, pcc, iter)


if __name__ == "__main__":
    run_vgg_inference("sample_image.JPEG", 0.99, 2)
