from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from torchvision import models
import torch
from datasets import load_dataset
from loguru import logger

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, Profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report

from tt.vgg import *

BATCH_SIZE = 1


def test_perf(imagenet_sample_input):
    profiler = Profiler()
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    image = imagenet_sample_input

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    tt_image = tt_lib.tensor.Tensor(
        image.reshape(-1).tolist(),
        get_shape(image.shape),
        tt_lib.tensor.DataType.BFLOAT16,
        tt_lib.tensor.Layout.ROW_MAJOR,
    )

    tt_vgg = vgg16(device, host, disable_conv_on_tt_device=True)

    torch_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_vgg.eval()

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_vgg(image).unsqueeze(1).unsqueeze(1)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_vgg(tt_image)
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output = tt_vgg(tt_image)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report("vgg", BATCH_SIZE, first_iter_time, second_iter_time, "vgg16", cpu_time)
