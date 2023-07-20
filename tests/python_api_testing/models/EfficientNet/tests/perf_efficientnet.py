import tt_lib
import torch
from loguru import logger
import torchvision
from datasets import load_dataset

from tests.python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    Profiler,
    prep_report
)
from models.EfficientNet.tt.efficientnet_model import efficientnet_b0
from models.utility_functions import disable_compile_cache, enable_compile_cache


def make_input_tensor(imagenet_sample_input, resize=256, crop=224):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return transform(imagenet_sample_input)


def test_perf_efficientnet_b0(imagenet_sample_input):
    profiler = Profiler()
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    test_input = make_input_tensor(imagenet_sample_input)

    hf_model = torchvision.models.efficientnet_b0(pretrained=True)
    tt_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    tt_model = efficientnet_b0(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        hf_model(test_input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_model(tt_input)
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_model(tt_input)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report(
        "EfficientNet", 1, first_iter_time, second_iter_time, "b0", cpu_time
    )
