import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import tt_lib
import pytest

from tt_models.mnist.tt.mnist_model import mnist_model
from tt_models.utility_functions import (
    torch2tt_tensor,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)

from tt_models.utility_functions import profiler, prep_report


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.1,
            9.0,
        ),
    ),
)
def test_perf(use_program_cache, expected_inference_time, expected_compile_time):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    dataloader = DataLoader(test_dataset, batch_size=1)

    # Load model
    tt_model, pt_model = mnist_model(device)

    with torch.no_grad():
        test_input, _ = next(iter(dataloader))
        tt_input = torch2tt_tensor(
            test_input, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        profiler.start(cpu_key)
        pt_out = pt_model(test_input)
        tt_lib.device.Synchronize()
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_model(tt_input)
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_out

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_out = tt_model(tt_input)
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_out

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    tt_lib.device.CloseDevice(device)

    prep_report("mnist", 1, first_iter_time, second_iter_time, "", cpu_time)

    logger.info(f"mnist inference time: {second_iter_time}")
    logger.info(f"mnist compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, "mnist is too slow"
    assert compile_time < expected_compile_time, "mnist compile time is too slow"
