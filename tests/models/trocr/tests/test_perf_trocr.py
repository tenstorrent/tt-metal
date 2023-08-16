import torch
from loguru import logger
from PIL import Image

import pytest
import tt_lib
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.utility_functions import Profiler, prep_report

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from models.trocr.tt.trocr import trocr_causal_llm

BATCH_SIZE = 1


def test_perf(expected_inference_time, expected_compile_time):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten"
    )
    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    pixel_values = processor(
        images=iam_ocr_sample_input, return_tensors="pt"
    ).pixel_values

    tt_model = trocr_causal_llm(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = model.generate(pixel_values)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model.generate(pixel_values)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model.generate(pixel_values)
        profiler.end(second_key)

        first_iter_time = profiler.get(first_key)
        second_iter_time = profiler.get(second_key)
        cpu_time = profiler.get(cpu_key)

        prep_report(
            "trocr",
            BATCH_SIZE,
            first_iter_time,
            second_iter_time,
            expected_compile_time,
            expected_inference_time,
            "causal_llm",
            cpu_time,
        )
        compile_time = first_iter_time - second_iter_time

        logger.info(f"trocr inference time: {second_iter_time}")
        logger.info(f"trocr compile time: {compile_time}")
        tt_lib.device.CloseDevice(device)
        assert second_iter_time < expected_inference_time, "trocr is too slow"
        assert compile_time < expected_compile_time, "trocr compile time is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            199.57233452796936,
            16.2,
        ),
    ),
)
def test_perf_bare_metal(
    use_program_cache, expected_inference_time, expected_compile_time
):
    test_perf(expected_inference_time, expected_compile_time)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            199.57233452796936,
            16.2,
        ),
    ),
)
def test_perf_virtual_machine(
    use_program_cache, expected_inference_time, expected_compile_time
):
    test_perf(expected_inference_time, expected_compile_time)
