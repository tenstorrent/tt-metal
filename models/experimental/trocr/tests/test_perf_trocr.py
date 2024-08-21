# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from loguru import logger
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    Profiler,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.trocr.tt.trocr import trocr_causal_llm

BATCH_SIZE = 1


def test_perf(expected_inference_time, expected_compile_time, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    iam_ocr_sample_input = Image.open("models/sample_data/iam_ocr_image.jpg")
    pixel_values = processor(images=iam_ocr_sample_input, return_tensors="pt").pixel_values

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

        prep_perf_report(
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
def test_perf_bare_metal(device, use_program_cache, expected_inference_time, expected_compile_time):
    test_perf(expected_inference_time, expected_compile_time, device)


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
def test_perf_virtual_machine(device, use_program_cache, expected_inference_time, expected_compile_time):
    test_perf(expected_inference_time, expected_compile_time, device)
