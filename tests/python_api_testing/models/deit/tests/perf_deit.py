from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from datasets import load_dataset
from loguru import logger
import pytest
import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report
from tt.deit_for_image_classification_with_teacher import deit_for_image_classification_with_teacher

BATCH_SIZE = 1

@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (0.24,
            9,
        ),
    ),
)
def test_perf(use_program_cache, expected_inference_time, expected_compile_time):
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    comments = "distilled-patch16-wteacher"

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    HF_model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    inputs = image_processor(image, return_tensors="pt")

    tt_inputs = torch_to_tt_tensor_rm(
        inputs["pixel_values"], device, put_on_device=False
    )
    tt_model = deit_for_image_classification_with_teacher(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = HF_model(**inputs).logits
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_inputs)[0]
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_inputs)[0]
        tt_lib.device.Synchronize()
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    tt_lib.device.CloseDevice(device)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_report("deit", BATCH_SIZE, first_iter_time, second_iter_time, comments, cpu_time)
    logger.info(f"deit {comments} inference time: {second_iter_time}")
    logger.info(f"deit {comments} compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, f"deit {comments} is too slow"
    assert compile_time < expected_compile_time, "deit compile time is too slow"
