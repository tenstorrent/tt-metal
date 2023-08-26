from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import RobertaForSequenceClassification
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from loguru import logger
import pytest
import tt_lib
from tt_models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, Profiler, \
         disable_persistent_kernel_cache, enable_persistent_kernel_cache, prep_report

from python_api_testing.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.roberta.roberta_for_sequence_classification import (
    TtRobertaForSequenceClassification,
)

BATCH_SIZE = 1


def run_perf_roberta(expected_inference_time, expected_compile_time):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "Base Emotion"
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion"
        )
        model = RobertaForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-emotion"
        )
        model.eval()

        # Tt roberta
        tt_model = TtRobertaForSequenceClassification(
            config=model.config,
            base_address="",
            device=device,
            state_dict=model.state_dict(),
            reference_model=model,
        )
        tt_model.eval()

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        profiler.start(cpu_key)
        torch_output = model(**inputs).logits
        profiler.end(cpu_key)

        tt_attention_mask = torch.unsqueeze(inputs.attention_mask, 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

        profiler.start(first_key)

        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits
        tt_lib.device.Synchronize()
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(inputs.input_ids, tt_attention_mask).logits
        tt_lib.device.Synchronize()
        profiler.end(second_key)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    tt_lib.device.CloseDevice(device)
    prep_report(
        model_name="roberta",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    compile_time = first_iter_time - second_iter_time

    logger.info(f"roberta {comments} inference time: {second_iter_time}")
    logger.info(f"roberta compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, f"roberta {comments} is too slow"
    assert compile_time < expected_compile_time, f"roberta {comments} compile time is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (0.48,
         17,
        ),
    ),
)
def test_perf_bare_metal(use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_roberta(expected_inference_time, expected_compile_time)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (0.75,
         17.5,
        ),
    ),
)
def test_perf_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time):
    run_perf_roberta(expected_inference_time, expected_compile_time)
