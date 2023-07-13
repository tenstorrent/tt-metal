from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import WhisperModel, AutoFeatureExtractor
import torch
from datasets import load_dataset
from loguru import logger
import pytest

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, Profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report
from python_api_testing.models.whisper.whisper_model import TtWhisperModel

from python_api_testing.models.whisper.whisper_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)

BATCH_SIZE = 1

@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (74,
         22,
        ),
    ),
)
def test_perf(use_program_cache, expected_inference_time, expected_compile_time):
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

    pytorch_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = pytorch_model.config

    pytorch_model.eval()
    state_dict = pytorch_model.state_dict()

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # original from HF example should be: seq_len = 3000, when max_source_positions=1500
    input_features = inputs.input_features

    dec_seq_len = 32
    decoder_input_ids = (
        torch.tensor(
            [
                [
                    1,
                ]
                * dec_seq_len
            ]
        )
        * pytorch_model.config.decoder_start_token_id
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        pytorch_output = pytorch_model(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        profiler.end(cpu_key)

    tt_whisper = TtWhisperModel(
        state_dict=state_dict, device=device, config=pytorch_model.config
    )
    tt_whisper.eval()

    with torch.no_grad():
        input_features = torch2tt_tensor(input_features, device)

        profiler.start(first_key)
        ttm_output = tt_whisper(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        ttm_output = tt_whisper(
            input_features=input_features, decoder_input_ids=decoder_input_ids
        )
        tt_lib.device.Synchronize()
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    tt_lib.device.CloseDevice(device)


    prep_report(
        "whisper", BATCH_SIZE, first_iter_time, second_iter_time, "tiny", cpu_time
    )
    logger.info(f"whisper tiny inference time: {second_iter_time}")
    logger.info(f"whisper compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, "whisper tiny is too slow"
    assert compile_time < expected_compile_time, "whisper compile time is too slow"
