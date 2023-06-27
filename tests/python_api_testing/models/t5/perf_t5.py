from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from transformers import AutoTokenizer, T5Tokenizer, T5Model
import torch
from datasets import load_dataset
from loguru import logger
import json

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, Profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report

from python_api_testing.models.t5.t5_model import TtT5Model

BATCH_SIZE = 1


def test_perf():
    profiler = Profiler()
    disable_compile_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    use_attention_mask = True
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=32)
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    tt_model = TtT5Model(config, hf_reference_model.state_dict(), device)

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    tokenized = tokenizer(
        input_sentance, padding="max_length", max_length=32, return_tensors="pt"
    )  # Batch size 1

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask if use_attention_mask else None

    decoder_input_sentence = "Studies show that"
    tokenized = tokenizer(
        decoder_input_sentence, padding="max_length", max_length=32, return_tensors="pt"
    )  # Batch size 1

    decoder_input_ids = tokenized.input_ids
    decoder_attention_mask = tokenized.attention_mask if use_attention_mask else None

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = hf_reference_model._shift_right(decoder_input_ids)

    with torch.no_grad():
        # PyTorch forward pass
        profiler.start(cpu_key)
        pt_out = hf_reference_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_model_outputs = tt_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        profiler.end(first_key)

        enable_compile_cache()

        profiler.start(second_key)
        tt_model_outputs = tt_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report("t5", BATCH_SIZE, first_iter_time, second_iter_time, "small", cpu_time)
