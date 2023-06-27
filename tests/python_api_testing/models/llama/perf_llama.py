from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from python_api_testing.models.llama.llama_model import TtLlamaModel
import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, Profiler
from utility_functions_new import disable_compile_cache, enable_compile_cache
from utility_functions_new import prep_report

BATCH_SIZE = 1


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, num_decoders, max_position_embeddings, on_weka",
    (
        (
            "decapoda-research/llama-7b-hf",
            "hf-internal-testing/llama-tokenizer",
            1,
            64,
            2,
            2048,
            False,
        ),
    ),
)
def test_perf(
    model_version,
    tokenizer_version,
    batch,
    seq_len,
    num_decoders,
    max_position_embeddings,
    on_weka,
):
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

    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # get only llama model (no linear layer at the end)
    llama_model = hugging_face_reference_model.get_decoder()

    batch = BATCH_SIZE
    seq_len = seq_len
    if 1:
        llama_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        oneseq = [torch.arange(seq_len)] * batch
        llama_input = torch.stack(oneseq)
        llama_input = llama_input.reshape(batch, seq_len)

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    num_decoders = num_decoders
    base_url = "model.layers"
    max_position_embeddings = max_position_embeddings

    tt_llama_model = TtLlamaModel(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        pytorch_out = llama_model(llama_input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_llama_model(llama_input)

        profiler.end(first_key)

        enable_compile_cache()
        profiler.start(second_key)
        tt_out = tt_llama_model(llama_input)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_report("llama", BATCH_SIZE, first_iter_time, second_iter_time, "7B", cpu_time)
