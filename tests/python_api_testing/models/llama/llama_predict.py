from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from loguru import logger
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from libs import tt_lib as ttl
from typing import List, Optional, Tuple, Union

from transformers import T5Tokenizer, T5Model, AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast, CausalLMOutputWithPast
from collections import OrderedDict

from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax
from utility_functions import enable_compile_cache, get_compile_cache_enabled

from python_api_testing.models.llama.llama_utils import *
from python_api_testing.models.llama.llama_causallm import LlamaForCausalLM


def run_LlamaModel_real_inference():
    # PYTORCH HUGGINGFACE PREDICTION
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # generate input
    prompt = "I believe the meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = hugging_face_reference_model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"PyTorch response: {output}")
    # PyTorch response: I believe the meaning of life is to live it to the fullest and to enjoy every moment of it.
    # I believe that the meaning of

    # TT CALL
    num_decoders = 32
    tt_llama_model = LlamaForCausalLM(configuration, num_decoders, state_dict, device)

    tt_out = tt_llama_model(inputs.input_ids).to(host)
    tt_out1 = tt2torch_tensor(tt_out).squeeze(1)

    # decode output
    softmax_last = torch.nn.Softmax(dim=1)
    ids = softmax_last(tt_out1).argmax(2)

    output = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Tenstorrent response: {output}")


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_LlamaModel_real_inference()
    ttl.device.CloseDevice(device)
