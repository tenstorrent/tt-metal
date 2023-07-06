from pathlib import Path
import sys


import os
import pickle
import tiktoken

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
import pytest

from transformers import GPT2LMHeadModel

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.nanogpt.tt.nanogpt_block as nanogpt_block
import python_api_testing.models.nanogpt.tt.nanogpt_attention as nanogpt_attention
import python_api_testing.models.nanogpt.tt.nanogpt_model as nanogpt_model
from python_api_testing.models.nanogpt.tt.nanogpt_config import GPTConfig



from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

# -----------------------------------------------------------------------------
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
max_new_tokens = 20 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device_select = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------


def run_nanogpt_model_test(device, prompt, temperature, max_new_tokens):
    # Prepare input

    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    sd = model_hf.state_dict()
    model_hf.eval()
    torch.manual_seed(0)


    model_type = 'gpt2'

    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    }[model_type]

    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    config_args['bias'] = True # always True for GPT model checkpoints

    config = GPTConfig(**config_args)

    tt_model = nanogpt_model.TtGPT(config, sd, device)

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    text = prompt
    start_ids = encode(text)

    x = (torch.tensor(start_ids, dtype=torch.long, device='cpu')[None, ...])
    y = tt_model.generate(x, max_new_tokens, temperature, top_k=top_k)
    logger.info(decode(y[0].tolist()))

@pytest.mark.parametrize(
    "prompt, max_new_tokens, temperature",
    (
        (
            "Where do you go?",
            25,
            0.8,
        ),
    ),
)
def test_nanogpt_model(prompt, max_new_tokens, temperature):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_nanogpt_model_test(device, prompt, temperature, max_new_tokens)
    tt_lib.device.CloseDevice(device)
