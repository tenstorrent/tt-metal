from pathlib import Path
import sys

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
import python_api_testing.models.nanogpt.tt.nanogpt_attention as nanogpt_attention
from python_api_testing.models.nanogpt.tt.nanogpt_config import GPTConfig

from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_nanogpt_attn(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    # Prepare input

    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.attn"

    torch.manual_seed(0)

    test_in = torch.rand(1, 60, 768)
    pt_attn = model_hf.transformer.h[block].attn
    pt_out = pt_attn.forward(test_in)

    model_type = 'gpt2'

    # n_layer, n_head and n_embd are determined from model_type
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    }[model_type]

    config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    config_args['bias'] = True # always True for GPT model checkpoints
    # we can override the dropout rate, if desired

    config = GPTConfig(**config_args)

    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

    tt_attn = nanogpt_attention.TtCausalSelfAttention(config, sd, base_address, device)

    tt_out = tt_attn.forward(
        tt_test_in
    )

    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("nanogpt_attention: Passed!")
    else:
        logger.warning("nanogpt_attention: Failed!")

    assert does_pass
