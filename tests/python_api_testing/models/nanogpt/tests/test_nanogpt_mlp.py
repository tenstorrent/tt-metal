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
import python_api_testing.models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

def run_nanogpt_mlp_test(device, pcc):

    model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
    sd = model_hf.state_dict()
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.mlp"

    torch.manual_seed(0)

    test_in = torch.rand(1, 43, 768)

    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

    tt_mlp = nanogpt_mlp.TtMLP(base_address, sd, device)

    tt_out = tt_mlp.forward(
        tt_test_in
    )


    pt_mlp = model_hf.transformer.h[block].mlp
    pt_out = pt_mlp.forward(test_in)

    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_mlp: Passed!")
    else:
        logger.warning("nanogpt_mlp: Failed!")

    assert does_pass

@pytest.mark.parametrize(
    "pcc",
    (
        (
            0.99,
        ),
    ),
)
def test_nanogpt_mlp(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_nanogpt_mlp_test(device, pcc)
    tt_lib.device.CloseDevice(device)
