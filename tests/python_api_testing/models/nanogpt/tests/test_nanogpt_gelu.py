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
import python_api_testing.models.nanogpt.tt.nanogpt_gelu as nanogpt_gelu

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
def test_nanogpt_gelu_inference(pcc):
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    torch.manual_seed(0)
    test_in = torch.rand(1, 43, 768)

    pt_out = nanogpt_gelu.new_gelu(test_in)

    tt_test_in = torch2tt_tensor(test_in, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
    tt_out = nanogpt_gelu.tt_nanogpt_gelu(tt_test_in, device)
    tt_out_converted = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("nanogpt_gelu: Passed!")
    else:
        logger.warning("nanogpt_gelu: Failed!")

    assert does_pass
