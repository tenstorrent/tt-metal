import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
import tt_lib
from tt_lib.fallback_ops import fallback_ops

from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from python_api_testing.models.llama.llama_utils import *
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.llama.llama_mlp import TtLlamaMLP


if __name__ == "__main__":
    torch.manual_seed(0)
    silu_input = (torch.rand(1, 1, 1, 32) * 2) - 1

    act_fn = fallback_ops.silu

    # pytorch ----------------------------------------------------------------
    pytorch_out1 = act_fn(silu_input)
    pytorch_out1 = tt2torch_tensor(pytorch_out1)
    print(f"PT original out: {pytorch_out1.shape}")

    # Pytorch scratch
    pytorch_out = 0 - silu_input
    pytorch_out = torch.exp(pytorch_out)
    pytorch_out = 1 + pytorch_out
    pytorch_out = 1 / pytorch_out
    pytorch_out = silu_input * pytorch_out
    print(f"PT out: {pytorch_out.shape}")

    # tt output --------------------------------------------------------------
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    silu_input = torch2tt_tensor(silu_input, device)

    term_sub = tt_lib.tensor.bcast(
        tt_const_tensor(0, silu_input.shape(), device),
        silu_input,
        tt_lib.tensor.BcastOpMath.SUB,
        tt_lib.tensor.BcastOpDim.H,
    )
    term_exp = tt_lib.tensor.exp(term_sub)
    term_add = tt_lib.tensor.bcast(
        tt_const_tensor(1, silu_input.shape(), device),
        term_exp,
        tt_lib.tensor.BcastOpMath.ADD,
        tt_lib.tensor.BcastOpDim.HW,
    )
    term_recip = tt_lib.tensor.recip(term_add)
    tt_out = tt_lib.tensor.mul(silu_input, term_recip)

    tt_out = tt2torch_tensor(tt_out)

    tt_lib.device.CloseDevice(device)
    print(f"TT out: {tt_out.shape}")
    tt_out = tt_out.squeeze(1)

    # check outputs -----------------------------------------------------------
    pcc = 0.98
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Llama Model Passed!")
    else:
        logger.warning("Llama Model Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"
