import math
from pathlib import Path
import sys
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.roberta.roberta_lm_head import TtRobertaLMHead
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from transformers import RobertaForMaskedLM
from transformers import AutoTokenizer
from dataclasses import dataclass


def test_roberta_lm_head():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    base_address = f"lm_head"

    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    torch_model = model.lm_head

    # Tt roberta
    tt_model = TtRobertaLMHead(
        config=model.config,
        base_address=base_address,
        device=device,
        state_dict=model.state_dict(),
    )

    input_x = torch.rand([1, 9, 768])

    # Run torch model
    torch_output = torch_model(input_x)

    # Run tt model
    input_x = torch.unsqueeze(input_x, 0)
    input_x = torch2tt_tensor(input_x, device)

    tt_output = tt_model(input_x)

    # Compare outputs
    tt_output_torch = tt2torch_tensor(tt_output)
    tt_output_torch = tt_output_torch.squeeze(0)

    does_pass, pcc_message = comp_pcc(torch_output, tt_output_torch, 0.98)

    logger.info(comp_allclose(torch_output, tt_output_torch))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("RobertaLMHead Passed!")
    else:
        logger.warning("RobertaLMHead Failed!")

    assert does_pass


if __name__ == "__main__":
    test_roberta_lm_head()
