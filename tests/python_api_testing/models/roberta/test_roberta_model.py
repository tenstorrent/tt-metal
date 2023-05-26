import math
from pathlib import Path
import sys
import random
from typing import Optional, Tuple, Union, List
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
from python_api_testing.models.roberta.roberta_model import TtRobertaModel
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from transformers import RobertaModel
from transformers import AutoTokenizer


def test_roberta_model_inference():
    torch.manual_seed(1234)
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    base_address = f""
    torch_model = RobertaModel.from_pretrained("roberta-base")

    torch_model.eval()
    with torch.no_grad():
        # Tt roberta
        tt_model = TtRobertaModel(
            config=torch_model.config,
            base_address=base_address,
            device=device,
            state_dict=torch_model.state_dict(),
            reference_model=torch_model,
            add_pooling_layer=False,
        )

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

        # Run torch model
        torch_output = torch_model(**inputs)
        torch_output = torch_output.last_hidden_state

        # Run tt model
        tt_attention_mask = torch.unsqueeze(inputs.attention_mask.float(), 0)
        tt_attention_mask = torch.unsqueeze(tt_attention_mask, 0)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, device)

        tt_output = tt_model(inputs.input_ids, tt_attention_mask)
        tt_output = tt_output.last_hidden_state

        # Compare outputs
        tt_output = tt2torch_tensor(tt_output)
        while len(torch_output.size()) < len(tt_output.size()):
            tt_output = tt_output.squeeze(0)

        does_pass, pcc_message = comp_pcc(torch_output, tt_output, 0.98)

        logger.info(comp_allclose(torch_output, tt_output))
        logger.info(pcc_message)

        tt_lib.device.CloseDevice(device)

        if does_pass:
            logger.info("RobertaModel Passed!")
        else:
            logger.warning("RobertaModel Failed!")

        assert does_pass


if __name__ == "__main__":
    test_roberta_model_inference()
