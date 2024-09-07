# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
import pytest
from loguru import logger

from transformers import T5Model, AutoModelForSeq2SeqLM
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
    is_wormhole_b0,
)
from models.experimental.t5.tt.t5_stack import TtT5Stack

pytestmark = pytest.mark.skipif(is_wormhole_b0(), reason="Skip for Wormhole B0")


def run_test_T5Stack_inference(device, model_name, input_h, input_w, pcc):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False
    config["use_cache"] = False

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder
        base_address = f"decoder"
    else:
        hf_reference_module = hf_reference_model.encoder
        base_address = f"encoder"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(2, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(inputs_embeds=test_input)
    pt_out = pt_out.last_hidden_state
    pt_out = pt_out.unsqueeze(0)

    # Move test input to Tt device test_input
    test_input = test_input.unsqueeze(0)
    test_input = torch2tt_tensor(test_input, device)

    tt_model = TtT5Stack(config, hf_reference_model.state_dict(), base_address, device)
    tt_model_outputs = tt_model(inputs_embeds=test_input)
    last_hidden_state = tt_model_outputs[0]
    tt_out = tt2torch_tensor(last_hidden_state)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5Stack_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Stack_inference {model_name} Failed!")

    assert does_pass


def test_T5Stack_inference_t5_small(device):
    run_test_T5Stack_inference(device, "t5-small", 64, 512, 0.99)


def test_T5Stack_inference_flan_t5_small(device):
    run_test_T5Stack_inference(device, "google/flan-t5-small", 64, 512, 0.94)


def test_T5Stack_inference_t5_base(device):
    run_test_T5Stack_inference(device, "t5-base", 64, 768, 0.99)
