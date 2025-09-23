# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import json
from loguru import logger

from transformers import T5Model
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
    is_e75,
    is_wormhole_b0,
)
from models.experimental.t5.tt.t5_block import TtT5Block

pytestmark = pytest.mark.skipif(is_wormhole_b0(), reason="Skip for Wormhole B0")


def run_test_T5Block_inference(device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    block = 1
    has_relative_attention_bias = block == 0

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[block]
        base_address = f"decoder.block.{block}"
    else:
        hf_reference_module = hf_reference_model.encoder.block[block]
        base_address = f"encoder.block.{block}"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(32, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(0)
    test_input = test_input.unsqueeze(0)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5Block(
        config,
        hf_reference_model.state_dict(),
        base_address,
        device,
        has_relative_attention_bias,
    )
    tt_out = tt_model(torch2tt_tensor(test_input, device))[0]
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5Block_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Block_inference {model_name} Failed!")

    assert does_pass


def test_T5Block_inference_t5_small(device):
    run_test_T5Block_inference(device, "t5-small", 64, 512)


def test_T5Block_inference_flan_t5_small(device):
    run_test_T5Block_inference(device, "google/flan-t5-small", 64, 512)


def test_T5Block_inference_t5_base(device):
    if is_e75(device):
        pytest.skip("T5 Block T5 base config is not supported on E75")
    run_test_T5Block_inference(device, "t5-base", 64, 768)
