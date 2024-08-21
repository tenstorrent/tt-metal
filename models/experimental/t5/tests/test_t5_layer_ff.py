# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
from loguru import logger

from transformers import T5Model
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from models.experimental.t5.tt.t5_layer_ff import TtT5LayerFF


def run_test_T5LayerFF_inference(device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[2]
        base_address = f"decoder.block.0.layer.2"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1]
        base_address = f"encoder.block.0.layer.1"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5LayerFF(config, hf_reference_model.state_dict(), base_address, device)
    tt_out = tt_model(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5LayerFF_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5LayerFF_inference {model_name} Failed!")

    assert does_pass


def test_T5LayerFF_inference_t5_small(device):
    run_test_T5LayerFF_inference(device, "t5-small", 64, 512)


def test_T5LayerFF_inference_flan_t5_small(device):
    run_test_T5LayerFF_inference(device, "google/flan-t5-small", 64, 512)


def test_T5LayerFF_inference_t5_base(device):
    run_test_T5LayerFF_inference(device, "t5-base", 64, 768)
