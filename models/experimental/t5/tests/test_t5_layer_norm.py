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
from models.experimental.t5.tt.t5_layer_norm import TtT5LayerNorm


def run_test_T5LayerNorm_inference(device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    # Module to test
    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[1].layer_norm
        base_address = f"decoder.block.0.layer.1.layer_norm"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1].layer_norm
        base_address = f"encoder.block.0.layer.1.layer_norm"

    # Prepare input
    torch.manual_seed(0)
    t5_layer_norm_input = (torch.rand(1, 1, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(t5_layer_norm_input)[0].unsqueeze(1)
    tt_T5LayerNorm_model = TtT5LayerNorm(config, hf_reference_model.state_dict(), base_address, device)

    # TT hardware execution
    tt_layer_norm_input = torch2tt_tensor(t5_layer_norm_input, device)

    tt_out = tt_T5LayerNorm_model(tt_layer_norm_input)
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5LayerNorm_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5LayerNorm_inference {model_name} Failed!")

    assert does_pass


def test_T5LayerNorm_inference_t5_small(device):
    run_test_T5LayerNorm_inference(device, "t5-small", 64, 512)


def test_T5LayerNorm_inference_flan_t5_small(device):
    run_test_T5LayerNorm_inference(device, "google/flan-t5-small", 64, 512)


def test_T5LayerNorm_inference_t5_base(device):
    run_test_T5LayerNorm_inference(device, "t5-base", 64, 768)
