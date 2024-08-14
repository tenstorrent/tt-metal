# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import torch
from loguru import logger

from transformers import T5Model, AutoModelForSeq2SeqLM
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
)
from models.experimental.t5.tt.t5_dense_gated_act_dense import TtT5DenseGatedActDense


def run_test_T5DenseGatedActDense_inference(device):
    hugging_face_reference_model = T5Model.from_pretrained("google/flan-t5-small")
    # hugging_face_reference_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    hugging_face_reference_model.eval()

    config = json.loads(hugging_face_reference_model.config.to_json_string())
    config["is_decoder"] = False

    if config["is_decoder"]:
        hf_reference_module = hugging_face_reference_model.decoder.block[0].layer[2].DenseReluDense
        base_address = f"decoder.block.0.layer.2.DenseReluDense"
    else:
        hf_reference_module = hugging_face_reference_model.encoder.block[0].layer[1].DenseReluDense
        base_address = f"encoder.block.0.layer.1.DenseReluDense"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5DenseGatedActDense(config, hugging_face_reference_model.state_dict(), base_address, device)
    tt_out = tt_model(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_T5DenseGatedActDense_inference Passed!")
    else:
        logger.warning("test_T5DenseGatedActDense_inference Failed!")

    assert does_pass


def test_T5DenseGatedActDense_inference(device):
    run_test_T5DenseGatedActDense_inference(device)
