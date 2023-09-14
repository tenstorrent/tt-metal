# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
from transformers import BloomForCausalLM
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

from loguru import logger
from models.bloom.tt.bloom_mlp import TtBloomMLP


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bloom_mlp(pcc, reset_seeds, device):
    # Prepare input
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m"
    )
    hugging_bloom_reference_model.eval()

    block = 2
    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = f"transformer.h.{block}.mlp"
    hidden_size = config.hidden_size

    # Prepare Input
    test_in = torch.rand(1, 1, 61, hidden_size)
    res = torch.rand(1, 1, 61, hidden_size)

    tt_mlp = TtBloomMLP(config, state_dict, base_address, device)

    pt_mlp = hugging_bloom_reference_model.transformer.h[block].mlp

    with torch.no_grad():
        # tt output
        tt_out = tt_mlp(
            torch_to_tt_tensor_rm(test_in, device),
            torch_to_tt_tensor_rm(res, device),
        )
        # pytorch output
        pt_out = pt_mlp(test_in, res)

    tt_out_converted = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_mlp: Passed!")
    else:
        logger.warning("bloom_mlp: Failed!")

    assert does_pass, f"bloom_mlp output does not meet PCC requirement {pcc}."
