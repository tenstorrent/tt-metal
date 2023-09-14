# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib

import pytest
from transformers import BloomForCausalLM
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from loguru import logger
from models.bloom.tt.bloom_block import TtBloomBlock


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bloom_block(pcc, reset_seeds, device):
    tt_lib.device.SetDefaultDevice(device)

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m"
    )
    hugging_bloom_reference_model.eval()

    block = 0
    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = f"transformer.h.{block}"
    hidden_size = config.hidden_size
    n_head = config.n_head

    tt_bloom_block = TtBloomBlock(config, state_dict, base_address, device)
    pt_bloom_block = hugging_bloom_reference_model.transformer.h[block]

    seq_len = 62

    # Prepare Input
    hidden_states = ((torch.rand(1, seq_len, hidden_size) * 2) - 1) / hidden_size
    alibi = ((torch.rand(n_head, seq_len, seq_len) * 2) - 1) / (seq_len * seq_len)
    attention_mask = torch.randint(0, 2, (1, 1, seq_len, seq_len))

    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device, put_on_device=False)
    tt_alibi = torch_to_tt_tensor_rm(alibi, device, put_on_device=False)
    tt_attention_mask = torch_to_tt_tensor_rm(
        attention_mask, device, put_on_device=False
    )

    with torch.no_grad():
        # Pytorch Output
        pt_out = pt_bloom_block(hidden_states, alibi, attention_mask)[0]
        # tt output
        tt_out = tt_bloom_block(tt_hidden_states, tt_alibi, tt_attention_mask)[0]

    tt_out_converted = tt_to_torch_tensor(tt_out)
    tt_out_converted = tt_out_converted.squeeze()

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_block: Passed!")
    else:
        logger.warning("bloom_block: Failed!")

    assert does_pass, f"bloom_block output does not meet PCC requirement {pcc}."
