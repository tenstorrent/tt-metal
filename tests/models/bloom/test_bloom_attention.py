# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest

from transformers import BloomForCausalLM
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from loguru import logger
from models.bloom.tt.bloom_attention import TtBloomAttention


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bloom_attention(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m"
    )
    hugging_bloom_reference_model.eval()

    block = 2
    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = f"transformer.h.{block}.self_attention"
    hidden_size = config.hidden_size

    tt_bloom_attention = TtBloomAttention(config, state_dict, base_address, device)
    pt_bloom_attention = hugging_bloom_reference_model.transformer.h[
        block
    ].self_attention

    # Prepare input
    seq_len = 62

    hidden_states = ((torch.rand(1, seq_len, hidden_size) * 2) - 1) / hidden_size
    residual = ((torch.rand(1, seq_len, hidden_size) * 2) - 1) / hidden_size
    alibi = ((torch.rand(config.n_head, seq_len, seq_len) * 2) - 1) / seq_len
    attention_mask = torch.randint(0, 2, (1, 1, seq_len, seq_len))

    tt_hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
    tt_residual = torch_to_tt_tensor_rm(residual, device)
    tt_alibi = torch_to_tt_tensor_rm(alibi, device)
    tt_attention_mask = torch_to_tt_tensor_rm(attention_mask, device)

    with torch.no_grad():
        # pytorch output
        pt_out = pt_bloom_attention(hidden_states, residual, alibi, attention_mask)[0]

        # tt output
        tt_out = tt_bloom_attention(
            tt_hidden_states, tt_residual, tt_alibi, tt_attention_mask
        )[0]

    tt_out_converted = tt_to_torch_tensor(tt_out)
    pt_out_unsqueezed = pt_out.unsqueeze(0)

    does_pass, pcc_message = comp_pcc(pt_out_unsqueezed, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out_unsqueezed, tt_out_converted))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("bloom_attention: Passed!")
    else:
        logger.warning("bloom_attention: Failed!")

    assert does_pass, f"bloom_attention output does not meet PCC requirement {pcc}."
