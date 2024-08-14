# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from transformers import BloomForCausalLM
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from loguru import logger

import models.experimental.bloom.bloom_utils as bloom_utils
import models.experimental.bloom.tt.bloom_attention as bloom_attention


def run_bloom_attention_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    block = 2
    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = f"transformer.h.{block}.self_attention"
    hidden_size = config.hidden_size

    tt_bloom_attention = bloom_attention.TtBloomAttention(config, state_dict, base_address, device)
    pt_bloom_attention = hugging_bloom_reference_model.transformer.h[block].self_attention

    # Prepare input
    torch.manual_seed(0)
    seq_len = 62

    hidden_states = ((torch.rand(1, seq_len, hidden_size) * 2) - 1) / hidden_size
    residual = ((torch.rand(1, seq_len, hidden_size) * 2) - 1) / hidden_size
    alibi = ((torch.rand(config.n_head, seq_len, seq_len) * 2) - 1) / seq_len
    attention_mask = torch.randint(0, 2, (1, 1, seq_len, seq_len)).type(torch.bool)
    pt_out = pt_bloom_attention.forward(hidden_states, residual, alibi, attention_mask)[0]

    tt_hidden_states = bloom_utils.torch2tt_tensor(hidden_states, device)
    tt_residual = bloom_utils.torch2tt_tensor(residual, device)
    tt_alibi = bloom_utils.torch2tt_tensor(alibi, device)

    tt_out = tt_bloom_attention.forward(device, tt_hidden_states, tt_residual, tt_alibi, attention_mask)[0]

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    pt_out_unsqueezed = pt_out.unsqueeze(0)

    does_pass, pcc_message = comp_pcc(pt_out_unsqueezed, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_attention: Passed!")
    else:
        logger.warning("bloom_attention: Failed!")

    assert does_pass


def test_bloom_attention(device):
    run_bloom_attention_test(device)
