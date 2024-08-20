# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from transformers import BloomForCausalLM
from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from loguru import logger
import models.experimental.bloom_old.bloom_utils as bloom_utils
import models.experimental.bloom_old.tt.bloom_mlp as bloom_mlp


def run_bloom_mlp_test(device):
    # Prepare input
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    block = 6
    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = f"transformer.h.{block}.mlp"
    hidden_size = config.hidden_size

    torch.manual_seed(0)

    test_in = torch.rand(1, 1, 64, hidden_size)
    res = torch.rand(1, 1, 64, hidden_size)

    tt_mlp = bloom_mlp.TtBloomMLP(config, state_dict, base_address, device)
    tt_out = tt_mlp.forward(
        bloom_utils.torch2tt_tensor(test_in, device),
        bloom_utils.torch2tt_tensor(res, device),
        device,
    )

    pt_mlp = hugging_bloom_reference_model.transformer.h[block].mlp
    pt_out = pt_mlp.forward(test_in, res)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)

    print(pcc_message)

    if does_pass:
        logger.info("bloom_mlp: Passed!")
    else:
        logger.warning("bloom_mlp: Failed!")

    assert does_pass


def test_bloom_mlp(device):
    run_bloom_mlp_test(device)
