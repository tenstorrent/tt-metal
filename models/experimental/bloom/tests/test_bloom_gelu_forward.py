# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from loguru import logger
import models.experimental.bloom.bloom_utils as bloom_utils
import models.experimental.bloom.tt.bloom_gelu_forward as bloom_gelu_forward


def run_bloom_gelu_forward_test(device):
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 61, 1024) / 1024

    pt_out = bloom_gelu_forward.bloom_gelu_forward(test_in)
    tt_test_in = bloom_utils.torch2tt_tensor(test_in, device)
    tt_out = bloom_gelu_forward.tt_bloom_gelu_forward(tt_test_in, device)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_gelu_forward: Passed!")
    else:
        logger.warning("bloom_gelu_forward: Failed!")

    assert does_pass


def test_bloom_gelu_forward(device):
    run_bloom_gelu_forward_test(device)
