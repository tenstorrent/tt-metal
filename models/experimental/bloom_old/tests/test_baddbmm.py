# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from loguru import logger
import models.experimental.bloom_old.bloom_utils as bloom_utils
import models.experimental.bloom_old.tt.baddbmm as baddbmm


def run_baddbmm_test(device):
    torch.manual_seed(0)

    input = torch.rand(32, 64, 64)
    batch1 = torch.rand(32, 64, 32)
    batch2 = torch.rand(32, 32, 64)

    alpha = 0.25
    beta = 0.5
    tt_alpha = bloom_utils.tt_const_tensor(alpha, [1, 32, 64, 64], device)
    tt_beta = bloom_utils.tt_const_tensor(beta, [1, 32, 64, 64], device)

    pt_out = torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)
    pt_out_size = list(pt_out.shape)

    while len(pt_out_size) < 4:
        pt_out_size.insert(0, 1)

    pt_out = torch.reshape(pt_out, pt_out_size)

    input = bloom_utils.torch2tt_tensor(input, device)
    batch1 = bloom_utils.torch2tt_tensor(batch1, device)
    batch2 = bloom_utils.torch2tt_tensor(batch2, device)

    tt_out = baddbmm.tt_baddbmm(device, input, batch1, batch2, beta=tt_beta, alpha=tt_alpha)
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("baddbmm: Passed!")
    else:
        logger.warning("baddbmm: Failed!")

    assert does_pass


def test_baddbmm(device):
    run_baddbmm_test(device)
