# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from models.common.utility_functions import pad_by_zero
from tests.ttnn.utils_for_testing import assert_numeric_metrics

shapes = [[1, 1, 32, 32], [1, 1, 32, 128], [1, 2, 128, 128]]


@pytest.mark.parametrize("shape", shapes)
def test_softmax(shape, device):
    torch.manual_seed(1234)
    x = torch.randn(shape).bfloat16().float()
    xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    xtt = ttnn.softmax_in_place(xt)

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_out = torch.nn.functional.softmax(x, dim=-1)

    assert_numeric_metrics(
        pt_out,
        tt_got_back,
        pcc_threshold=0.999,
        rtol=0.105,
        atol=0.017,
        frobenius_threshold=0.030,
    )


@pytest.mark.parametrize("shape", shapes)
def test_layernorm(shape, device):
    torch.manual_seed(1234)
    x = torch.randn(shape).bfloat16().float()
    gamma = torch.randn([shape[-1]]).bfloat16().float()
    beta = torch.randn([shape[-1]]).bfloat16().float()

    xt = ttnn.Tensor(x, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    gammat = pad_by_zero(gamma, device)[0]
    betat = pad_by_zero(beta, device)[0]

    xtt = ttnn.layer_norm(xt, epsilon=1e-5, weight=gammat, bias=betat)

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_out = torch.nn.functional.layer_norm(x, x.shape[-1:], gamma, beta, 1e-5)

    assert_numeric_metrics(
        pt_out,
        tt_got_back,
        pcc_threshold=0.999,
        rtol=1.100,
        atol=0.047,
        frobenius_threshold=0.003,
    )
