# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
    comp_equal,
    comp_allclose,
)

from tt_py_test_utils_common.utility_functions import is_wormhole_b0
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.complex_ops.backward_complex_utility_funcs import (
    Complex,
    convert_to_torch_tensor,
    random_complex_tensor,
)


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2), (2, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_abs_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    grad_data, grad_tensor = data_gen_with_range(input_shape, -50, 40, device)

    tt_dev = ttnn.abs_bw(grad_tensor, input_tensor, memory_config=memcfg)
    tt_dev = convert_to_torch_tensor(tt_dev)

    golden_function = ttnn.get_golden_function(ttnn.abs_bw)
    golden_tensor = golden_function(grad_data, in_data)

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        else:
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize("bs", ((1, 1), (1, 2)))
@pytest.mark.parametrize("hw", ((32, 64), (320, 384)))
def test_level2_abs_bw_inp_zero(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (0, 0), (0, 0))
    in_data.requires_grad = True

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    grad_data, grad_tensor = data_gen_with_range(input_shape, -50, 80, device)

    tt_dev = ttnn.abs_bw(grad_tensor, input_tensor, memory_config=memcfg)
    tt_dev = convert_to_torch_tensor(tt_dev)

    golden_function = ttnn.get_golden_function(ttnn.abs_bw)
    golden_tensor = golden_function(grad_data, in_data)

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        else:
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize(
    "scale_exp", [80, -80], ids=["large_exp_2p80", "small_exp_2m80"]
)
def test_level2_abs_bw_extreme_exponent_range(
    scale_exp, device, function_level_defaults
):
    input_shape = torch.Size([1, 1, 32, 64])
    scale = math.ldexp(1.0, scale_exp)
    torch.manual_seed(0)
    real_part = (
        torch.empty(input_shape, dtype=torch.float32).uniform_(0.5, 2.0) * scale
    ).to(torch.bfloat16)
    imag_part = (
        torch.empty(input_shape, dtype=torch.float32).uniform_(-2.0, -0.5) * scale
    ).to(torch.bfloat16)
    in_data = torch.complex(real_part.float(), imag_part.float())
    in_data.requires_grad = True

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(real_part, ttnn.bfloat16)
        .to(ttnn.TILE_LAYOUT)
        .to(device, ttnn.DRAM_MEMORY_CONFIG),
        ttnn.Tensor(imag_part, ttnn.bfloat16)
        .to(ttnn.TILE_LAYOUT)
        .to(device, ttnn.DRAM_MEMORY_CONFIG),
    )
    grad_data, grad_tensor = data_gen_with_range(input_shape, -10, 10, device)

    tt_dev = ttnn.abs_bw(
        grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_dev = convert_to_torch_tensor(tt_dev)

    in_f64 = torch.complex(real_part.double(), imag_part.double()).requires_grad_(True)
    in_f64.abs().backward(grad_data.double())
    golden = in_f64.grad.to(torch.complex64)

    assert not torch.isnan(tt_dev[0]).any()
    assert not torch.isinf(tt_dev[0]).any()
    tt_dev_real = tt_dev[0][..., : input_shape[-1]]
    tt_dev_imag = tt_dev[0][..., input_shape[-1] :]
    passing_r, out_r = comp_allclose(
        golden.real.float(), tt_dev_real.float(), rtol=2e-2, atol=1e-2
    )
    passing_i, out_i = comp_allclose(
        golden.imag.float(), tt_dev_imag.float(), rtol=2e-2, atol=1e-2
    )
    logger.info(f"{out_r} | {out_i}")
    assert passing_r and passing_i
