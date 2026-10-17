# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose

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
def test_level2_angle_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))
    in_data.requires_grad = True

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    grad_data, grad_tensor = data_gen_with_range(input_shape, -50, 40, device)

    tt_dev = ttnn.angle_bw(grad_tensor, input_tensor, memory_config=memcfg)
    tt_dev = convert_to_torch_tensor(tt_dev)

    golden_function = ttnn.get_golden_function(ttnn.angle_bw)
    golden_tensor = golden_function(grad_data, in_data)

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        else:
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        logger.info(output)
        assert passing


@pytest.mark.parametrize("dtype", (ttnn.float32, ttnn.bfloat16))
@pytest.mark.parametrize(
    "z_pair",
    (
        (1e20, 0.0),
        (0.0, 1e20),
        (3e19, 4e19),
        (1e-20, 0.0),
        (0.0, 1e-20),
        (3e-20, 4e-20),
        (0.0, 0.0),
    ),
)
def test_angle_bw_extreme_exponent_range(dtype, z_pair, device):
    shape = torch.Size([1, 1, 32, 32])
    xr = torch.full(shape, z_pair[0], dtype=torch.float32)
    xi = torch.full(shape, z_pair[1], dtype=torch.float32)
    in_data = torch.complex(xr, xi).requires_grad_(True)
    grad_data = torch.ones(shape, dtype=torch.float32)

    input_tensor = ttnn.complex_tensor(
        ttnn.from_torch(in_data.real, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(in_data.imag, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
    )
    grad_tensor = ttnn.from_torch(grad_data, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tt_dev = convert_to_torch_tensor(ttnn.angle_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    golden_tensor = ttnn.get_golden_function(ttnn.angle_bw)(grad_data, in_data)

    assert not torch.isnan(tt_dev[0]).any()
    assert not torch.isinf(tt_dev[0]).any()
    passing_r, out_r = comp_allclose(golden_tensor[0].real, tt_dev[0].real, rtol=2e-2, atol=1e-25)
    passing_i, out_i = comp_allclose(golden_tensor[0].imag, tt_dev[0].imag, rtol=2e-2, atol=1e-25)
    logger.info(f"{out_r} | {out_i}")
    assert passing_r and passing_i
