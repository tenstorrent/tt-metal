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


@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
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
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device),
    )
    grad_tensor = ttnn.Tensor(grad_data, dtype).to(ttnn.TILE_LAYOUT).to(device)

    tt_dev = convert_to_torch_tensor(
        ttnn.angle_bw(grad_tensor, input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    )

    # PyTorch's native angle_backward evaluates self.abs().pow(2) in single precision,
    # which underflows to 0.0f and produces NaN for |z| < 2^-63 when FTZ (flush-to-zero)
    # is enabled on host CPUs. Evaluating golden autograd in float64 avoids this host underflow artifact.
    xr_64 = xr.to(torch.float64)
    xi_64 = xi.to(torch.float64)
    in_data_64 = torch.complex(xr_64, xi_64).requires_grad_(True)
    grad_data_64 = grad_data.to(torch.float64)
    golden_tensor = ttnn.get_golden_function(ttnn.angle_bw)(grad_data_64, in_data_64)
    golden_tensor = [golden_tensor[0].to(torch.float32)]

    assert not torch.isnan(tt_dev[0]).any()
    assert not torch.isinf(tt_dev[0]).any()
    passing, output = comp_allclose(golden_tensor[0], tt_dev[0], rtol=3e-2, atol=1e-18)
    logger.info(output)
    assert passing
