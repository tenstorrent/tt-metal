# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
import sys

import torch

import pytest
import ttnn
from loguru import logger
from math import pi

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose

from models.utility_functions import is_wormhole_b0
from tests.ttnn.unit_tests.operations.eltwise.backward.complex_ops.backward_complex_utility_funcs import (
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
def test_level2_polar_bw(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    # polar_bw use polar fwd op, which uses sin and cos ops with range (0, 2*pi).
    in_data = random_complex_tensor(input_shape, (-90, 90), (0, 2 * pi))
    in_data.requires_grad = True

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttnn.complex_tensor(
        ttnn.Tensor(grad_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(grad_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )
    tt_dev = ttnn.polar_bw(grad_tensor, input_tensor, memory_config=memcfg)
    tt_dev = convert_to_torch_tensor(tt_dev)

    golden_function = ttnn.get_golden_function(ttnn.polar_bw)
    golden_tensor = golden_function(grad_data, in_data)

    for i in range(len(tt_dev)):
        if is_wormhole_b0():
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        else:
            passing, output = comp_pcc(golden_tensor[i], tt_dev[i])
        logger.info(output)
        assert passing
