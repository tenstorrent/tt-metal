# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
import ttnn
from loguru import logger
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_equal, comp_allclose

from models.utility_functions import is_wormhole_b0, skip_for_grayskull
from tests.ttnn.unit_tests.operations.eltwise.complex.utility_funcs import (
    convert_complex_to_torch_tensor,
    random_complex_tensor,
)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "memcfg",
    (
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ),
    ids=["out_DRAM", "out_L1"],
)
@pytest.mark.parametrize("dtype", ((ttnn.float32,)))
@pytest.mark.parametrize("bs", ((1, 1),))
@pytest.mark.parametrize("hw", ((32, 32),))
def test_conj(bs, hw, memcfg, dtype, device, function_level_defaults):
    input_shape = torch.Size([bs[0], bs[1], hw[0], hw[1]])

    in_data = random_complex_tensor(input_shape, (-90, 90), (-70, 70))

    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, memcfg),
    )

    tt_dev = ttnn.conj(input_tensor, memory_config=memcfg)

    tt_to_torch = convert_complex_to_torch_tensor(tt_dev)

    golden_function = ttnn.get_golden_function(ttnn.conj)
    golden_tensor = golden_function(in_data)

    passing, output = comp_pcc(golden_tensor, tt_to_torch)
    logger.info(output)
    assert passing
