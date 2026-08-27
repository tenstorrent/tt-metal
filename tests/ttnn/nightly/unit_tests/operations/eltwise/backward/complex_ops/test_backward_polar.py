# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

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

from models.common.utility_functions import is_wormhole_b0
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


@pytest.mark.parametrize("dtype", ((ttnn.bfloat16,)))
@pytest.mark.parametrize(
    "input_memcfg, requested_memcfg, forbidden_buffer_type",
    (
        (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, "L1"),
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, "DRAM"),
    ),
    ids=["in_L1_out_DRAM", "in_DRAM_out_L1"],
)
def test_polar_bw_intermediate_placement(
    input_memcfg, requested_memcfg, forbidden_buffer_type, dtype, device, function_level_defaults
):
    """polar_bw must allocate its intermediates in the requested memory config.

    Regression for the `ones_like` that built the imaginary half of `flip_tensor`
    without `output_mem_config`: it inherited the input's config instead, so a caller
    asking for DRAM got an unrequested L1 allocation. The existing tests pass the same
    memcfg for inputs and output, so requested and inherited never diverge there.
    """
    input_shape = torch.Size([1, 1, 32, 64])

    in_data = random_complex_tensor(input_shape, (-90, 90), (0, 2 * pi))
    input_tensor = ttnn.complex_tensor(
        ttnn.Tensor(in_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, input_memcfg),
        ttnn.Tensor(in_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, input_memcfg),
    )

    grad_data = random_complex_tensor(input_shape, (-50, 50), (-60, 60))
    grad_tensor = ttnn.complex_tensor(
        ttnn.Tensor(grad_data.real, dtype).to(ttnn.TILE_LAYOUT).to(device, input_memcfg),
        ttnn.Tensor(grad_data.imag, dtype).to(ttnn.TILE_LAYOUT).to(device, input_memcfg),
    )

    # Inputs are built before capture, so only polar_bw's own allocations are recorded.
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    ttnn.polar_bw(grad_tensor, input_tensor, memory_config=requested_memcfg)
    captured_graph = ttnn.graph.end_graph_capture()

    allocations = [node["params"] for node in captured_graph if node.get("node_type") == "buffer_allocate"]
    assert allocations, "expected polar_bw to allocate at least one buffer"

    stray = [params for params in allocations if params.get("type") == forbidden_buffer_type]
    logger.info(f"polar_bw allocated {len(allocations)} buffers, {len(stray)} in {forbidden_buffer_type}")
    assert not stray, (
        f"polar_bw allocated {len(stray)} buffer(s) in {forbidden_buffer_type} "
        f"despite memory_config requesting the other buffer type"
    )
