# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "beta",
    [0.5, -3, 1, 4, 0],
)
@pytest.mark.parametrize(
    "threshold",
    [-20, -10, 10, 20, 5, 0],
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_bw_softplus(input_shapes, beta, threshold, input_dtype, layout, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    input_tensor = ttnn.from_torch(
        in_data, dtype=input_dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=layout
    )
    grad_tensor = ttnn.from_torch(
        grad_data, dtype=input_dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=layout
    )

    tt_output_tensor_on_device = ttnn.softplus_bw(grad_tensor, input_tensor, beta=beta, threshold=threshold)

    golden_function = ttnn.get_golden_function(ttnn.softplus_bw)
    golden_tensor = golden_function(grad_data, in_data, beta, threshold)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_bw_default_softplus(input_shapes, input_dtype, layout, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    input_tensor = ttnn.from_torch(
        in_data, dtype=input_dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=layout
    )
    grad_tensor = ttnn.from_torch(
        grad_data, dtype=input_dtype, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=layout
    )

    tt_output_tensor_on_device = ttnn.softplus_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.softplus_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
