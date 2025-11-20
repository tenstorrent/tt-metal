# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, assert_with_ulp

############# Unary (expect sfpu util > 0, fpu util = 0) ##############################

@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 320, 320])),
        (torch.Size([1, 1, 3200, 3200])),
    ),
)
def test_unary_square_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.square(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.square(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass

@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([1, 1, 3200, 3200])),
    ),
)
def test_unary_tanh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.tanh(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.tanh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 3200, 3200],
    ],
)
def test_add(device, shape):
    x_torch = torch.ones(shape, dtype=torch.bfloat16)
    y_torch = torch.ones(shape, dtype=torch.bfloat16)
    z_torch = x_torch + y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
    )
    tt_out = ttnn.to_torch(z_tt_add)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status

@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 320, 384],
    ],
)
def test_add_input_activ(device, shape):
    x_torch = torch.ones(shape, dtype=torch.bfloat16)
    y_torch = torch.ones(shape, dtype=torch.bfloat16)
    z_torch = torch.square(torch.nn.functional.silu(x_torch) + y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(
        x_tt,
        y_tt,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        activations=[ttnn.UnaryOpType.SQUARE],
        use_legacy=None,
    )
    tt_out = ttnn.to_torch(z_tt_add)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.9999
    assert status
