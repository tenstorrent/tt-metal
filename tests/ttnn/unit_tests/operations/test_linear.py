# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


@pytest.mark.parametrize("batch_sizes", [(1,)])
@pytest.mark.parametrize("m_size", [384])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("use_bias", [True, False])
def test_linear(
    batch_sizes,
    m_size,
    k_size,
    n_size,
    use_bias,
    *,
    device,
):
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [384])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("core_grid", [False])
def test_linear_with_core_grid(
    batch_size,
    m_size,
    k_size,
    n_size,
    use_bias,
    core_grid,
    *,
    device,
):
    input_shape_a = (batch_size, 1, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    if batch_size == 1:
        with pytest.raises(RuntimeError) as exception:
            output_tensor = ttnn.linear(
                input_tensor_a,
                input_tensor_b,
                bias=bias,
                core_grid=ttnn.CoreGrid(y=batch_size, x=6),
            )
        assert "ttnn.linear: ttl.operations.primary.matmul failed" in str(exception.value)
    else:
        output_tensor = ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=bias,
            core_grid=ttnn.CoreGrid(y=batch_size, x=6),
        )

        output_tensor = ttnn.to_torch(output_tensor)

        assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
