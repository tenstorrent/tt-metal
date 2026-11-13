# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact

# Neither H nor W is a multiple of 32, and the top-left block covers part of each plane.
N, C, H, W = 2, 3, 37, 70
H_ONES, W_ONES = 5, 33


def _expected(val_hi, val_lo):
    expected = torch.full((N, C, H, W), val_lo, dtype=torch.float32).to(torch.bfloat16)
    expected[:, :, :H_ONES, :W_ONES] = torch.tensor(val_hi, dtype=torch.float32).to(torch.bfloat16)
    return expected


def _run_docstring_supported_inputs(op, values, dtype, layout, any_buffer_type, output_buffer_type, device):
    """Runs ``op`` with an ``any`` of the given dtype, layout and buffer type, with an explicit output buffer type and
    with the default memory configuration, and compares the output exactly."""
    check_readable(dtype, layout, any_buffer_type, output_buffer_type)
    any_tensor = ttnn.from_torch(
        torch.zeros((1, 1, 32, 32)),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=interleaved_memory_config(any_buffer_type),
    )
    val_hi, val_lo = values if values else (1.0, 0.0)
    for memory_config in (interleaved_memory_config(output_buffer_type), None):
        output = op(N, C, H, W, H_ONES, W_ONES, any_tensor, *values, memory_config=memory_config)

        assert output.dtype == ttnn.bfloat16
        assert output.layout == ttnn.ROW_MAJOR_LAYOUT
        assert output.memory_config() == (memory_config or any_tensor.memory_config())
        assert_exact(_expected(val_hi, val_lo), ttnn.to_torch(output))


@pytest.mark.parametrize("output_buffer_type", docstring_interleaved_buffer_type_params(ttnn.fill_rm, label="Output"))
@pytest.mark.parametrize("any_buffer_type", docstring_interleaved_buffer_type_params(ttnn.fill_rm, label="Input"))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.fill_rm))
def test_fill_rm_docstring_supported_inputs(dtype, layout, any_buffer_type, output_buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.fill_rm docstring lists for ``any`` and the
    output. 1/3 and -1e-3 are not BFLOAT16 values, so the output also checks their rounding."""
    _run_docstring_supported_inputs(
        ttnn.fill_rm, (1 / 3, -1e-3), dtype, layout, any_buffer_type, output_buffer_type, device
    )


@pytest.mark.parametrize(
    "output_buffer_type", docstring_interleaved_buffer_type_params(ttnn.fill_ones_rm, label="Output")
)
@pytest.mark.parametrize("any_buffer_type", docstring_interleaved_buffer_type_params(ttnn.fill_ones_rm, label="Input"))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.fill_ones_rm))
def test_fill_ones_rm_docstring_supported_inputs(dtype, layout, any_buffer_type, output_buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.fill_ones_rm docstring lists for ``any`` and the
    output."""
    _run_docstring_supported_inputs(ttnn.fill_ones_rm, (), dtype, layout, any_buffer_type, output_buffer_type, device)
