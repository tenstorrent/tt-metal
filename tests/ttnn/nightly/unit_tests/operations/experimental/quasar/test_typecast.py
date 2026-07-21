# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Public API and short-row regression coverage for ``ttnn.experimental.quasar.typecast``."""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.test_utils import get_ttnn_torch_dtype


@pytest.mark.parametrize(
    "input_dtype",
    (
        ttnn.bfloat16,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        ttnn.int32,
        ttnn.bfloat16,
        ttnn.float32,
    ),
    ids=[
        "INT32",
        "BFLOAT16",
        "FLOAT32",
    ],
)
def test_typecast_row_major_short_row_is_deterministic(input_dtype, output_dtype, device):
    """A short row must not let an LLK tile overlap the next double-buffered CB page."""
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")
    torch.manual_seed(54321)
    torch_input = torch.rand((1, 1, 361, 512), dtype=get_ttnn_torch_dtype(input_dtype))
    expected = torch_input.to(get_ttnn_torch_dtype(output_dtype))
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=input_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    first_output = None
    for run_idx in range(5):
        output = ttnn.experimental.quasar.typecast(
            input_tensor,
            output_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        torch_output = ttnn.to_torch(output)

        assert torch.equal(torch_output, expected), f"Incorrect output on run {run_idx}"
        if first_output is None:
            first_output = torch_output
        else:
            assert torch.equal(torch_output, first_output), f"Non-deterministic output on run {run_idx}"
