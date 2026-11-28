# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Last-dim concat of many TILE inputs whose concat dim is not tile-aligned.

ConcatTiledUnalignedProgramFactory holds a whole 32-row band of every input in its circular buffers, so
the input count decides whether one program fits. ttnn::concat splits a longer list into groups of at
most kMaxTiledUnalignedConcatInputs inputs and concatenates the group results, which keeps every step on
the on-device paths. Above the old hardcoded 32-input cap these counts used to fall back to the composite
untilize/transpose/chunk pipeline, which is orders of magnitude slower and materializes a full-size
row-major intermediate per input.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

# (input shape, torch dtype, ttnn dtype). Both widths leave tile padding on the concat dim.
SHAPES_AND_DTYPES = [
    pytest.param([1, 32, 16], torch.bfloat16, ttnn.bfloat16, id="r3_w16_bf16"),
    pytest.param([1, 32, 16], torch.float32, ttnn.float32, id="r3_w16_fp32"),
    pytest.param([1, 2, 32, 48], torch.bfloat16, ttnn.bfloat16, id="r4_w48_bf16"),
    pytest.param([1, 2, 32, 48], torch.float32, ttnn.float32, id="r4_w48_fp32"),
]

# The old cap was 32 inputs; one more already fell back. These span both sides of it and several group
# splits, including a count whose groups and final concat are tile-aligned.
NUM_INPUTS = [pytest.param(n, id=str(n) + "_inputs") for n in (33, 40, 64, 100)]


@pytest.mark.parametrize("num_inputs", NUM_INPUTS)
@pytest.mark.parametrize("shape, torch_dtype, ttnn_dtype", SHAPES_AND_DTYPES)
def test_concat_many_unaligned_tile_inputs(device, num_inputs, shape, torch_dtype, ttnn_dtype):
    torch.manual_seed(0)
    torch_input_tensors = [torch.rand(shape, dtype=torch_dtype) for _ in range(num_inputs)]
    torch_output_tensor = torch.concat(torch_input_tensors, dim=-1)

    input_tensors = [
        ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for torch_input_tensor in torch_input_tensors
    ]

    output = ttnn.concat(input_tensors, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert_equal(torch_output_tensor, ttnn.to_torch(output))


@pytest.mark.parametrize("num_inputs", NUM_INPUTS)
@pytest.mark.parametrize("shape, torch_dtype, ttnn_dtype", SHAPES_AND_DTYPES)
def test_concat_many_unaligned_tile_inputs_dram_output(device, num_inputs, shape, torch_dtype, ttnn_dtype):
    # Same counts with the output in DRAM, which is the buffer type the routing-time L1 reservation
    # assumes.
    torch.manual_seed(0)
    torch_input_tensors = [torch.rand(shape, dtype=torch_dtype) for _ in range(num_inputs)]
    torch_output_tensor = torch.concat(torch_input_tensors, dim=-1)

    input_tensors = [
        ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        for torch_input_tensor in torch_input_tensors
    ]

    output = ttnn.concat(input_tensors, dim=-1)
    assert_equal(torch_output_tensor, ttnn.to_torch(output))


@pytest.mark.parametrize("num_inputs", [pytest.param(32, id="32_inputs")])
def test_concat_at_the_input_limit_is_unchanged(device, num_inputs):
    # At the limit the call still takes the single-program path, so every value must match exactly.
    shape = [1, 32, 16]
    torch.manual_seed(0)
    torch_input_tensors = [torch.rand(shape, dtype=torch.bfloat16) for _ in range(num_inputs)]
    input_tensors = [
        ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for torch_input_tensor in torch_input_tensors
    ]

    output = ttnn.concat(input_tensors, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert_equal(torch.concat(torch_input_tensors, dim=-1), ttnn.to_torch(output))
