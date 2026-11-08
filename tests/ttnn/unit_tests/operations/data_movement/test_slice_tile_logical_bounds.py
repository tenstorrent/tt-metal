# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A slice end past the input's logical shape must be rejected, not padded out.

A TILE tensor's padded shape is wider than its logical shape, and the tile slice route rounds the
requested ends up to a tile boundary before the device op sees them, so the device op only validates
padded bounds. An end between the two was therefore accepted and the result came back with more
elements than the input had, reading tile padding lanes. A ROW_MAJOR tensor, whose two shapes agree,
already rejected the same request in the device op.
"""

import pytest

import torch

import ttnn

pytestmark = pytest.mark.use_module_device

# Logical width 35, so the padded width is 64: the gap an oversized end used to hide in.
SHAPE = (1, 1, 3, 35)
LOGICAL_WIDTH = SHAPE[-1]

# 36 and 40 sit inside the padded width; 63 is the last padded lane.
OVERSIZED_ENDS = [36, 40, 63]


@pytest.mark.parametrize("end_w", OVERSIZED_ENDS)
@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint16], ids=["bf16", "fp32", "i32", "u16"]
)
def test_slice_tile_end_past_logical_width_rejected(device, dtype, end_w, expect_error):
    """TILE input: an end beyond the logical width used to return the extra columns."""
    torch_input = torch.randint(0, 1000, SHAPE, dtype=torch.int64).to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "must not exceed the input tensor's logical shape"):
        ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 1, 3, end_w), steps=(1, 1, 1, 1))


@pytest.mark.parametrize("end_w", OVERSIZED_ENDS)
@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint16], ids=["bf16", "fp32", "i32", "u16"]
)
def test_slice_row_major_end_past_logical_width_rejected(device, dtype, end_w, expect_error):
    """Control: on ROW_MAJOR the same request was already rejected, and must stay rejected."""
    torch_input = torch.randint(0, 1000, SHAPE, dtype=torch.int64).to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "must not exceed the input tensor's logical shape"):
        ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 1, 3, end_w), steps=(1, 1, 1, 1))


@pytest.mark.parametrize("end_w", [LOGICAL_WIDTH, LOGICAL_WIDTH - 1, 1, 33])
@pytest.mark.parametrize(
    "dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint16], ids=["bf16", "fp32", "i32", "u16"]
)
def test_slice_tile_end_within_logical_width_unchanged(device, dtype, end_w):
    """Control: an end inside the logical shape slices as before on both layouts."""
    torch_input = torch.randint(0, 1000, SHAPE, dtype=torch.int64).to(ttnn.ttnn_dtype_to_torch_dtype(dtype))
    expected = torch_input[..., :end_w]

    for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
        ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
        actual = ttnn.to_torch(ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 1, 3, end_w), steps=(1, 1, 1, 1)))
        assert tuple(actual.shape) == tuple(expected.shape), (layout, actual.shape, expected.shape)
        assert torch.equal(actual.to(torch.int64), expected.to(torch.int64)), layout


def test_slice_tile_exact_logical_width_is_a_view_not_a_padding_read(device):
    """Asking for exactly the logical width clips nothing, so no padding lane can appear."""
    torch_input = torch.randint(0, 1000, SHAPE, dtype=torch.int64).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.to_torch(ttnn.slice(ttnn_input, starts=(0, 0, 0, 0), ends=(1, 1, 3, 35), steps=(1, 1, 1, 1)))
    assert tuple(actual.shape) == tuple(torch_input.shape)
    assert torch.equal(actual.to(torch.int64), torch_input.to(torch.int64))
