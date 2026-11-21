# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


def _isin_matches(dtype, invert, device):
    # 190 is not a multiple of the uint32 width, and matches sit in the second half.
    elements = torch.arange(190, dtype=torch.int64)
    test_elements = torch.arange(100, 190, 7, dtype=torch.int64)
    elements_ttnn = ttnn.from_torch(elements, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    test_elements_ttnn = ttnn.from_torch(test_elements, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_result = torch.isin(elements, test_elements, invert=invert)
    ttnn_result = ttnn.to_torch(ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, invert=invert))
    assert ttnn_result.shape == torch_result.shape
    assert torch.equal(ttnn_result != 0, torch_result)


@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.uint8, ttnn.bfloat16])
@pytest.mark.parametrize("invert", [False, True])
def test_isin_narrow_dtype_writes_uint32_mask(dtype, invert, device):
    _isin_matches(dtype, invert, device)


# Sizes just past each dtype's single-subchunk limit on WH, so a later subchunk is written at a non-zero offset.
@pytest.mark.parametrize("dtype, size", [(ttnn.uint16, 160000), (ttnn.bfloat16, 160000), (ttnn.uint8, 210000)])
def test_isin_spans_two_subchunks(dtype, size, device):
    elements = torch.arange(size, dtype=torch.int64) % 250  # exact in uint8 and bfloat16
    test_elements = torch.tensor([0, 17, 100, 249], dtype=torch.int64)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.int64
    elements_ttnn = ttnn.from_torch(elements.to(torch_dtype), device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    test_elements_ttnn = ttnn.from_torch(
        test_elements.to(torch_dtype), device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    torch_result = torch.isin(elements, test_elements)
    ttnn_result = ttnn.to_torch(ttnn.experimental.isin(elements_ttnn, test_elements_ttnn))
    assert torch.equal(ttnn_result != 0, torch_result)


def _isin_inputs(device, shape=(190,)):
    elements = torch.arange(190, dtype=torch.int64).reshape(shape)
    test_elements = torch.arange(100, 190, 7, dtype=torch.int64)
    elements_ttnn = ttnn.from_torch(elements, device=device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    test_elements_ttnn = ttnn.from_torch(test_elements, device=device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return elements, test_elements, elements_ttnn, test_elements_ttnn


def test_isin_preallocated_output(device):
    elements, test_elements, elements_ttnn, test_elements_ttnn = _isin_inputs(device)
    out = ttnn.from_torch(
        torch.full((190,), 7, dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, out=out)
    assert torch.equal(ttnn.to_torch(out) != 0, torch.isin(elements, test_elements))


# The writer stores the whole flat mask into out, so any other shape or layout must be rejected.
@pytest.mark.parametrize(
    "elements_shape, out_shape, out_layout",
    [
        ((190,), (64,), ttnn.ROW_MAJOR_LAYOUT),
        ((190,), (1, 190), ttnn.ROW_MAJOR_LAYOUT),
        ((2, 95), (2, 95), ttnn.ROW_MAJOR_LAYOUT),
        ((190,), (190,), ttnn.TILE_LAYOUT),
    ],
)
def test_isin_rejects_mismatched_preallocated_output(elements_shape, out_shape, out_layout, device, expect_error):
    _, _, elements_ttnn, test_elements_ttnn = _isin_inputs(device, elements_shape)
    out = ttnn.from_torch(
        torch.zeros(out_shape, dtype=torch.int64), device=device, dtype=ttnn.uint32, layout=out_layout
    )
    with expect_error(RuntimeError, "Preallocated output"):
        ttnn.experimental.isin(elements_ttnn, test_elements_ttnn, out=out)
