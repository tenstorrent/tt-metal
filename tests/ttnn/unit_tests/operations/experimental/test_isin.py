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


@pytest.mark.parametrize(
    "case, message",
    [
        ("host_elements", "Elements tensor is not on a device"),
        ("deallocated_elements", "Elements tensor's buffer is null"),
        ("host_test_elements", "Test elements tensor is not on a device"),
        ("deallocated_test_elements", "Test elements tensor's buffer is null"),
    ],
)
def test_isin_rejects_inputs_not_on_device(case, message, expect_error, device):
    elements = torch.arange(100, dtype=torch.int32)
    test_elements = torch.tensor([3, 5, 7], dtype=torch.int32)
    elements_ttnn = ttnn.from_torch(elements, dtype=ttnn.int32, device=None if case == "host_elements" else device)
    test_elements_ttnn = ttnn.from_torch(
        test_elements, dtype=ttnn.int32, device=None if case == "host_test_elements" else device
    )
    if case == "deallocated_elements":
        ttnn.deallocate(elements_ttnn)
    elif case == "deallocated_test_elements":
        ttnn.deallocate(test_elements_ttnn)

    with expect_error(RuntimeError, message):
        ttnn.experimental.isin(elements_ttnn, test_elements_ttnn)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_isin_rejects_block_float(dtype, expect_error, device):
    values = ttnn.from_torch(torch.ones(32, 32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "is not supported"):
        ttnn.experimental.isin(values, values)


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("shape", [[0], [2, 0, 3]])
def test_isin_empty_elements(shape, layout, use_out, device):
    elements = ttnn.from_torch(torch.zeros(shape, dtype=torch.int32), dtype=ttnn.int32, layout=layout, device=device)
    test_elements = ttnn.from_torch(torch.tensor([0, 1], dtype=torch.int32), dtype=ttnn.int32, device=device)
    out = (
        ttnn.from_torch(torch.zeros(shape, dtype=torch.int64), dtype=ttnn.uint32, layout=layout, device=device)
        if use_out
        else None
    )

    result = ttnn.experimental.isin(elements, test_elements, out=out)

    assert result.dtype == ttnn.uint32
    assert result.layout == layout
    assert list(result.shape) == shape
