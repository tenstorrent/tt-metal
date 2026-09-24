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


def test_isin_uint16_spans_two_subchunks(device):
    # Wide enough that the row is split, so a later subchunk is written at a non-zero offset.
    # Values stay inside uint16.
    elements = torch.arange(160000, dtype=torch.int64) % 30000
    test_elements = torch.tensor([0, 17, 1000, 29999], dtype=torch.int64)
    elements_ttnn = ttnn.from_torch(elements, device=device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    test_elements_ttnn = ttnn.from_torch(test_elements, device=device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)

    torch_result = torch.isin(elements, test_elements)
    ttnn_result = ttnn.to_torch(ttnn.experimental.isin(elements_ttnn, test_elements_ttnn))
    assert torch.equal(ttnn_result != 0, torch_result)
