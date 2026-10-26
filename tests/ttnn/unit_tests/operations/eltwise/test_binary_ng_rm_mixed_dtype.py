# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Row-major binary_ng reads each operand at its own element stride.

The row-major reader used to pitch both circular buffers by operand A's byte
stride. A bf16 row is 2 bytes per element and an fp32 row is 4, so the wider
operand's second row was unpacked as the tail of the first row.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose

pytestmark = pytest.mark.use_module_device

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


def _rm_add(device, a_shape, b_shape, a_dtype, b_dtype):
    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.float32)
    b = torch.randn(b_shape, dtype=torch.float32)
    # Distinct per-column values so a stride mix-up cannot pass by accident.
    b = b + torch.arange(b_shape[-1], dtype=torch.float32)

    a_rm = ttnn.from_torch(a, dtype=a_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    b_rm = ttnn.from_torch(b, dtype=b_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.add(a_rm, b_rm)).float()

    out_dtype = _TORCH_DTYPE[a_dtype]
    golden = (a.float() + b.float()).to(out_dtype).float()
    assert_allclose(golden, got, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
    ],
    ids=["a_bf16_b_fp32", "a_fp32_b_bf16", "a_bf16_b_bf16", "a_fp32_b_fp32"],
)
@pytest.mark.parametrize("shape", [(2, 32), (2, 64)], ids=["w32", "w64"])
def test_rm_mixed_dtype_equal_shapes(device, a_dtype, b_dtype, shape):
    _rm_add(device, shape, shape, a_dtype, b_dtype)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1, 32), (4, 32)),
        ((4, 1), (4, 32)),
        ((1, 1), (2, 32)),
        ((4, 32), (1, 32)),
        ((4, 32), (4, 1)),
        ((2, 32), (1, 1)),
        ((1, 32), (4, 1)),
        ((4, 1), (1, 32)),
    ],
    ids=[
        "row_bcast_b",
        "col_bcast_b",
        "scalar_bcast_b",
        "row_bcast_a",
        "col_bcast_a",
        "scalar_bcast_a",
        "row_a_col_b",
        "col_a_row_b",
    ],
)
@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
    ],
    ids=["a_bf16_b_fp32", "a_fp32_b_bf16"],
)
def test_rm_mixed_dtype_broadcast(device, a_shape, b_shape, a_dtype, b_dtype):
    _rm_add(device, a_shape, b_shape, a_dtype, b_dtype)
