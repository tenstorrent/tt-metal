# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Row-major binary_ng row broadcast serves the broadcast page from L1.

Every row block of a core re-reads the one broadcast page. The reader keeps that page in a circular buffer and
refetches it only when the (batch, channel) page changes, so these shapes cover many row blocks per core, batch
dims that move the page, widths that are not tile multiples, pages above the L1 capacity and mixed dtypes.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_allclose

pytestmark = pytest.mark.use_module_device

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
_OPS = {
    "add": (ttnn.add, torch.add),
    "sub": (ttnn.subtract, torch.sub),
    "mul": (ttnn.multiply, torch.mul),
}


def _run(device, a_shape, b_shape, a_dtype, b_dtype, op="add"):
    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.float32)
    # Distinct values along the last dim so a wrong chunk or page cannot pass by accident.
    b = torch.randn(b_shape, dtype=torch.float32) + torch.arange(b_shape[-1], dtype=torch.float32)

    a_rm = ttnn.from_torch(a, dtype=a_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    b_rm = ttnn.from_torch(b, dtype=b_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_op, torch_op = _OPS[op]
    got = ttnn.to_torch(ttnn_op(a_rm, b_rm)).float()

    golden = torch_op(ttnn.to_torch(a_rm).float(), ttnn.to_torch(b_rm).float())
    assert tuple(got.shape) == tuple(golden.shape)
    assert_allclose(golden, got, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("width", [31, 33, 100, 1000, 1025, 4096])
@pytest.mark.parametrize("height", [7, 257])
@pytest.mark.parametrize("bcast_b", [True, False], ids=["bcast_b", "bcast_a"])
def test_rm_row_bcast_many_row_blocks(device, dtype, width, height, bcast_b):
    full, row = (1, 1, height, width), (1, 1, 1, width)
    a_shape, b_shape = (full, row) if bcast_b else (row, full)
    _run(device, a_shape, b_shape, dtype, dtype, "sub")


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((2, 3, 40, 100), (2, 3, 1, 100)),
        ((2, 3, 40, 100), (1, 3, 1, 100)),
        ((2, 3, 40, 100), (2, 1, 1, 100)),
        ((2, 1, 40, 100), (1, 3, 1, 100)),
        ((3, 2, 5, 40, 100), (3, 2, 5, 1, 100)),
        ((3, 2, 5, 40, 100), (1, 2, 1, 1, 100)),
        ((4, 16, 64, 1000), (1, 16, 1, 1000)),
        ((1, 100), (3, 50, 100)),
        ((3, 50, 100), (1, 100)),
    ],
)
def test_rm_row_bcast_batch_dims(device, dtype, a_shape, b_shape):
    _run(device, a_shape, b_shape, dtype, dtype, "mul")


@pytest.mark.parametrize("width", [4096, 8192, 8193, 9000])
def test_rm_row_bcast_page_over_l1_capacity(device, width):
    # 8192 bf16 elements fill the 16 KiB cache page; wider rows read the broadcast page from DRAM as before.
    _run(device, (1, 1, 65, width), (1, 1, 1, width), ttnn.bfloat16, ttnn.bfloat16)
    _run(device, (1, 1, 65, width // 2), (1, 1, 1, width // 2), ttnn.float32, ttnn.float32)


@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)],
    ids=["a_bf16_b_fp32", "a_fp32_b_bf16"],
)
@pytest.mark.parametrize("width", [33, 1000, 4096])
@pytest.mark.parametrize("bcast_b", [True, False], ids=["bcast_b", "bcast_a"])
def test_rm_row_bcast_mixed_dtype(device, a_dtype, b_dtype, width, bcast_b):
    full, row = (1, 1, 65, width), (1, 1, 1, width)
    a_shape, b_shape = (full, row) if bcast_b else (row, full)
    _run(device, a_shape, b_shape, a_dtype, b_dtype)


def test_rm_row_bcast_program_cache_reuse(device):
    # One cached program serves several shapes, including a broadcast page above the cache capacity.
    for width, height in [(1000, 64), (4096, 33), (9000, 40), (33, 300), (1000, 65)]:
        _run(device, (1, 1, height, width), (1, 1, 1, width), ttnn.bfloat16, ttnn.bfloat16)
