# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Large-shape rungs lifted out of the ttnn reduce sanity group.
Each test delegates to the sanity implementation so the assertions can never drift apart.
"""

import pytest

import torch

import ttnn

from tests.ttnn.unit_tests.operations.test_utils import compute_kernel_options, compute_kernel_ids
from tests.ttnn.unit_tests.operations.reduce.test_cumprod import test_cumprod_normal as _cumprod_normal
from tests.ttnn.unit_tests.operations.reduce.test_cumprod import test_cumprod_backward as _cumprod_backward
from tests.ttnn.unit_tests.operations.reduce.test_cumprod import test_cumprod_preallocated as _cumprod_preallocated
from tests.ttnn.unit_tests.operations.reduce.test_argmax import test_argmax as _argmax
from tests.ttnn.unit_tests.operations.reduce.test_cumsum import test_cumsum as _cumsum
from tests.ttnn.unit_tests.operations.reduce.test_cumsum import test_cumsum_backward as _cumsum_backward
from tests.ttnn.unit_tests.operations.reduce.test_cumsum import test_cumsum_bf16_accuracy as _cumsum_bf16_accuracy
from tests.ttnn.unit_tests.operations.reduce.test_cumsum import (
    test_cumsum_disable_compensated_sum as _cumsum_disable_compensated_sum,
)
from tests.ttnn.unit_tests.operations.reduce.test_cumsum import (
    test_cumsum_fp32_long_scan_accuracy as _cumsum_fp32_long_scan_accuracy,
)
from tests.ttnn.unit_tests.operations.reduce.test_cumsum import (
    test_cumsum_with_preallocated_output as _cumsum_with_preallocated_output,
)
from tests.ttnn.unit_tests.operations.reduce.test_fast_reduce_nc import test_fast_reduce_nc as _fast_reduce_nc
from tests.ttnn.unit_tests.operations.reduce.test_intimg import test_cumsum_channel_last as _cumsum_channel_last
from tests.ttnn.unit_tests.operations.reduce.test_reduction import test_2d_topk as _2d_topk
from tests.ttnn.unit_tests.operations.reduce.test_sum import test_sum_subcores as _sum_subcores
from tests.ttnn.unit_tests.operations.reduce.test_topk import test_topk as _topk
from tests.ttnn.unit_tests.operations.reduce.test_topk import test_topk_large_2d_shapes as _topk_large_2d_shapes


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize("shape", [[1000, 32, 32]])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_large(dim, shape, dtypes, device):
    _cumprod_normal(dim, shape, dtypes, device)


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize("shape", [[1000, 32, 32]])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
        (torch.bfloat16, ttnn.float32),
    ],
)
def test_cumprod_backward_large(dim, shape, dtypes, device):
    _cumprod_backward(dim, shape, dtypes, device)


@pytest.mark.parametrize("dim", [0, 2, -1])
@pytest.mark.parametrize("shape", [[1000, 32, 32]])
@pytest.mark.parametrize(
    "dtypes",
    [
        (torch.float32, None),
    ],
)
def test_cumprod_preallocated_large(dim, shape, dtypes, device):
    _cumprod_preallocated(dim, shape, dtypes, device)


@pytest.mark.parametrize(
    "input_shape_nhwc",
    [
        # fmt: off
        ([1, 48, 160, 256]),
        ([1, 96, 160, 256])
    ],
    ids=["OFT8", "big_one"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["DRAM"])
def test_cumsum_channel_last_large(device, input_shape_nhwc, dtype, memory_config):
    _cumsum_channel_last(device, input_shape_nhwc, dtype, memory_config)


@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # single core
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("shape", [(16, 41, 63, 63)])
def test_sum_subcores_large(device, sub_core_grids, dtype, shape):
    _sum_subcores(device, sub_core_grids, dtype, shape)


# mixtral_2k differs from the sanity group's mixtral_1k in H (2048 vs 1024)
@pytest.mark.parametrize(
    "input_shape",
    ([1, 8, 2048, 4096],),
    ids=["mixtral_2k"],
)
@pytest.mark.parametrize(
    "dims",
    ([0], [1], [0, 1]),
    ids=["0", "1", "0_1"],
)
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
@pytest.mark.parametrize("dataformat", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bfloat16", "bfloat8_b"])
def test_fast_reduce_nc_large(input_shape, dims, compute_kernel_options, dataformat, device):
    _fast_reduce_nc(input_shape, dims, compute_kernel_options, dataformat, device)


# 128256 is the Llama-3 vocab.
@pytest.mark.parametrize("dim1", [1])
@pytest.mark.parametrize("dim2", [128256])
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("k", [50])
@pytest.mark.parametrize("largest", [True])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_large_2d_topk(device, dim1, dim2, dim, k, largest, dtype):
    _2d_topk(device, dim1, dim2, dim, k, largest, dtype)


# 151936 is the Qwen vocab: one long sequential scan on one core.
@pytest.mark.parametrize("size, dim", [([1, 151936], -1)])
@pytest.mark.parametrize(
    "dtypes",
    [(torch.float32, None), (torch.bfloat16, ttnn.bfloat16), (torch.int32, ttnn.int32)],
)
def test_cumsum_large(size, dim, dtypes, device):
    _cumsum(size, dim, dtypes, device)


@pytest.mark.parametrize("size, dim", [([1, 151936], -1)])
@pytest.mark.parametrize("dtypes", [(torch.float32, None), (torch.float32, ttnn.bfloat16)])
def test_cumsum_with_preallocated_output_large(size, dim, dtypes, device):
    _cumsum_with_preallocated_output(size, dim, dtypes, device)


@pytest.mark.parametrize("size, dim", [([1, 151936], -1)])
@pytest.mark.parametrize("dtypes", [(torch.float32, None)])
def test_cumsum_backward_large(size, dim, dtypes, device):
    _cumsum_backward(size, dim, dtypes, device)


@pytest.mark.parametrize(
    "size, dim",
    [
        ([1, 72192, 9], 1),  # the shape from #55542
        ([1, 151936], -1),
        ([4, 65536], -1),
    ],
)
@pytest.mark.parametrize("sequence_type", ["iid", "piecewise_constant"])
def test_cumsum_fp32_long_scan_accuracy_large(size, dim, sequence_type, device):
    _cumsum_fp32_long_scan_accuracy(size, dim, sequence_type, device)


# The #55542 length; the sanity group runs the same check at n=8192.
def test_cumsum_disable_compensated_sum_large(device):
    _cumsum_disable_compensated_sum(device, n=72192)


@pytest.mark.parametrize(
    "size, dim",
    [
        ([1, 151936], -1),
        ([1, 72192, 9], 1),  # the shape from #55542
    ],
)
@pytest.mark.parametrize("sequence_type", ["iid", "piecewise_constant"])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_cumsum_bf16_accuracy_large(size, dim, sequence_type, reverse_order, device):
    _cumsum_bf16_accuracy(size, dim, sequence_type, reverse_order, device)


# k=3000 runs on one core (k > 64 and above the routed range); the sanity group keeps a W=3100 case.
@pytest.mark.parametrize(
    "dtype", (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32), ids=["BFLOAT16_B", "BFLOAT8_B", "FLOAT32"]
)
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 32768, 3, 3000),
        (1, 1, 32, 18992, 3, 3000),
    ),
)
@pytest.mark.parametrize("sorted", [True, False])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_large_k(N, C, H, W, dim, k, dtype, sorted, largest, device):
    _topk(N, C, H, W, dim, k, dtype, sorted, largest, device, None)


# 151936 is the Qwen vocab, 128256 the Llama-3 vocab.
@pytest.mark.parametrize("dtype", (ttnn.bfloat16,), ids=["BFLOAT16_B"])
@pytest.mark.parametrize(
    "N, C, H, W, dim, k",
    (
        (1, 1, 32, 151936, 3, 50),
        (1, 1, 32, 128256, 3, 50),
    ),
)
@pytest.mark.parametrize("sorted", [True, False])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("pass_indices_tensor", [True, False])
def test_topk_large_2d_shapes_vocab(N, C, H, W, dim, k, dtype, sorted, largest, device, pass_indices_tensor):
    _topk_large_2d_shapes(N, C, H, W, dim, k, dtype, sorted, largest, device, None, pass_indices_tensor)


# The batch dims the sanity group cuts (it keeps the last two dims, which set the work split).
@pytest.mark.parametrize(
    "tensor_shape, tensor_layout, dim, keepdim, dtype, error_msg, enable_secondary_dm",
    [
        ([16, 32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, torch.bfloat16, None, None),
        ([16, 32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, torch.float32, None, None),
        ([16, 32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, torch.int32, None, None),
        ([16, 32, 64, 128], ttnn.TILE_LAYOUT, -1, True, torch.bfloat16, None, None),
        ([16, 32, 64, 128], ttnn.TILE_LAYOUT, -1, False, torch.float32, None, None),
        ([16, 32, 70, 130], ttnn.TILE_LAYOUT, -1, True, torch.bfloat16, None, None),
        ([16, 32, 70, 130], ttnn.TILE_LAYOUT, -1, False, torch.bfloat16, None, None),
        ([12, 24, 48, 96], ttnn.ROW_MAJOR_LAYOUT, -1, True, torch.bfloat16, None, None),
    ],
)
def test_argmax_large(
    device, tensor_shape, tensor_layout, dim, keepdim, dtype, error_msg, enable_secondary_dm, expect_error
):
    _argmax(device, tensor_shape, tensor_layout, dim, keepdim, dtype, error_msg, enable_secondary_dm, expect_error)
