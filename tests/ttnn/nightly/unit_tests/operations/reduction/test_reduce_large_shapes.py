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
from tests.ttnn.unit_tests.operations.reduce.test_fast_reduce_nc import test_fast_reduce_nc as _fast_reduce_nc
from tests.ttnn.unit_tests.operations.reduce.test_intimg import test_cumsum_channel_last as _cumsum_channel_last
from tests.ttnn.unit_tests.operations.reduce.test_sum import test_sum_subcores as _sum_subcores


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
