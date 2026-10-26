# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded reduction sanity coverage; the full matrices run in nightly/reduction."""

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.reduction import test_two_pass_reduction as regression
from tests.ttnn.unit_tests.operations.reduce.test_reduction import enabled_program_cache


def test_two_pass_hw_batch_combine(device, enabled_program_cache):
    regression.test_std_var_hw_compact_lane_combine(
        device, enabled_program_cache, ttnn.float32, True, (2, 3, 33, 128), (1, 2, 3)
    )


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
def test_two_pass_hw_partial_width(device, enabled_program_cache, dtype):
    regression.test_std_var_hw_compact_partial_width(device, enabled_program_cache, dtype, 15, 3, 33)


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat8_b)],
)
@pytest.mark.parametrize("op", [ttnn.var, ttnn.std])
def test_two_pass_hw_output_padding(device, torch_dtype, ttnn_dtype, op):
    regression.test_std_var_hw_output_padding_is_zero(device, torch_dtype, ttnn_dtype, op, 128, 64)


@pytest.mark.parametrize(
    "op,torch_op,dtype,dim,scalar",
    [
        (ttnn.std, torch.std, ttnn.bfloat16, -1, 2.5),
        (ttnn.var, torch.var, ttnn.float32, -2, 1.0),
        (ttnn.std, torch.std, ttnn.bfloat8_b, -2, 2.5),
    ],
)
def test_two_pass_w_h_output_padding(device, op, torch_op, dtype, dim, scalar):
    regression.test_std_var_w_h_output_padding_is_zero(device, op, torch_op, dtype, dim, 33, scalar)
