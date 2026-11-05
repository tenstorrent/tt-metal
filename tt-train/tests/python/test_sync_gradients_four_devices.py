# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The fp32 gradient sum over four devices on one axis, a ``[1, 4]`` ring.

With two devices every element of a sum is one add, so the two fp32 routes give the same bits whatever order they add
in. With four, each element takes three adds, so comparing the routes bit for bit checks that the reduce-scatter route
adds in the order ttnn's all_reduce does; the sum also runs as a ring here.
"""

import numpy as np
import pytest
import ttnn
import ttml

from two_device_gradients import (
    device_operations,
    divides_before_gathering,
    exact_mean,
    fp32_all_reduce_mean,
    make_gradients,
    per_device,
    per_device_tensor,
    placements,
)

pytestmark = pytest.mark.requires_device

_FOUR_DEVICES = (1, 4)
_DP_AXIS = 1
_FP32 = ttnn.DataType.FLOAT32


@pytest.fixture(scope="module")
def four_device_mesh(fresh_device_mesh):
    with fresh_device_mesh(
        _FOUR_DEVICES, ("_0", "dp"), what="fp32 gradient sum over four devices", require_mgd=True
    ) as mesh:
        yield mesh


# 512 x 1024 values per device is 2^19 elements, the reduce-scatter route's threshold, and the last dim splits into
# four whole-tile slices.
def test_reduce_scatter_route_matches_the_fp32_all_reduce_over_four_devices(four_device_mesh):
    values = make_gradients(512, 1024, seed=11, devices=4)
    gradients = per_device_tensor(values).get_value()

    fp32, operations = device_operations(
        lambda: ttml.core.distributed.synchronize_tensor(gradients, [_DP_AXIS], reduce_dtype=_FP32)
    )
    all_reduce = fp32_all_reduce_mean(gradients, [_DP_AXIS])
    bf16 = ttml.core.distributed.synchronize_tensor(gradients, [_DP_AXIS])

    assert divides_before_gathering(operations)
    fp32_values = per_device(fp32)
    np.testing.assert_array_equal(fp32_values, per_device(all_reduce))
    for copy in fp32_values[1:]:
        np.testing.assert_array_equal(copy, fp32_values[0])
    expected = exact_mean(values)[0]
    fp32_share = np.mean(fp32_values[0] == expected)
    assert fp32_share >= 0.95
    assert fp32_share > np.mean(per_device(bf16)[0] == expected)
    assert placements(fp32)[_DP_AXIS] == ("Replicate",)
