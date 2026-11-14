# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


def _two_core_grid():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})


def _rank2_spec():
    return ttnn.TensorSpec([64, 64], ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("method", ["sharded_across_dims", "sharded_across_dims_except"])
def test_sharding_dims_reject_out_of_range_negative_dim(method, expect_error):
    # Negative indexing on a rank-2 shape reaches -4 through the internal padding, so dim -3 used to
    # write the padding and leave the dimension untouched instead of raising.
    spec = _rank2_spec()

    with expect_error(RuntimeError, "out of range for a tensor of rank 2"):
        getattr(spec, method)([-3], _two_core_grid())


@pytest.mark.parametrize("method", ["sharded_across_dims", "sharded_across_dims_except"])
def test_sharding_dims_valid_negative_dim_control(method):
    spec = _rank2_spec()

    sharded = getattr(spec, method)([-2], _two_core_grid())

    assert sharded.memory_config.is_sharded()
    assert sharded.memory_config.nd_shard_spec is not None
    assert sharded.shape == spec.shape
