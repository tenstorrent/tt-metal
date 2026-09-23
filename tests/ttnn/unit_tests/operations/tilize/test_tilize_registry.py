# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime registry tagging for tilize: validate() must tag a live call exactly as
the scenario-dict INPUT_TAGGERS tag the golden case it came from.

Regression for the shard_api tag: an allocated legacy 2-D sharded tensor reports
a derived `nd_shard_spec`, and an ND tensor with a 2-D equivalent reports a
legacy `memory_layout` + `shard_spec`, so `nd_shard_spec is not None` alone
mis-tags both. validate() reads `created_with_nd_shard_spec` instead.
"""
import torch
import ttnn

from ttnn.operations.tilize import validate


def _crs(x1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, 0))})


def _rm(device, shape, memory_config):
    x = torch.zeros(shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )


# Sharding is SUPPORTED since Refinement 1, so validate() returns the tagged axes
# instead of refusing; the tags themselves are what this file pins.
def test_legacy_sharded_input_tags_legacy_2d(device):
    spec = ttnn.ShardSpec(_crs(3), (128, 64), ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    t = _rm(device, (1, 1, 512, 64), mc)
    axes = validate(t, mc)
    assert axes["shard_api"] == "legacy_2d"
    assert axes["out_scheme"] == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert axes["buffer"] == "l1_to_l1"


def test_nd_sharded_input_tags_nd(device):
    nd = ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), _crs(1), ttnn.ShardOrientation.ROW_MAJOR)
    t = _rm(device, (1, 1, 128, 64), ttnn.MemoryConfig(ttnn.BufferType.L1, nd))
    axes = validate(t, ttnn.DRAM_MEMORY_CONFIG)
    assert axes["shard_api"] == "nd"
    assert axes["out_scheme"] == "interleaved"
    assert axes["buffer"] == "l1_to_dram"


def test_interleaved_phase0_cell_is_supported(device):
    t = _rm(device, (1, 1, 64, 64), ttnn.DRAM_MEMORY_CONFIG)
    axes = validate(t)
    assert axes["shard_api"] == "none" and axes["tile_grid"] == "small"
