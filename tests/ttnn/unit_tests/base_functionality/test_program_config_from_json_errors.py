# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Malformed JSON passed to from_json must raise a catchable error, not abort the process.

The from_json_t deserializers used to be noexcept, so a missing key or wrong type threw out of a
noexcept function and std::terminate killed the interpreter. No device needed.
"""

import copy
import json

import pytest

import ttnn

# nanobind maps nlohmann exceptions to one of these depending on their base class.
ERRORS = (RuntimeError, ValueError, TypeError, IndexError)
MATCH = "json.exception"

ProgramConfig = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig


def _config_dict():
    cfg = ProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 2),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=2,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
    return json.loads(cfg.to_json())


def test_program_config_json_round_trip():
    d = _config_dict()
    restored = ProgramConfig.from_json(json.dumps(d))
    assert json.loads(restored.to_json()) == d


def _missing_grid_x(d):
    del d["compute_with_storage_grid_size"]["x"]


def _string_grid_x(d):
    d["compute_with_storage_grid_size"]["x"] = "abc"


def _missing_in0_block_w(d):
    del d["in0_block_w"]


def _string_in0_block_w(d):
    d["in0_block_w"] = "abc"


@pytest.mark.parametrize("mutate", [_missing_grid_x, _string_grid_x, _missing_in0_block_w, _string_in0_block_w])
def test_program_config_from_json_malformed_raises(mutate, expect_error):
    d = copy.deepcopy(_config_dict())
    mutate(d)
    with expect_error(ERRORS, MATCH):
        ProgramConfig.from_json(json.dumps(d))


def _memory_config_dict():
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        [1, 1024],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return mc, json.loads(mc.to_json())


def test_memory_config_json_round_trip():
    mc, d = _memory_config_dict()
    assert ttnn.MemoryConfig.from_json(json.dumps(d)) == mc


@pytest.mark.parametrize("field", ["grid", "shape", "orientation"])
def test_memory_config_from_json_shard_spec_missing_field_raises(field, expect_error):
    _, d = _memory_config_dict()
    del d["shard_spec"][field]
    with expect_error(ERRORS, MATCH):
        ttnn.MemoryConfig.from_json(json.dumps(d))


def test_memory_config_from_json_missing_top_level_key_raises(expect_error):
    _, d = _memory_config_dict()
    del d["memory_layout"]
    with expect_error(ERRORS, MATCH):
        ttnn.MemoryConfig.from_json(json.dumps(d))
