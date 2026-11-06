# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
JSON serialization of NdShardSpec / ND MemoryConfig must carry the shard shape.

Shape serialized through a span that had no to_json overload, so shard_shape became an error
string: configs differing only in shard shape serialized identically and the JSON could not be
read back. No device needed.
"""

import json

import pytest

import ttnn


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])


def _spec(shard_shape):
    return ttnn.NdShardSpec(shard_shape, _grid(), orientation=ttnn.ShardOrientation.ROW_MAJOR)


def _config(shard_shape):
    return ttnn.MemoryConfig(ttnn.BufferType.L1, _spec(shard_shape))


@pytest.mark.parametrize("shard_shape", [[64], [2, 32, 64], [1, 2, 32, 32]])
def test_nd_shard_spec_json_round_trip(shard_shape):
    spec = _spec(shard_shape)
    restored = ttnn.NdShardSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.to_json() == spec.to_json()


@pytest.mark.parametrize("shard_shape", [[64], [2, 32, 64], [1, 2, 32, 32]])
def test_nd_memory_config_json_round_trip(shard_shape):
    config = _config(shard_shape)
    restored = ttnn.MemoryConfig.from_json(config.to_json())
    assert restored == config
    assert restored.to_json() == config.to_json()


@pytest.mark.parametrize("shard_shape", [[64], [2, 32, 64], [1, 2, 32, 32]])
def test_nd_shard_shape_dims_in_json(shard_shape):
    text = _config(shard_shape).to_json()
    assert "Unsupported type" not in text
    assert shard_shape in _find_lists(json.loads(text)), text


def _find_lists(obj):
    if isinstance(obj, list):
        yield obj
        for item in obj:
            yield from _find_lists(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _find_lists(item)


def test_shard_shape_participates_in_json_identity():
    a = _config([2, 32, 64])
    b = _config([1, 64, 32])
    assert a != b
    assert a.to_json() != b.to_json()
    assert _spec([2, 32, 64]).to_json() != _spec([1, 64, 32]).to_json()
