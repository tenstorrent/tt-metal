# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests that per_core_allocation is part of a MemoryConfig's identity.

The flag changes allocator semantics -- each core gets an independent L1 address instead of one
lockstep address -- so anything that describes or compares a MemoryConfig has to see it. It used
to be missing from to_json, operator==, and the reflection attributes, so two configs differing
only in this flag compared equal, hashed equal, and serialized identically. Callers that key a
cache on to_json() (e.g. tt-blaze's TensorCache) collided per-core and lockstep variants onto one
artifact id.

No device needed: MemoryConfig is a host-side value.
"""

import json

import ttnn


def _shard_spec():
    return ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        [1, 1024],
        ttnn.ShardOrientation.ROW_MAJOR,
    )


def _configs():
    """A lockstep config and an otherwise-identical per-core config."""
    lockstep = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec())
    per_core = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec())
    per_core.experimental_set_per_core_allocation(True)
    return lockstep, per_core


def test_per_core_allocation_breaks_equality():
    lockstep, per_core = _configs()
    assert lockstep != per_core, "configs differing only in per_core_allocation must not compare equal"
    assert not (lockstep == per_core)


def test_per_core_allocation_changes_hash():
    lockstep, per_core = _configs()
    assert hash(lockstep) != hash(per_core), "per_core_allocation must participate in the reflected hash"


def test_per_core_allocation_in_repr():
    lockstep, per_core = _configs()
    assert "per_core_allocation=1" in repr(per_core), repr(per_core)
    assert "per_core_allocation=0" in repr(lockstep), repr(lockstep)


def test_per_core_allocation_serialized_to_json():
    lockstep, per_core = _configs()
    assert json.loads(per_core.to_json())["per_core_allocation"] is True
    assert json.loads(lockstep.to_json())["per_core_allocation"] is False
    # The two must not serialize identically -- this is what collided cache keys.
    assert per_core.to_json() != lockstep.to_json()


def test_per_core_allocation_round_trips_through_json():
    lockstep, per_core = _configs()
    assert ttnn.MemoryConfig.from_json(per_core.to_json()) == per_core
    assert ttnn.MemoryConfig.from_json(lockstep.to_json()) == lockstep
    # Round-tripping a per-core config must not yield something equal to the lockstep one.
    assert ttnn.MemoryConfig.from_json(per_core.to_json()) != lockstep


def test_json_without_the_key_loads_as_lockstep():
    """JSON serialized before per_core_allocation was emitted must still load, as lockstep."""
    lockstep, per_core = _configs()
    legacy = json.loads(per_core.to_json())
    legacy.pop("per_core_allocation")
    assert ttnn.MemoryConfig.from_json(json.dumps(legacy)) == lockstep
