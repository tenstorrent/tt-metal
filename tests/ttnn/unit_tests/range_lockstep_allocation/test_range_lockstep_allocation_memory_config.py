# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests that range_lockstep_allocation is part of a MemoryConfig's identity, and that the nanobind
binding for it is wired to the right C++ function.

The flag changes allocator semantics, so anything that describes or compares a MemoryConfig has
to see it. per_core_allocation was originally missing from to_json, operator== and the reflection
attributes, which collided its configs with lockstep ones onto a single cache key.

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


def _sharded_config():
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, _shard_spec())


def _configs():
    """A default-lockstep config and an otherwise-identical range lockstep config."""
    lockstep = _sharded_config()
    range_lockstep = _sharded_config()
    range_lockstep.experimental_set_range_lockstep_allocation(True)
    return lockstep, range_lockstep


# -- binding wiring ------------------------------------------------------------------------


def test_binding_exists():
    """A rename or a missing .def would surface here and nowhere in the C++ tests."""
    assert hasattr(_sharded_config(), "experimental_set_range_lockstep_allocation")


def test_binding_sets_range_lockstep_not_per_core():
    """The binding must reach the range lockstep overload, not the per-core one.

    Both namespaces export set_..._allocation(MemoryConfig&, bool), so dropping the range_lockstep
    include from the nanobind TU compiles and silently calls per-core.
    """
    config = _sharded_config()
    config.experimental_set_range_lockstep_allocation(True)
    serialized = json.loads(config.to_json())
    assert serialized["range_lockstep_allocation"] is True
    assert serialized["per_core_allocation"] is False, "binding resolved to the per-core overload"


def test_binding_disable_round_trips():
    config = _sharded_config()
    config.experimental_set_range_lockstep_allocation(True)
    config.experimental_set_range_lockstep_allocation(False)
    assert json.loads(config.to_json())["range_lockstep_allocation"] is False


# -- guards --------------------------------------------------------------------------------


def test_rejects_interleaved(expect_error):
    """An interleaved buffer spans every bank, so there is no narrower core set to scope to."""
    interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    with expect_error(RuntimeError, "range_lockstep_allocation requires a sharded memory layout"):
        interleaved.experimental_set_range_lockstep_allocation(True)


def test_rejects_per_core_allocation(expect_error):
    """One address across the cores, or an independent address on each -- not both."""
    config = _sharded_config()
    config.experimental_set_per_core_allocation(True)
    with expect_error(RuntimeError, "mutually exclusive"):
        config.experimental_set_range_lockstep_allocation(True)


def test_rejects_dram(expect_error):
    """The scan it narrows runs over L1 banks, so the flag would be ignored anywhere else."""
    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, _shard_spec())
    with expect_error(RuntimeError, "range_lockstep_allocation is only supported for L1 buffers"):
        dram.experimental_set_range_lockstep_allocation(True)


def test_per_core_rejects_range_lockstep(expect_error):
    """Mutual exclusion has to hold in both orders, or both flags end up set."""
    config = _sharded_config()
    config.experimental_set_range_lockstep_allocation(True)
    with expect_error(RuntimeError, "mutually exclusive"):
        config.experimental_set_per_core_allocation(True)


def test_disable_is_always_allowed():
    """Turning the flag off asserts nothing, so it must not be guarded."""
    interleaved = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    interleaved.experimental_set_range_lockstep_allocation(False)

    per_core = _sharded_config()
    per_core.experimental_set_per_core_allocation(True)
    per_core.experimental_set_range_lockstep_allocation(False)


# -- identity ------------------------------------------------------------------------------


def test_range_lockstep_breaks_equality():
    lockstep, range_lockstep = _configs()
    assert lockstep != range_lockstep, "configs differing only in range_lockstep_allocation must not compare equal"
    assert not (lockstep == range_lockstep)


def test_range_lockstep_changes_hash():
    lockstep, range_lockstep = _configs()
    assert hash(lockstep) != hash(range_lockstep), "range_lockstep_allocation must participate in the reflected hash"


def test_range_lockstep_in_repr():
    lockstep, range_lockstep = _configs()
    assert "range_lockstep_allocation=1" in repr(range_lockstep), repr(range_lockstep)
    assert "range_lockstep_allocation=0" in repr(lockstep), repr(lockstep)


def test_range_lockstep_serialized_to_json():
    lockstep, range_lockstep = _configs()
    assert json.loads(range_lockstep.to_json())["range_lockstep_allocation"] is True
    assert json.loads(lockstep.to_json())["range_lockstep_allocation"] is False
    # The two must not serialize identically -- this is what collides cache keys.
    assert range_lockstep.to_json() != lockstep.to_json()


def test_range_lockstep_round_trips_through_json():
    lockstep, range_lockstep = _configs()
    assert ttnn.MemoryConfig.from_json(range_lockstep.to_json()) == range_lockstep
    assert ttnn.MemoryConfig.from_json(lockstep.to_json()) == lockstep
    # Round-tripping a range lockstep config must not yield something equal to the default one.
    assert ttnn.MemoryConfig.from_json(range_lockstep.to_json()) != lockstep


def test_json_without_the_key_loads_as_default_lockstep():
    """JSON serialized before range_lockstep_allocation was emitted must still load."""
    lockstep, range_lockstep = _configs()
    legacy = json.loads(range_lockstep.to_json())
    legacy.pop("range_lockstep_allocation")
    assert ttnn.MemoryConfig.from_json(json.dumps(legacy)) == lockstep


def test_json_asking_for_both_modes_is_rejected(expect_error):
    """Hand-edited or corrupted JSON must trip the mutual-exclusion check, not pick a winner."""
    _lockstep, range_lockstep = _configs()
    both = json.loads(range_lockstep.to_json())
    both["per_core_allocation"] = True
    with expect_error(RuntimeError, "mutually exclusive"):
        ttnn.MemoryConfig.from_json(json.dumps(both))


def test_range_lockstep_is_independent_of_per_core():
    """The two flags must not alias onto one another in either direction."""
    range_lockstep = _sharded_config()
    range_lockstep.experimental_set_range_lockstep_allocation(True)
    per_core = _sharded_config()
    per_core.experimental_set_per_core_allocation(True)
    assert range_lockstep != per_core

    rl_json = json.loads(range_lockstep.to_json())
    pc_json = json.loads(per_core.to_json())
    assert (rl_json["range_lockstep_allocation"], rl_json["per_core_allocation"]) == (True, False)
    assert (pc_json["range_lockstep_allocation"], pc_json["per_core_allocation"]) == (False, True)
