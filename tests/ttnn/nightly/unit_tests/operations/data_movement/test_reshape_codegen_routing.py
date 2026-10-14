# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) a perf-demoted case
# still lands on native. This block is emitted from the port's coverage ledger
# and is regenerated on every port update; hand-add off-grid regressions after its END line.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.reshape` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see reshape_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.reshape_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.reshape_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_CALL_CONTRACT_CASE = ([12, 100], {"shape": [12, 10, 10]}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)


def test_reshape_codegen_unsupported_output_memory_config_falls_back(device):
    shape, kwargs, dtype, layout, placement = _CALL_CONTRACT_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    sharded = ttnn.create_sharded_memory_config(
        list(_force_native(xt, **kwargs).shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    # Sweeps only ever run the default memory config, which inherits the input's placement,
    # so nothing else in this file reaches the output side of the call contract.
    golden = _force_native(xt, **kwargs, memory_config=sharded)
    out = ttnn.reshape(xt, **kwargs, memory_config=sharded)
    assert out.memory_config() == golden.memory_config()
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(out))


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---


_TILE_ROWS = [
    ([1, 64, 1024], {"shape": [1, 64, 8, 128]}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG),
    ([1, 64, 1024], {"shape": [1, 64, 8, 128]}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _TILE_ROWS, ids=["l1", "dram"])
def test_reshape_codegen_tile_rows_program_key_is_frontier_invariant(device, shape, kwargs, dtype, layout, placement):
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_before = device.num_program_cache_entries()
    # Trace capture compiles nothing it has not already cached, so a CB plan that reads live
    # L1 occupancy would re-key the cache once this tensor moves the frontier.
    resident = ttnn.from_torch(
        torch.zeros([1, 1, 256, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
        msg = "the program key moved with the L1 frontier, so the op cannot be traced"
        assert device.num_program_cache_entries() == entries_before, msg
    finally:
        ttnn.deallocate(resident)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32768}], indirect=True)
def test_reshape_routes_tile_rows_case_under_trace(device):
    shape, kwargs, dtype, layout, placement = _TILE_ROWS[1]
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    ttnn.reshape(xt, **kwargs)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = ttnn.reshape(xt, **kwargs)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
    assert_equal(golden, ttnn.to_torch(out))
