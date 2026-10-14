# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Contracts: (1) every case the codegen scope or call contract rejects falls back to native and matches
# reshape_force_native; (2) a demoted case runs native; (3) a routed tile-rows call traces and replays new data;
# (4) a small row-major call keeps its program key when the L1 frontier moves.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.reshape` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see reshape_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.reshape_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.reshape_force_codegen

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _make_input(shape, dtype, seed=0):
    torch.manual_seed(seed)
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _to_device(device, source, dtype, layout, mem=_DRAM):
    return ttnn.from_torch(source, dtype=dtype, layout=layout, device=device, memory_config=mem)


_CALL_CONTRACT_CASE = ([12, 100], {"shape": [12, 10, 10]}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)


def test_reshape_codegen_unsupported_output_memory_config_falls_back(device):
    shape, kwargs, dtype, layout, placement = _CALL_CONTRACT_CASE
    xt = _to_device(device, _make_input(shape, dtype), dtype, layout, placement)
    sharded = ttnn.create_sharded_memory_config(
        list(_force_native(xt, **kwargs).shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    golden = _force_native(xt, **kwargs, memory_config=sharded)
    out = ttnn.reshape(xt, **kwargs, memory_config=sharded)
    assert out.memory_config() == golden.memory_config()
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(out))


# (id, in shape, out shape, dtype, layout): outside the codegen scope.
_OUT_OF_SCOPE = [
    ("tile_unaligned_width", [1, 1, 32, 100], [1, 1, 100, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ("bfloat8_b", [1, 1, 64, 128], [1, 1, 128, 64], ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


@pytest.mark.parametrize("case,in_shape,out_shape,dtype,layout", _OUT_OF_SCOPE, ids=[c[0] for c in _OUT_OF_SCOPE])
def test_reshape_codegen_out_of_scope_falls_back(device, case, in_shape, out_shape, dtype, layout):
    xt = _to_device(device, _make_input(in_shape, dtype), dtype, layout)
    golden = ttnn.to_torch(_force_native(xt, out_shape))
    assert_equal(golden, ttnn.to_torch(ttnn.reshape(xt, out_shape)))


_SHARD_STRATEGIES = {
    "height": (ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(x=2, y=1)),
    "width": (ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(x=2, y=1)),
    "block": (ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(x=2, y=2)),
}


@pytest.mark.parametrize("strategy", list(_SHARD_STRATEGIES), ids=list(_SHARD_STRATEGIES))
def test_reshape_codegen_sharded_input_falls_back(device, strategy):
    shape, out_shape = [1, 1, 64, 256], [1, 1, 128, 128]
    source = _make_input(shape, ttnn.bfloat16)
    shard_strategy, core_grid = _SHARD_STRATEGIES[strategy]
    sharded = ttnn.create_sharded_memory_config(shape, core_grid=core_grid, strategy=shard_strategy)
    xt = _to_device(device, source, ttnn.bfloat16, ttnn.TILE_LAYOUT, sharded)
    golden = ttnn.to_torch(_force_native(xt, out_shape, memory_config=_DRAM))
    out = ttnn.to_torch(ttnn.reshape(xt, out_shape, memory_config=_DRAM))
    assert_equal(golden, out)
    assert_equal(source.reshape(out_shape), out)


def test_reshape_codegen_explicit_pad_value_falls_back(device):
    shape, out_shape = [1, 64, 1024], [1, 64, 8, 128]
    xt = _to_device(device, _make_input(shape, ttnn.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    golden = _force_native(xt, out_shape, pad_value=0.0)
    out = ttnn.reshape(xt, out_shape, pad_value=0.0)
    assert_equal(golden.cpu().to_torch_with_padded_shape(), out.cpu().to_torch_with_padded_shape())


def test_reshape_codegen_explicit_padded_shape_falls_back(device):
    shape, logical, padded = [1, 2, 30, 256], [1, 1, 60, 256], [1, 1, 64, 256]
    source = _make_input(shape, ttnn.bfloat16)
    xt = _to_device(device, source, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    golden = ttnn.to_torch(_force_native(xt, logical))
    out = ttnn.reshape(xt, ttnn.Shape(logical), ttnn.Shape(padded))
    assert list(out.shape) == logical
    assert_equal(golden, ttnn.to_torch(out))
    assert_equal(source.reshape(logical), ttnn.to_torch(out))


def test_reshape_codegen_demoted_case_runs_native(device):
    # A width change saves only 2x the reads on far fewer cores, so the tile-rows program is demoted.
    shape, out_shape = [1, 1, 64, 96], [1, 1, 32, 192]
    xt = _to_device(device, _make_input(shape, ttnn.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    golden = ttnn.to_torch(_force_native(xt, out_shape))
    # Native compiled its program; auto taking codegen would compile a new one.
    entries = device.num_program_cache_entries()
    assert_equal(golden, ttnn.to_torch(ttnn.reshape(xt, out_shape)))
    assert device.num_program_cache_entries() == entries, "auto did not run native"


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32768}], indirect=True)
def test_reshape_routes_tile_rows_case_under_trace(device):
    shape, out_shape, dtype = [1, 64, 1024], [1, 64, 8, 128], ttnn.bfloat16
    xt = _to_device(device, _make_input(shape, dtype), dtype, ttnn.TILE_LAYOUT)
    _force_codegen(xt, out_shape)
    entries = device.num_program_cache_entries()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = ttnn.reshape(xt, out_shape)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    # Auto must reuse the codegen program: trace capture cannot compile a new one.
    assert device.num_program_cache_entries() == entries
    fresh = _make_input(shape, dtype, seed=1)
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(fresh, dtype=dtype, layout=ttnn.TILE_LAYOUT), xt)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
    assert_equal(fresh.reshape(out_shape), ttnn.to_torch(out))


@pytest.mark.parametrize("placement", [ttnn.L1_MEMORY_CONFIG, _DRAM], ids=["l1", "dram"])
def test_reshape_codegen_rm_program_key_is_frontier_invariant(device, placement):
    shape, out_shape, dtype = [12, 100], [12, 10, 10], ttnn.bfloat16
    xt = _to_device(device, _make_input(shape, dtype), dtype, ttnn.ROW_MAJOR_LAYOUT, placement)
    golden = ttnn.to_torch(_force_native(xt, out_shape))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, out_shape)))
    entries_before = device.num_program_cache_entries()
    # Trace capture compiles nothing it has not already cached, so a call whose key moves with the L1 frontier
    # cannot be traced. This tensor moves the frontier by many pages per bank.
    resident = ttnn.from_torch(
        torch.zeros([1, 1, 256, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, out_shape)))
        msg = "the program key moved with the L1 frontier, so the op cannot be traced"
        assert device.num_program_cache_entries() == entries_before, msg
    finally:
        ttnn.deallocate(resident)
