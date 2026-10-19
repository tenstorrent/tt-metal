# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. This block is emitted from the port's coverage ledger
# and is regenerated on every port update; hand-add off-grid regressions after its END line.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.permute` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see permute_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.permute_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.permute_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_CACHE_HIT = [
    ([64, 96], {"dims": [1, 0]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_CACHE_HIT_IDS = [
    "[64, 96]|dims=[1, 0]|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_permute_codegen_auto_routes_to_codegen(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.permute(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a supported case to native (program cache did not grow); expected codegen"
    assert device.num_program_cache_entries() > entries_before, msg


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_permute_codegen_program_cache_hit(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    second_golden = ttnn.to_torch(_force_native(yt, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


_CALL_CONTRACT_CASE = ([64, 96], {"dims": [1, 0]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
_SHORTCUT_CASE = ([1, 2, 3, 64, 96], {"dims": [0, 1, 2, 3, 4]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)


def test_permute_codegen_forced_dispatches_past_the_frontend_shortcut(device):
    shape, kwargs, dtype, layout = _SHORTCUT_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    # Both cache-count assertions below need a cache holding only the native program.
    device.clear_program_cache()
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    assert_equal(golden, ttnn.to_torch(ttnn.permute(xt, **kwargs)))
    msg = "the routed entry dispatched a call the front-end shortcut answers"
    assert device.num_program_cache_entries() == entries_before, msg
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    msg = "forced codegen was swallowed by the front-end shortcut instead of dispatching"
    assert device.num_program_cache_entries() > entries_before, msg


def test_permute_codegen_unsupported_output_memory_config_falls_back(device):
    shape, kwargs, dtype, layout = _CALL_CONTRACT_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    golden = _force_native(xt, **kwargs)
    sharded = ttnn.create_sharded_memory_config(
        list(golden.shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    # Sweeps only ever run the default memory config, which inherits the input's placement,
    # so nothing else in this file reaches the output side of the call contract.
    out = ttnn.permute(xt, **kwargs, memory_config=sharded)
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(out))


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---


def _brimful_row_invariant_case(device):
    """A row-invariant permute whose deepest CB plan fills the per-core L1 budget.

    The CB holds 2 * batch slots of one stick, so a stick of an eighth of the budget fits the
    deepest batch with about a page of slack, and any L1 tensor lowers the plan to a shallower
    batch. total_bytes_per_bank is already net of the allocator base the static CB region starts at.
    """
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    stick_bytes = (budget // 8 - 128) // 128 * 128
    return [3, 4, stick_bytes // 2], {"dims": [1, 0, 2]}


def _resident_l1_tensor(device):
    return ttnn.from_torch(
        torch.zeros([1, 1, 2048, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def test_permute_codegen_row_invariant_plan_yields_to_a_live_l1_buffer(device):
    # L1 tensors allocate downward from the top of L1 and static CBs stack upward from the allocator
    # base, so a co-resident L1 tensor is a hard ceiling on the CB: a batch taken from the static
    # window fails program creation instead of building a shallower plan.
    shape, kwargs = _brimful_row_invariant_case(device)
    xt = ttnn.from_torch(
        _make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    resident = _resident_l1_tensor(device)
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    finally:
        ttnn.deallocate(resident)


def test_permute_codegen_row_invariant_replans_when_l1_fills_after_the_first_dispatch(device):
    # Program::validate_circular_buffer_region re-reads the lowest occupied L1 address on every
    # enqueue, cache hit included. The batch derived from that frontier is a cache-key field
    # (manifest: frontier_derived), so the second dispatch misses and rebuilds a shallower plan
    # instead of reusing one that was legal only on a clear frontier; the same field is why the
    # op cannot be captured into a trace.
    shape, kwargs = _brimful_row_invariant_case(device)
    xt = ttnn.from_torch(
        _make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    resident = _resident_l1_tensor(device)
    entries_before = device.num_program_cache_entries()
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
        msg = "the lowered frontier did not re-key the program cache, so the second dispatch reused the deep plan"
        assert device.num_program_cache_entries() > entries_before, msg
    finally:
        ttnn.deallocate(resident)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_permute_codegen_row_invariant_traces_when_l1_is_allocated_before_warmup(device):
    # The batch is a cache-key field, so capture only finds its program when warm-up saw the same
    # frontier: L1 that lives through capture has to be allocated before the warm-up call.
    shape, kwargs = _brimful_row_invariant_case(device)
    xt = ttnn.from_torch(
        _make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    resident = _resident_l1_tensor(device)
    trace_id = None
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
        entries_before = device.num_program_cache_entries()
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        traced = _force_codegen(xt, **kwargs)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        assert device.num_program_cache_entries() == entries_before, "capture compiled a program warm-up had not cached"
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert_equal(golden, ttnn.to_torch(traced))
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        ttnn.deallocate(resident)
