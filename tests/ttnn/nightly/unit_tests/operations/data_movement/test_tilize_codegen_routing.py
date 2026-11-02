# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. This block is emitted from the port's coverage ledger
# and is regenerated on every port update; hand-add off-grid regressions after its END line.
# Contract (4), for the cache-transition legs: earlier tensors stay alive across later calls, and
# program-cache entries are counted around the op call alone.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.tilize` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see tilize_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.tilize_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.tilize_force_codegen


@pytest.fixture(autouse=True)
def _require_codegen_arch(device):
    # Codegen serves calls only on these architectures; elsewhere every call is native and the
    # codegen legs have nothing to exercise.
    if not (ttnn.device.is_wormhole_b0(device) or ttnn.device.is_blackhole(device)):
        pytest.skip("tilize codegen runs only on Wormhole and Blackhole")


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


def _reference(host, **kwargs):
    # Native's int32 conversion emits float bit patterns with junk low bits, so it is no golden
    # there. Truncation toward zero in float64 (exact for bf16/fp32), +/-Inf and out-of-range
    # values clamp to the int32 limits, NaN saturates by its sign
    # bit (+NaN -> INT32_MAX, -NaN -> INT32_MIN), as the device does.
    dtype = kwargs.get("dtype")
    if dtype is None or host.dtype == torch.int32:
        return None
    name = getattr(dtype, "name", None) or str(dtype)
    if name.rsplit(".", 1)[-1].upper() != "INT32":
        return None
    int32_min, int32_max = -(2**31), 2**31 - 1
    wide = host.to(torch.float64)
    out = torch.trunc(wide).clamp(int32_min, int32_max)
    out = torch.where(torch.isnan(wide), torch.where(torch.signbit(wide), int32_min, int32_max), out)
    return out.to(torch.int32)


def _golden(tensor, **kwargs):
    # Native runs either way, so the program cache holds what a native golden would leave in it.
    native = ttnn.to_torch(_force_native(tensor, **kwargs))
    reference = _reference(ttnn.to_torch(tensor), **kwargs)
    return native if reference is None else reference


def _codegen_addrs(tensors):
    # Every buffer address of one tensor or of a list of them, so a list op is compared per input.
    each = tensors if isinstance(tensors, (list, tuple)) else [tensors]
    return tuple(t.buffer_address() for t in each)


def _codegen_added(device, call):
    # The count is read immediately around the call, so setup, readback and golden computation
    # cannot move it: only the op itself can.
    before = device.num_program_cache_entries()
    out = call()
    return out, device.num_program_cache_entries() - before


def _codegen_assert_distinct_addresses(calls):
    # `calls` holds `(input, kwargs, output)` per call, every tensor still alive; a refused call has
    # no output. Tensors of one call may share an address (an op may return its own input), tensors
    # of different calls may not: a program that kept an earlier address would otherwise read or
    # write the other call's recycled buffer and still pass.
    owner = {}
    for index, (inp, call_kwargs, out) in enumerate(calls):
        tensors = [inp, *(v for v in call_kwargs.values() if hasattr(v, "buffer_address"))]
        if out is not None:
            tensors.append(out)
        for address in {a for t in tensors for a in _codegen_addrs(t)}:
            first = owner.setdefault(address, index)
            assert first == index, (
                f"two live tensors from different calls ({first} and {index}) share a buffer address ({address}); "
                "the hit proves nothing"
            )


def _codegen_fresh(shape, kwargs, dtype, layout, device, placement):
    # New data in a new allocation on every call. A secondary input is built again with it.
    tensor = ttnn.from_torch(
        _make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement
    )
    return tensor, kwargs


def _codegen_draw(earlier, shape, kwargs, dtype, layout, device, placement):
    # `_codegen_fresh` and its golden, drawn again while the golden repeats an earlier call's: a
    # stale binding to that call would read the same values, so the value checks could not tell it
    # from a correct one. When every draw repeats, the last is kept: the checks only lose power.
    for _ in range(4):
        tensor, call_kwargs = _codegen_fresh(shape, kwargs, dtype, layout, device, placement)
        golden = _golden(tensor, **call_kwargs)
        if not any(torch.equal(golden, each) for each in earlier):
            break
    return tensor, call_kwargs, golden


_CACHE_HIT = [
    (
        [1, 1, 32, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": ttnn.int32},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    ([32, 32], {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
]
_CACHE_HIT_IDS = [
    "[1, 1, 32, 32]|dtype=DataType.INT32&memory_config=BufferType.DRAM|bfloat16|row_major",
    "[32, 32]@L1|memory_config=BufferType.DRAM|int32|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_tilize_codegen_auto_routes_to_codegen(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    device.clear_program_cache()
    golden = _golden(xt, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.tilize(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a supported case to native (program cache did not grow); expected codegen"
    assert device.num_program_cache_entries() > entries_before, msg


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_tilize_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = _golden(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    second_golden = _golden(yt, **kwargs)
    assert all(
        a != b for a, b in zip(_codegen_addrs(yt), _codegen_addrs(xt))
    ), "the second input reused the first allocation; the hit proves nothing"
    entries_before_hit = device.num_program_cache_entries()
    out = _force_codegen(yt, **kwargs)
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_before_hit, msg
    assert_equal(second_golden, ttnn.to_torch(out))


_SEQ_FRESH = [
    (
        [1, 1, 32, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": ttnn.int32},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_SEQ_FRESH_IDS = [
    "[1, 1, 32, 32]|dtype=DataType.INT32&memory_config=BufferType.DRAM|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _SEQ_FRESH, ids=_SEQ_FRESH_IDS)
def test_tilize_codegen_cache_hit_on_fresh_buffers(device, shape, kwargs, dtype, layout, placement):
    device.clear_program_cache()
    live = []
    for call in range(3):
        inp, call_kwargs, golden = _codegen_draw([g for *_, g in live], shape, kwargs, dtype, layout, device, placement)
        out, added = _codegen_added(device, lambda: _force_codegen(inp, **call_kwargs))
        live.append((inp, call_kwargs, out, golden))
        if call == 0:
            if added == 0:
                pytest.skip("the first forced-codegen dispatch built no program, so there is nothing to reuse")
        else:
            assert added == 0, f"dispatch {call} on fresh buffers missed the program cache"
        assert_equal(golden, ttnn.to_torch(out))
    _codegen_assert_distinct_addresses([(i, k, o) for i, k, o, _ in live])
    for *_, out, golden in live[:2]:
        assert_equal(golden, ttnn.to_torch(out))


_SEQ_SWITCH = [
    (
        [1, 1, 32, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG, "dtype": ttnn.int32},
        [64, 32],
        {"memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
]
_SEQ_SWITCH_IDS = [
    "[1, 1, 32, 32]|dtype=DataType.INT32&memory_config=BufferType.DRAM|bfloat16|row_major->[64, 32]|memory_config=BufferType.DRAM|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,b_shape,b_kwargs,dtype,layout,placement", _SEQ_SWITCH, ids=_SEQ_SWITCH_IDS)
def test_tilize_codegen_cache_survives_a_program_switch(
    device, shape, kwargs, b_shape, b_kwargs, dtype, layout, placement
):
    device.clear_program_cache()
    a_first, a_first_kwargs = _codegen_fresh(shape, kwargs, dtype, layout, device, placement)
    a_first_golden = _golden(a_first, **a_first_kwargs)
    a_first_out, added = _codegen_added(device, lambda: _force_codegen(a_first, **a_first_kwargs))
    if added == 0:
        pytest.skip("the first forced-codegen dispatch built no program, so there is nothing to reuse")
    assert_equal(a_first_golden, ttnn.to_torch(a_first_out))
    b_input, b_call_kwargs = _codegen_fresh(b_shape, b_kwargs, dtype, layout, device, placement)
    b_golden = _golden(b_input, **b_call_kwargs)
    b_out = _force_codegen(b_input, **b_call_kwargs)
    assert_equal(b_golden, ttnn.to_torch(b_out))
    a_again, a_again_kwargs, a_again_golden = _codegen_draw(
        [a_first_golden], shape, kwargs, dtype, layout, device, placement
    )
    a_again_out, added = _codegen_added(device, lambda: _force_codegen(a_again, **a_again_kwargs))
    assert added == 0, "returning to the first program after another one missed the program cache"
    assert_equal(a_again_golden, ttnn.to_torch(a_again_out))
    _codegen_assert_distinct_addresses(
        [
            (a_first, a_first_kwargs, a_first_out),
            (b_input, b_call_kwargs, b_out),
            (a_again, a_again_kwargs, a_again_out),
        ]
    )
    assert_equal(a_first_golden, ttnn.to_torch(a_first_out))


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---
