# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Routing coverage for the codegen permute path. Contracts:
#   (1) a perf-demoted case still lands on native under the routed entry;
#   (2) a case outside the codegen support scope falls back to native rather than failing;
#   (3) an accepted case dispatched twice under forced codegen stays a program-cache hit and
#       rebinds its buffers;
#   (4) the forced codegen entry refuses anything outside the support scope instead of falling back.
# The blocks below are generated from the port's coverage data; hand-add off-grid regressions
# beneath them.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.permute` takes no implementation argument -- it routes on its own. The forced-native golden
# leg therefore comes from the verification-only entries in the private module; see
# permute_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.permute_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.permute_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


# is_demoted() is `!is_row_invariant(dims)` behind a rank guard: it reads the permutation and
# nothing else. Shape, dtype and which particular axes move are all invisible to it, so a wider
# tensor or a second permutation of the same rank re-proves one boolean at the cost of a native
# golden and a full comparison. Kept below are the axes that do discriminate -- the rank guard's
# lower bound, and one case per accepted dtype -- plus rank 5, where the fused width-height check
# in supported_by_codegen() has outer dims to fold and so takes a different branch than at rank 2.
_DEMOTED = [
    ([32, 64], {"dims": [1, 0]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 4, 96, 128], {"dims": [3, 2, 1, 0]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 2, 3, 64, 96], {"dims": [2, 1, 4, 3, 0]}, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
    ([2, 3, 4, 32, 64], {"dims": [2, 1, 4, 3, 0]}, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
]
_DEMOTED_IDS = [
    "[32, 64]|dims=[1, 0]|bfloat16|row_major",
    "[1, 4, 96, 128]|dims=[3, 2, 1, 0]|bfloat16|row_major",
    "[1, 2, 3, 64, 96]|dims=[2, 1, 4, 3, 0]|float32|row_major",
    "[2, 3, 4, 32, 64]|dims=[2, 1, 4, 3, 0]|int32|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _DEMOTED, ids=_DEMOTED_IDS)
def test_permute_codegen_demotion(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.permute(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "the routed entry sent a perf-demoted case to codegen (program cache grew); expected native"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([1, 2, 3, 64, 96], {"dims": [1, 2, 0, 3, 4]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
]
_CACHE_HIT_IDS = [
    "[1, 2, 3, 64, 96]|dims=[1, 2, 0, 3, 4]|bfloat16|row_major",
]


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


_ACCEPTED_CASE = ([1, 2, 3, 64, 96], {"dims": [1, 2, 0, 3, 4]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
_SHORTCUT_CASE = ([1, 2, 3, 64, 96], {"dims": [0, 1, 2, 3, 4]}, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)


def test_permute_codegen_forced_dispatches_past_the_frontend_shortcut(device):
    shape, kwargs, dtype, layout = _SHORTCUT_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    # A permutation that moves no bytes is answered from host state, so the routed entry must not
    # dispatch it -- while the forced entry has to, or a comparison drawn through it measures
    # nothing.
    assert_equal(golden, ttnn.to_torch(ttnn.permute(xt, **kwargs)))
    msg = "the routed entry dispatched a call the front-end shortcut answers"
    assert device.num_program_cache_entries() == entries_before, msg
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    msg = "forced codegen was swallowed by the front-end shortcut instead of dispatching"
    assert device.num_program_cache_entries() > entries_before, msg


def test_permute_codegen_unsupported_output_memory_config_falls_back(device):
    shape, kwargs, dtype, layout = _ACCEPTED_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    golden = _force_native(xt, **kwargs)
    sharded = ttnn.create_sharded_memory_config(
        list(golden.shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    # Sweeps only ever run the default memory config, which inherits the input's placement,
    # so nothing else in this file reaches the output side of the call contract.
    out = ttnn.permute(xt, **kwargs, memory_config=sharded)
    assert_equal(ttnn.to_torch(golden), ttnn.to_torch(out))


def test_forced_codegen_refuses_a_sharded_output(device, expect_error):
    shape, kwargs, dtype, layout = _ACCEPTED_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    golden = _force_native(xt, **kwargs)
    sharded = ttnn.create_sharded_memory_config(
        list(golden.shape), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs, memory_config=sharded)


def test_forced_codegen_refuses_a_tiled_input(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native measured against itself. This port is row-major only.
    shape, kwargs, dtype, _ = _ACCEPTED_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


# The row-invariant factory sizes its CB as whole sticks, and a stick scales with the tensor's last
# dim, so a wide enough input projects a CB no core's L1 can hold. Such a case is otherwise fully in
# codegen scope; without the capacity gate it routes to codegen and then throws out of
# circular-buffer allocation at program-compile time rather than falling back.
#
# The width has to clear codegen's budget while staying inside native's, so that the fallback this
# test asserts is a path that actually runs. 65536 float32 elements is a 256 KiB stick. The
# row-invariant writer waits on the previous batch plus the current one before releasing either, so
# its CB is eight slots deep -- 2 MiB, past per-core L1 on every arch (1.33 MiB at least). Native's
# row-invariant factory double-buffers a single stick page, so it needs 512 KiB and fits with margin.
# That leaves the verdict independent of the exact per-core budget, which moves with whatever else is
# allocated.
_L1_OVERFLOW_SHAPE = [2, 3, 65536]
_L1_OVERFLOW_DIMS = [1, 0, 2]


def test_permute_codegen_wide_rm_exceeds_l1_falls_back(device):
    xt = ttnn.from_torch(
        _make_input(_L1_OVERFLOW_SHAPE, ttnn.float32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    golden = ttnn.to_torch(_force_native(xt, _L1_OVERFLOW_DIMS))
    entries_before = device.num_program_cache_entries()
    out = ttnn.permute(xt, _L1_OVERFLOW_DIMS)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "the routed entry sent an L1-overflowing case to codegen (program cache grew); expected native"
    assert device.num_program_cache_entries() == entries_before, msg


def test_forced_codegen_refuses_a_wide_rm_case_that_exceeds_l1(device, expect_error):
    xt = ttnn.from_torch(
        _make_input(_L1_OVERFLOW_SHAPE, ttnn.float32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, _L1_OVERFLOW_DIMS)


def test_permute_force_entries_are_not_public_ttnn_operations():
    # ttnn.bind_function tags a callable for auto_register_ttnn_cpp_operations, which walks
    # ttnn._ttnn and republishes anything carrying the tag into the ttnn.* namespace. These are
    # bound with a plain def so they stay out of the public API.
    assert not hasattr(ttnn, "permute_force_native")
    assert not hasattr(ttnn, "permute_force_codegen")


def test_permute_takes_no_implementation_argument(device, expect_error):
    # Which implementation serves a call is an internal decision; the public entry exposes no lever
    # for it. The forced entries above are the only way to pin one.
    shape, kwargs, dtype, layout = _ACCEPTED_CASE
    xt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.permute(xt, **kwargs, implementation="codegen")


# A repeated axis is not a permutation, but it survives ttnn.permute's per-axis normalization, and
# it is not inert for these kernels: [1, 1, 2] leaves the last axis in place, so it selects the
# row-invariant factory, whose output extents come from the permuted shape while its row count comes
# from the input. For an input whose leading dim exceeds its second, the kernels are then asked to
# write more rows than the output tensor holds.
#
# Only the forced leg is exercised. The routed entry declines the case and hands it to native, which
# has no validation of its own for a malformed permutation -- calling it here to observe the fallback
# would dispatch native's kernels for the same ill-defined request.
@pytest.mark.parametrize("dims", [[1, 1, 2], [0, 0, 2]])
def test_forced_codegen_refuses_a_repeated_axis(device, expect_error, dims):
    xt = ttnn.from_torch(
        _make_input([5, 2, 64], ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, dims)


# The row-invariant CB holds a whole stick per slot, budgeted against the L1 that is occupied at the
# moment the gate runs. A requested L1 output is allocated only after that and lowers the very
# frontier the budget measured, so the gate declines the placement instead of admitting a stick it
# cannot account for. The routed entry falls back to native, which serves the same call.
def test_forced_codegen_refuses_an_l1_output(device, expect_error):
    xt = ttnn.from_torch(
        _make_input([3, 64, 96], ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, [1, 0, 2], memory_config=ttnn.L1_MEMORY_CONFIG)


def test_routed_l1_output_falls_back_to_native(device):
    torch_input = _make_input([3, 64, 96], ttnn.bfloat16)
    xt = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.permute(xt, [1, 0, 2], memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert_equal(ttnn.to_torch(out), torch_input.permute(1, 0, 2))
