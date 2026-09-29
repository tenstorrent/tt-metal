# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Routing-fallback coverage: every case the codegen gate rejects must fall back to native.
# The generated block below is emitted from the op's coverage matrix; hand-add off-grid
# regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.repeat_interleave` takes no implementation argument -- it routes on its own. The
# forced-native golden leg therefore comes from the verification-only entry in the private module;
# see repeat_interleave_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.repeat_interleave_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.repeat_interleave_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.int32]
_DTYPE_IDS = ["bfloat16", "float32", "int32"]

_ROUTING = [
    # within-stick (last W) dim deferred for RM path
    ([1, 1, 32, 64], {"repeats": 2, "dim": 3}, ttnn.ROW_MAJOR_LAYOUT),
    # sub-tile (last two) dims deferred for TILE path
    ([1, 1, 32, 64], {"repeats": 2, "dim": 3}, ttnn.TILE_LAYOUT),
    ([1, 1, 64, 32], {"repeats": 2, "dim": 2}, ttnn.TILE_LAYOUT),
    ([1, 32, 64], {"repeats": 2, "dim": 1}, ttnn.TILE_LAYOUT),
    ([2, 3, 32, 8], {"repeats": 2, "dim": 2}, ttnn.TILE_LAYOUT),
    ([2, 3, 4, 8], {"repeats": 3, "dim": 2}, ttnn.TILE_LAYOUT),
    ([4, 6, 8], {"repeats": 2, "dim": 1}, ttnn.TILE_LAYOUT),
]
_ROUTING_IDS = [
    # within-stick (last W) dim deferred for RM path
    "[1, 1, 32, 64]|dim=3&repeats=2|row_major",
    # sub-tile (last two) dims deferred for TILE path
    "[1, 1, 32, 64]|dim=3&repeats=2|tile",
    "[1, 1, 64, 32]|dim=2&repeats=2|tile",
    "[1, 32, 64]|dim=1&repeats=2|tile",
    "[2, 3, 32, 8]|dim=2&repeats=2|tile",
    "[2, 3, 4, 8]|dim=2&repeats=3|tile",
    "[4, 6, 8]|dim=1&repeats=2|tile",
]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("shape,kwargs,layout", _ROUTING, ids=_ROUTING_IDS)
def test_repeat_interleave_codegen_routing(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    # The golden call warms the native program, so a correct fallback leaves the cache
    # unchanged; only a mis-route to codegen compiles a new program and grows it.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat_interleave(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---


# repeats == 1 is a no-op the native path answers without dispatching anything, and the generated
# matrix above never varies repeats down to 1. Nothing else about these cases is out of scope, so
# the degenerate-repeats clause is the only thing keeping them off codegen.
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_repeat_interleave_codegen_routing_single_repeat(device, layout):
    x = _make_input([2, 3, 32, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, 1, 1))
    # Same cache-growth route assertion as the generated matrix above.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat_interleave(xt, 1, 1)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a no-op repeat to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# The RM factory sizes its CB as whole sticks, and a stick scales with the tensor's width, so a wide
# enough input projects a CB no core's L1 can hold. Such a case is otherwise fully in codegen scope;
# without the capacity gate it routes to codegen and then throws out of circular-buffer allocation
# at program-compile time rather than falling back.
#
# The width has to clear codegen's budget while staying inside native's, so that the fallback this
# test asserts is a path that actually runs: native reaches repeat_interleave through ttnn.concat,
# whose CB page is one whole stick, so too wide a stick throws there instead of falling back.
# 262144 float32 elements is a 1 MiB stick -- one page fits every arch's L1 (1.33 MiB at least),
# while the kernels deadlock below two in-flight slots, putting codegen's smallest possible CB at
# 2 MiB, past L1 everywhere. That leaves the verdict independent of the exact per-core budget, which
# moves with whatever else is allocated.
_L1_OVERFLOW_WIDTH = 262144


def test_repeat_interleave_codegen_routing_wide_rm_exceeds_l1(device):
    x = _make_input([2, _L1_OVERFLOW_WIDTH], ttnn.float32)
    xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    golden = ttnn.to_torch(_force_native(xt, 2, 0))
    # Same cache-growth route assertion as the generated matrix above.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat_interleave(xt, 2, 0)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an L1-overflowing case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


def test_forced_codegen_refuses_a_wide_rm_case_that_exceeds_l1(device, expect_error):
    x = _make_input([2, _L1_OVERFLOW_WIDTH], ttnn.float32)
    xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, 2, 0)


def test_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native-vs-native. A TILE repeat needs a dim above the two the tile subdivides, and dim 3 is
    # not.
    x = _make_input([1, 1, 32, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, 2, 3)


# Hand-added: tile-geometry routing, over both axes a tile varies on. An off-default *shape* changes
# the page count and the page size, which the host-side page map and the program factory's CB both
# take from the 32x32 constants. A transposed 32x32 leaves those two quantities alone -- so no
# page-geometry check can see it -- but the datums inside the page are swizzled, and the codegen
# output spec is derived from the layout alone and so comes back with the flags cleared. Both have
# to reach native. dim 1 is above the two dims the tile subdivides and the dtype/layout are in
# scope, so only the tile can drive the route.
#
# Route only, no value assertion: native is itself unreliable for either tile -- for 16x16 two
# native calls on the same input disagree with each other, and for a transposed 32x32 native and
# codegen both differ from torch. This port owes a refusal, not a matching answer.
_OFF_DEFAULT_TILES = [ttnn.Tile([16, 16]), ttnn.Tile([32, 32], transpose_tile=True)]
_OFF_DEFAULT_TILE_IDS = ["shape_16x16", "transposed_32x32"]


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_repeat_interleave_non_default_tile_routes_to_native(device, tile):
    x = _make_input([1, 2, 32, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    # Primes the cache with the native program, so an unchanged count means native served the call.
    _force_native(xt, 2, 1)
    entries_before = device.num_program_cache_entries()
    ttnn.repeat_interleave(xt, 2, 1)
    msg = "auto routed a non-default tile to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_forced_codegen_refuses_non_default_tile(device, expect_error, tile):
    x = _make_input([1, 2, 32, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, 2, 1)
