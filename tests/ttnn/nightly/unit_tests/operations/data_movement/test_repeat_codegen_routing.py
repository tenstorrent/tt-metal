# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# --- BEGIN GENERATED (agentic_port phase 8) ---
# Contracts: (1) every case the codegen gate rejects falls back to native; (2) an accepted case
# dispatched twice on the codegen path stays a program-cache hit and rebinds its buffers; (3) a
# perf-demoted case still lands on native. The generated block below is emitted from the port's
# coverage ledger; hand-add off-grid regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.repeat` takes no implementation argument -- it routes on its own. The forced legs below are
# the verification-only entries in the private module; see repeat_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.repeat_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.repeat_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_DEMOTED = [
    (
        [1, 2, 128, 64],
        {
            "repeat_dims": ttnn.Shape([1, 1, 1, 2]),
            "memory_config": ttnn.create_sharded_memory_config(
                shape=(1, 2, 128, 128),
                core_grid=ttnn.CoreGrid(x=4, y=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=False,
            ),
        },
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 256, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 256, 128),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.L1_MEMORY_CONFIG},
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
]
_DEMOTED_IDS = [
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input+output|repeat_dims=[1, 1, 1, 2]|float32|row_major",
    "[1, 2, 256, 128]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|float32|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|bfloat16|row_major",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.L1&repeat_dims=[2, 1, 1, 1]|float32|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _DEMOTED, ids=_DEMOTED_IDS)
def test_repeat_codegen_demotion(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a perf-demoted case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([1, 1, 1, 1], {"repeat_dims": ttnn.Shape([1, 2, 1, 1])}, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    (
        [1, 1, 1, 1],
        {"repeat_dims": ttnn.Shape([1, 3, 10, 20])},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    (
        [1, 1, 256, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 2, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 1, 256, 64),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 1, 256, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 2, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 1, 256, 64),
            core_grid=ttnn.CoreGrid(x=8, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 128),
            core_grid=ttnn.CoreGrid(x=2, y=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 128),
            core_grid=ttnn.CoreGrid(x=2, y=2),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 1, 2]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 128, 64],
        {"repeat_dims": ttnn.Shape([1, 1, 1, 2]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 128, 64),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 1, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
    (
        [1, 2, 64, 128],
        {"repeat_dims": ttnn.Shape([2, 2, 1, 1]), "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_sharded_memory_config(
            shape=(1, 2, 64, 128),
            core_grid=ttnn.CoreGrid(x=4, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        ),
    ),
]
_CACHE_HIT_IDS = [
    "[1, 1, 1, 1]|repeat_dims=[1, 2, 1, 1]|bfloat16|tile",
    "[1, 1, 1, 1]|repeat_dims=[1, 3, 10, 20]|bfloat16|row_major",
    "[1, 1, 256, 64]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 2, 1]|bfloat16|row_major",
    "[1, 1, 256, 64]@HEIGHT/8x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 2, 1]|bfloat16|tile",
    "[1, 2, 128, 128]@BLOCK/2x2/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|row_major",
    "[1, 2, 128, 128]@BLOCK/2x2/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|tile",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 1, 2]|bfloat16|row_major",
    "[1, 2, 128, 64]@HEIGHT/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[1, 1, 1, 2]|bfloat16|tile",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 1, 1, 1]|bfloat16|tile",
    "[1, 2, 64, 128]@WIDTH/4x1/ROW_MAJOR/input|memory_config=BufferType.DRAM&repeat_dims=[2, 2, 1, 1]|bfloat16|row_major",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout,placement", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_repeat_codegen_program_cache_hit(device, shape, kwargs, dtype, layout, placement):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    entries_after_miss = device.num_program_cache_entries()
    # Same spec, a distinct allocation: the cached program must rebind its Buffer*s
    # instead of reusing the first dispatch's addresses.
    yt = ttnn.from_torch(_make_input(shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=placement)
    second_golden = ttnn.to_torch(_force_native(yt, **kwargs))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(yt, **kwargs)))
    msg = "second forced-codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


# --- END GENERATED ---


# --- Off-grid regressions (hand-added; edit here, not the emitter) ---


def _auto_route_grows_cache(device, xt, repeat_dims, **kwargs):
    """Runs ttnn.repeat after priming the native program on an empty cache; returns (out, grew).

    Clearing first makes "grew" mean "auto compiled a program native does not use" regardless of what
    earlier tests left cached, so a codegen route is observable, not only a native fallback.
    """
    device.clear_program_cache()
    _force_native(xt, repeat_dims, **kwargs)
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, repeat_dims, **kwargs)
    return out, device.num_program_cache_entries() > entries_before


# Sub-tile TILE H/W repeat. Main's port refused these (tile-page copies would fold pad rows into the
# logical result) and asserted a native fallback over this whole grid. The update runs such a repeat
# row-major between one untilize and one retilize (needs_row_major_round_trip), so the grid now
# routes to codegen; what it owes is a bit-exact answer. The golden is torch, not native: a repeat
# is a pure copy, and native is not an oracle for the case it used to be the only path for.
_SUB_TILE_SHAPES = [
    [1, 1, 1, 1],
    [1, 2, 10, 20],
    [1, 2, 12, 24],
    [1, 2, 14, 28],
    [1, 2, 16, 32],
    [1, 2, 18, 36],
    [1, 2, 20, 40],
    [1, 2, 22, 44],
    [1, 2, 4, 4],
    [1, 2, 4, 8],
    [1, 2, 6, 12],
    [1, 2, 8, 16],
]
_SUB_TILE_REPEATS = [
    [1, 3, 10, 20],
    [1, 3, 12, 24],
    [1, 3, 14, 28],
    [1, 3, 16, 32],
    [1, 3, 18, 36],
    [1, 3, 20, 40],
    [1, 3, 22, 44],
    [1, 3, 4, 8],
    [1, 3, 6, 12],
    [1, 3, 8, 16],
    [2, 2, 2, 2],
]


@pytest.mark.parametrize("repeat_dims", _SUB_TILE_REPEATS, ids=lambda r: f"repeat_dims={r}")
@pytest.mark.parametrize("shape", _SUB_TILE_SHAPES, ids=lambda s: f"{s}")
def test_repeat_sub_tile_tile_routes_to_codegen(device, shape, repeat_dims):
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out, grew = _auto_route_grows_cache(device, xt, ttnn.Shape(repeat_dims))
    assert_equal(x.repeat(*repeat_dims), ttnn.to_torch(out))
    assert grew, "auto served a sub-tile TILE H/W repeat on native; expected the codegen round trip"


# Mixed placement: interleaved DRAM input, interleaved L1 output requested via memory_config. Main's
# port refused it because its RM factories sized CB slots and transfers from one side's aligned page,
# and DRAM/L1 alignments differ, so a cross-placement call would overrun destination pages or CB
# slots. The update sizes every row-major slot as the larger of the two aligned pages
# (rm_slot_bytes) and dropped the placement gate, so all three cases now route to codegen; the
# overrun this guarded against would show up as a wrong answer or a mis-placed output.
_MIXED_PLACEMENT = [
    # higher-dim RM: a writer paced by the wider DRAM pitch would overrun narrower L1 pages
    ([1, 2, 10, 20], [1, 3, 1, 1], ttnn.ROW_MAJOR_LAYOUT),
    # last-dim RM: a reader paced by the wider DRAM pitch would overrun out-pitched CB slots
    ([1, 2, 10, 20], [1, 1, 1, 3], ttnn.ROW_MAJOR_LAYOUT),
    # TILE: outer-axis tile-page copy, page size placement-agnostic
    ([1, 2, 10, 20], [1, 3, 1, 1], ttnn.TILE_LAYOUT),
]

_MIXED_PLACEMENT_IDS = [
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 1, 1]|row_major|dram_to_l1",
    "[1, 2, 10, 20]|repeat_dims=[1, 1, 1, 3]|row_major|dram_to_l1",
    "[1, 2, 10, 20]|repeat_dims=[1, 3, 1, 1]|tile|dram_to_l1",
]


@pytest.mark.parametrize("shape,repeat_dims,layout", _MIXED_PLACEMENT, ids=_MIXED_PLACEMENT_IDS)
def test_repeat_codegen_routing_mixed_placement(device, shape, repeat_dims, layout):
    l1_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
    out, grew = _auto_route_grows_cache(device, xt, ttnn.Shape(repeat_dims), memory_config=l1_mc)
    assert_equal(x.repeat(*repeat_dims), ttnn.to_torch(out))
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert grew, "auto served a DRAM->L1 repeat on native; expected codegen now that the placement gate is gone"


# A row-major leg pages one stick per CB slot, the slot holds the larger of the leg's input and
# output sticks, and the factory needs at least two slots (plan_rm_cb). A case where even two slots
# exceed the static L1 window is otherwise fully in codegen scope; without the capacity gate it
# routes to codegen and then throws out of circular-buffer allocation instead of falling back.
#
# Main tested this with a higher-dim repeat of a 256 KiB stick, but the updated factory shrinks its
# CB to two slots, so that case now fits (512 KiB) and routes to codegen. A last-dim repeat keeps the
# refusal and stays servable by native: native's last-dim CBs hold the *input* stick (2 x 256 KiB),
# while codegen's slot holds the 3x-wider output stick, so two slots are 1.5 MiB -- the whole of L1
# on the largest arch, hence past the window (L1 less its reserved base) on every arch.
_L1_OVERFLOW_WIDTH = 131072
_L1_OVERFLOW_REPEAT = [1, 1, 1, 3]


def test_repeat_codegen_routing_wide_rm_exceeds_l1(device):
    shape = [1, 2, 2, _L1_OVERFLOW_WIDTH]
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    repeat_dims = ttnn.Shape(_L1_OVERFLOW_REPEAT)
    golden = ttnn.to_torch(_force_native(xt, repeat_dims))
    # Same cache-growth route assertion as the generated matrix above.
    entries_before = device.num_program_cache_entries()
    out = ttnn.repeat(xt, repeat_dims)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an L1-overflowing case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


def test_forced_codegen_refuses_a_wide_rm_case_that_exceeds_l1(device, expect_error):
    x = _make_input([1, 2, 2, _L1_OVERFLOW_WIDTH], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape(_L1_OVERFLOW_REPEAT))


def test_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native-vs-native. Main used a bfloat16 TILE H-dim repeat of H=10; the update serves that through
    # the row-major round trip, so the same sub-tile H repeat is taken over bfloat8_b, which has no
    # row-major leg.
    x = _make_input([1, 2, 10, 20], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape([1, 1, 3, 1]))


# Hand-added: tile-geometry routing, over both axes a tile varies on. An off-default *shape* changes
# the page count and the page size, and the host-side page map feeding the codegen prim derives Ht/Wt
# from the 32x32 constants. A transposed 32x32 leaves those two quantities alone -- so no
# page-geometry check can see it -- but the datums inside the page are swizzled, and the codegen
# output spec is derived from the layout alone and so comes back with the flags cleared. Both have to
# reach native. H is tile-aligned and the dtype/layout are in scope, so only the tile drives the
# route.
#
# Route only, no value assertion: native does not serve either tile correctly -- for 16x16 two native
# calls on the same input disagree with each other, and for a transposed 32x32 native and codegen
# both differ from torch. What this port owes is that it declines the case instead of answering it
# wrongly in its own way. Native's answer is not a reference here.
_OFF_DEFAULT_TILES = [ttnn.Tile([16, 16]), ttnn.Tile([32, 32], transpose_tile=True)]
_OFF_DEFAULT_TILE_IDS = ["shape_16x16", "transposed_32x32"]


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_repeat_non_default_tile_routes_to_native(device, tile):
    shape = [1, 1, 32, 64]
    x = _make_input(shape, ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    repeat_dims = ttnn.Shape([1, 1, 3, 1])
    # Primes the cache with the native program, so an unchanged count means native served the call.
    _force_native(xt, repeat_dims)
    entries_before = device.num_program_cache_entries()
    ttnn.repeat(xt, repeat_dims)
    msg = "auto routed a non-default tile to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


@pytest.mark.parametrize("tile", _OFF_DEFAULT_TILES, ids=_OFF_DEFAULT_TILE_IDS)
def test_forced_codegen_refuses_non_default_tile(device, expect_error, tile):
    x = _make_input([1, 1, 32, 64], ttnn.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, ttnn.Shape([1, 1, 3, 1]))


def _sharded(shape, x, y, strategy):
    return ttnn.create_sharded_memory_config(
        shape=tuple(shape),
        core_grid=ttnn.CoreGrid(x=x, y=y),
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )


_H = ttnn.ShardStrategy.HEIGHT
_W = ttnn.ShardStrategy.WIDTH

# Cases measured faster on codegen than on native (device and wall) that a perf demotion must not catch.
# The small row-major shapes win on device by ~1.8x and hold only while the auto route adds no host cost
# over the forced one, so a demotion widened to cover them would surrender that win.
# Each: (shape, repeat_dims, dtype, layout, input placement, output memory_config or None).
_CODEGEN_WINS = [
    *[
        ([1, 2, h, w], [1, 2, 1, 1], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, None)
        for h, w in [(4, 4), (6, 12), (8, 16), (10, 20), (12, 24), (14, 28), (16, 32), (18, 36), (20, 40), (22, 44)]
    ],
    ([1, 1, 1, 1], [1, 2, 1, 1], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, None),
    *[
        (
            [1, 2, 256, 128],
            [2, 1, 1, 1],
            dtype,
            ttnn.TILE_LAYOUT,
            _sharded([1, 2, 256, 128], 8, 1, _H),
            _sharded([2, 2, 256, 128], 8, 1, _H),
        )
        for dtype in (ttnn.bfloat16, ttnn.float32)
    ],
    (
        [1, 2, 64, 128],
        [2, 2, 1, 1],
        ttnn.float32,
        ttnn.ROW_MAJOR_LAYOUT,
        _sharded([1, 2, 64, 128], 4, 1, _W),
        _sharded([2, 4, 64, 128], 4, 1, _W),
    ),
    (
        [1, 2, 128, 64],
        [1, 1, 1, 2],
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        _sharded([1, 2, 128, 64], 4, 1, _H),
        _sharded([1, 2, 128, 128], 4, 1, _H),
    ),
]


def _codegen_win_id(case):
    shape, repeat_dims, dtype, layout, placement, out_mc = case
    where = "dram" if placement == ttnn.DRAM_MEMORY_CONFIG else str(placement.memory_layout).split(".")[-1]
    return f"{shape}x{repeat_dims}-{dtype}-{layout}-{where}"


@pytest.mark.parametrize(
    "shape,repeat_dims,dtype,layout,placement,out_mc", _CODEGEN_WINS, ids=[_codegen_win_id(c) for c in _CODEGEN_WINS]
)
def test_repeat_codegen_win_routes_to_codegen(device, shape, repeat_dims, dtype, layout, placement, out_mc):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=placement)
    kwargs = {} if out_mc is None else {"memory_config": out_mc}
    out, grew = _auto_route_grows_cache(device, xt, ttnn.Shape(repeat_dims), **kwargs)
    assert_equal(x.repeat(repeat_dims).to(ttnn.to_torch(out).dtype), ttnn.to_torch(out))
    assert grew, "auto routed a codegen-winning case to native (program cache did not grow)"
