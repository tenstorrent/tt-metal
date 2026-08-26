# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Routing-fallback coverage: every case the codegen gate rejects must fall back to native.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.pad` takes no implementation argument -- it routes on its own. The forced-native golden leg
# therefore comes from the verification-only entry in the private module; see pad_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.pad_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.pad_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


# Cases the gate rejects and native can still serve, so the two legs are comparable. Tile front
# padding is excluded on purpose: native refuses it outright, so there is no golden to compare
# against -- test_tile_front_pad_refused_by_both covers it instead.
_ROUTING = [
    ([1, 1, 32, 32], ((0, 0), (0, 0), (0, 32), (0, 32)), ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
    ([1, 1, 64, 64], ((0, 0), (0, 0), (0, 15), (0, 31)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 32, 32], ((0, 0), (0, 0), (0, 25), (0, 6)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 32, 32], ((0, 0), (0, 7), (0, 9)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([32, 32], ((0, 2), (0, 6)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([64, 64], ((0, 31), (0, 15)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 40, 64], ((0, 0), (0, 0), (0, 32), (0, 32)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ([1, 1, 64, 40], ((0, 0), (0, 0), (0, 32), (0, 32)), ttnn.TILE_LAYOUT, ttnn.bfloat16),
]
_ROUTING_IDS = [
    "bf8_b|tile|tile_back",
    "tile|subtile_back_hw",
    "tile|subtile_back_4d",
    "tile|subtile_back_3d",
    "tile|subtile_back_2d",
    "tile|subtile_back_2d_wide",
    "tile|partial_input_h",
    "tile|partial_input_w",
]


@pytest.mark.parametrize("shape,padding,layout,dtype", _ROUTING, ids=_ROUTING_IDS)
def test_pad_codegen_routing(device, shape, padding, layout, dtype):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    golden = ttnn.to_torch(_force_native(xt, padding=padding, value=0))
    # The golden call warms the native program, so a correct fallback leaves the cache unchanged;
    # only a mis-route to codegen compiles a new program and grows it.
    entries_before = device.num_program_cache_entries()
    out = ttnn.pad(xt, padding=padding, value=0)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# A row-major width that is not a multiple of the buffer alignment makes the reader take its
# staging fallback: pull the alignment-padded page into scratch, then RISC-memmove the real bytes
# out. That path was demoted on the theory it lost to native; cross-arch CI measured it winning on
# both archs, so it routes to codegen now. Pinned here because the staging path is only reachable
# through these widths -- a future demotion that took it back out would silently drop its coverage.
@pytest.mark.parametrize("width", [17, 33], ids=["w17", "w33"])
def test_pad_codegen_routing_ragged_width(device, width):
    shape = [1, 1, 32, width]
    padding = ((0, 0), (0, 0), (0, 0), (0, 32))
    x = torch.rand(shape, dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    golden = ttnn.to_torch(_force_native(xt, padding=padding, value=0))
    entries_before = device.num_program_cache_entries()
    out = ttnn.pad(xt, padding=padding, value=0)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto declined a ragged-W case (program cache unchanged); expected the codegen staging path"
    assert device.num_program_cache_entries() > entries_before, msg


def test_pad_codegen_routing_sharded_input(device):
    x = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    sharded = ttnn.interleaved_to_sharded(
        xt,
        ttnn.create_sharded_memory_config(
            [32, 32],
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    padding = ((0, 0), (0, 0), (0, 32), (0, 32))
    golden = ttnn.to_torch(_force_native(sharded, padding=padding, value=0))
    entries_before = device.num_program_cache_entries()
    out = ttnn.pad(sharded, padding=padding, value=0)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed a sharded-input case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


# Every codegen factory places work over the full compute-with-storage grid, so a caller that asked
# for single-core placement or a specific sub-grid must reach native under auto -- silently widening
# the placement would break the resource partitioning the caller asked for. The golden leg runs with
# the same controls, because native builds a different program per placement.
_EXECUTION_CONTROLS = [
    ({"use_multicore": False}, "use_multicore_false"),
    (
        {"sub_core_grids": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})},
        "sub_core_grids",
    ),
]


@pytest.mark.parametrize("controls,control_id", _EXECUTION_CONTROLS, ids=[c[1] for c in _EXECUTION_CONTROLS])
def test_pad_execution_controls_route_to_native(device, controls, control_id):
    x = torch.rand([1, 1, 32, 32], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    padding = ((0, 0), (0, 0), (0, 32), (0, 32))
    golden = ttnn.to_torch(_force_native(xt, padding=padding, value=0, **controls))
    entries_before = device.num_program_cache_entries()
    out = ttnn.pad(xt, padding=padding, value=0, **controls)
    assert_equal(golden, ttnn.to_torch(out))
    msg = f"auto routed {control_id} to codegen (program cache grew); codegen cannot honour it"
    assert device.num_program_cache_entries() == entries_before, msg


# Front padding in tile layout is outside both prims' scope, so the only observable that separates a
# correct route from a mis-route is which error comes back: reaching codegen would either succeed or
# fail with the codegen gate's message.
@pytest.mark.parametrize(
    "padding",
    [((1, 0), (0, 0), (0, 32), (0, 32)), ((0, 0), (1, 0), (0, 32), (0, 32)), ((0, 0), (0, 0), (32, 0), (0, 32))],
    ids=["front_n", "front_c", "front_h"],
)
def test_tile_front_pad_refused_by_both(device, expect_error, padding):
    x = torch.rand([2, 2, 32, 32], dtype=torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "does not support front padding"):
        ttnn.pad(xt, padding=padding, value=0)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, padding=padding, value=0)
