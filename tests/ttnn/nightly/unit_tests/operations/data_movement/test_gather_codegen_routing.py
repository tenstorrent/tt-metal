# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Routing-fallback coverage: every case the codegen gate rejects must fall back to native, an
# accepted case dispatched twice must stay a program-cache hit and rebind its buffers, and the
# forced-codegen entry must refuse anything out of scope rather than fall back. The generated block
# below is emitted from the port's coverage set; hand-add off-grid regressions beneath it.

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal

# `ttnn.gather` takes no implementation argument -- it routes on its own. The forced golden and
# codegen legs therefore come from the verification-only entries in the private module; see
# gather_force.hpp.
_force_native = ttnn._ttnn.operations.data_movement.gather_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.gather_force_codegen


def _make_input(shape, dtype):
    if dtype in (ttnn.int32, ttnn.uint32):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


_UINT16_MAX = 65535


def _materialize_index(shape, kwargs, layout, device):
    """`index` case specs are shape lists, not tensors — materialize once per case so every leg
    (golden, routed, cache-hit pair) gathers the same elements. The dtype follows the gathered
    axis length: uint16 cannot name a position past 65535."""
    index_shape = kwargs.get("index")
    if not isinstance(index_shape, list):
        return kwargs
    dim = kwargs.get("dim", -1)
    axis = dim if dim >= 0 else len(shape) + dim
    axis_len = int(shape[axis])
    index = torch.randint(0, axis_len, index_shape, dtype=torch.int32)
    index_dtype = ttnn.uint16 if axis_len <= _UINT16_MAX else ttnn.uint32
    out = dict(kwargs)
    out["index"] = ttnn.from_torch(index, dtype=index_dtype, layout=layout, device=device)
    return out


_DTYPES = [ttnn.bfloat16]
_DTYPE_IDS = ["bfloat16"]

# Every case here is ROW_MAJOR, which supported_by_codegen() rejects on layout alone: the codegen
# kernels address purely in tile pages. So this list is a negative suite -- it asserts `auto` falls
# back to native -- and the rejection happens before any shape, geometry or L1 arithmetic is
# consulted. Rank, `dim` and the index dtype boundary are therefore the only axes that discriminate
# anything; a wider or taller row re-proves the same layout check at a cost that is all host-side
# golden and comparison. Positive per-factory coverage lives in the TILE_LAYOUT tests below, which
# size themselves off the device's real L1 budget.
_ROUTING = [
    ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32]}, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 1, 32, 64], {"dim": -2, "index": [1, 1, 16, 64]}, ttnn.ROW_MAJOR_LAYOUT),
    ([1, 32, 64], {"dim": -1, "index": [1, 32, 32]}, ttnn.ROW_MAJOR_LAYOUT),
    ([32, 64], {"dim": -1, "index": [32, 32]}, ttnn.ROW_MAJOR_LAYOUT),
    # Sole case whose gathered axis exceeds 65535, so _materialize_index picks uint32 over uint16.
    ([1, 151936], {"dim": -1, "index": [1, 151936]}, ttnn.ROW_MAJOR_LAYOUT),
]
_ROUTING_IDS = [
    "[1, 1, 32, 64]|dim=-1&index=[1, 1, 32, 32]|row_major",
    "[1, 1, 32, 64]|dim=-2&index=[1, 1, 16, 64]|row_major",
    "[1, 32, 64]|dim=-1&index=[1, 32, 32]|row_major",
    "[32, 64]|dim=-1&index=[32, 32]|row_major",
    "[1, 151936]|dim=-1&index=[1, 151936]|row_major",
]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("shape,kwargs,layout", _ROUTING, ids=_ROUTING_IDS)
def test_gather_codegen_routing(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    kwargs = _materialize_index(shape, kwargs, layout, device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    entries_before = device.num_program_cache_entries()
    out = ttnn.gather(xt, **kwargs)
    assert_equal(golden, ttnn.to_torch(out))
    msg = "auto routed an out-of-scope case to codegen (program cache grew); expected native fallback"
    assert device.num_program_cache_entries() == entries_before, msg


_CACHE_HIT = [
    ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32]}, ttnn.bfloat16, ttnn.TILE_LAYOUT),
]
_CACHE_HIT_IDS = [
    "[1, 1, 32, 64]|dim=-1&index=[1, 1, 32, 32]|bfloat16|tile",
]


@pytest.mark.parametrize("shape,kwargs,dtype,layout", _CACHE_HIT, ids=_CACHE_HIT_IDS)
def test_gather_codegen_program_cache_hit(device, shape, kwargs, dtype, layout):
    x = _make_input(shape, dtype)
    xt = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    kwargs = _materialize_index(shape, kwargs, layout, device)
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


_OUT_OF_SCOPE_CASE = ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32]}, ttnn.ROW_MAJOR_LAYOUT)


def test_forced_codegen_refuses_out_of_scope_case(device, expect_error):
    # The forced leg exists to be compared against native, so it has to fail loudly outside its
    # support scope: if it fell back, every bit-exactness result gathered through it would really be
    # native-vs-native. The codegen kernels address purely in tile pages, so ROW_MAJOR is out.
    shape, kwargs, layout = _OUT_OF_SCOPE_CASE
    xt = ttnn.from_torch(_make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)
    kwargs = _materialize_index(shape, kwargs, layout, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)


# --- Hand-added off-grid regressions ---


def _codegen_vs_native(device, shape, kwargs, *, layout=ttnn.TILE_LAYOUT, **gather_kwargs):
    xt = ttnn.from_torch(_make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)
    kwargs = _materialize_index(shape, kwargs, layout, device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    return golden, _force_codegen(xt, **kwargs, **gather_kwargs)


_TILE_BYTES = 32 * 32 * 2  # bfloat16 tile page, already 64-byte aligned


def _brimful_streaming_case(device, wt_index, ht_tiles=2):
    """A gather whose streaming input CB fills the whole per-core L1 budget in a single block.

    The block count, not the row width, is what puts the CB against the ceiling: a row the budget
    can just hold streams in one block whose depth IS the row, leaving under a page of slack, while
    a wider row splits into two blocks of half the depth and fits under any budget. So the witness
    row is the budget itself, in tile pages, minus the index and output pages that share it — and it
    has to be computed per device, because Wormhole's L1 is 36 pages smaller than Blackhole's and a
    row hardcoded for one is not brimful on the other.

    wt_index sizes the row-buffered plan's max(4, Wt_index)-deep output CB past the same budget, so
    selection reaches the streaming factory rather than the interleaved one.
    """
    # total_bytes_per_bank is the allocator's allocatable size, i.e. already net of the base the
    # static CB region starts at — the same quantity gather_static_l1() derives on the C++ side.
    # gather_usable_l1(), which the factory plans against, is this clamped to the live frontier, so
    # the two agree only on an otherwise-clear device; the tests that pin L1 rely on the difference.
    budget = ttnn.get_memory_view(device, ttnn.BufferType.L1).total_bytes_per_bank
    wt_input = (budget - 2 * _TILE_BYTES) // _TILE_BYTES
    height = 32 * ht_tiles
    return [1, 1, height, 32 * wt_input], {"dim": -1, "index": [1, 1, height, 32 * wt_index]}


def test_gather_codegen_streaming_yields_to_a_live_l1_buffer(device):
    # L1 tensors allocate downward from the top of L1 and static CBs stack upward from the allocator
    # base, so a co-resident L1 tensor is a hard ceiling on the streaming input CB: a depth taken
    # from the architecture's L1 size fails program creation instead of streaming in more blocks.
    # A brimful row leaves under one page of slack, so displacing a single page per bank is enough;
    # this tensor displaces dozens on any grid, well clear of alignment effects.
    shape, kwargs = _brimful_streaming_case(device, wt_index=16)
    resident = ttnn.from_torch(
        torch.zeros([1, 1, 2048, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    try:
        golden, out = _codegen_vs_native(device, shape, kwargs)
        assert_equal(golden, ttnn.to_torch(out))
    finally:
        ttnn.deallocate(resident)


def test_gather_codegen_streaming_honours_an_l1_output(device):
    # The op's own output is allocated before its program is created, so an L1 output_mem_config
    # lowers the same ceiling — with no unrelated tensor in play and nothing to deallocate.
    # Wt_index=256 over two tile-rows puts 512 output pages in L1, at least two per bank on any
    # grid, where one is already more than a brimful row's slack.
    shape, kwargs = _brimful_streaming_case(device, wt_index=256)
    golden, out = _codegen_vs_native(device, shape, kwargs, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert_equal(golden, ttnn.to_torch(out))


def test_gather_codegen_streaming_replans_when_l1_fills_after_the_first_dispatch(device):
    # The two tests above put the L1 buffer down before the first dispatch, so the plan is derived
    # under the lowered ceiling and cached that way. This is the opposite order: the first dispatch
    # caches the deepest plan a clear frontier affords, and the second one arrives with the ceiling
    # now below the baked CB region. Program::validate_circular_buffer_region re-reads
    # lowest_occupied_compute_l1_address on every enqueue, cache hit included, and checks it against
    # the region the cached program already carries — so unless the derived plan is part of the
    # program-cache key, the second dispatch fails program creation on a plan that was legal when it
    # was built. A brimful row leaves under a page of slack, so the displaced tensor only has to move
    # the frontier at all.
    shape, kwargs = _brimful_streaming_case(device, wt_index=16)
    xt = ttnn.from_torch(_make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = _materialize_index(shape, kwargs, ttnn.TILE_LAYOUT, device)
    golden = ttnn.to_torch(_force_native(xt, **kwargs))
    assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
    resident = ttnn.from_torch(
        torch.zeros([1, 1, 2048, 4096], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    entries_before = device.num_program_cache_entries()
    try:
        assert_equal(golden, ttnn.to_torch(_force_codegen(xt, **kwargs)))
        msg = "the lowered ceiling did not re-key the program cache, so the second dispatch reused the deep plan"
        assert device.num_program_cache_entries() > entries_before, msg
    finally:
        ttnn.deallocate(resident)


def test_gather_codegen_tiled_honours_a_partial_core_grid(device):
    # Ht=4 tile-rows and Wt_index=6 select the tiled factory, and restricting the split to 5 cores
    # gives 5 of the 24 output tiles to each: every core after the first starts mid-row, so its
    # w_count-sized pop batches straddle the output CB's ring end. cb_pop_front wraps the read
    # pointer only on an exact ring landing, and the full worker grid cannot produce this split
    # (tiled requires Ht < core count, which bounds work per core below the ring depth).
    #
    # test_gather.py::test_gather_sub_core_grids reaches the same kernels today only because
    # is_demoted() returns false for everything, so `auto` happens to pick codegen there. This one
    # forces codegen, and so keeps covering the sub-grid split once a demotion exists.
    shape, kwargs = ([1, 1, 128, 256], {"dim": -1, "index": [1, 1, 128, 192]})
    sub_core_grids = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (4, 0))])
    golden, out = _codegen_vs_native(device, shape, kwargs, sub_core_grids=sub_core_grids)
    assert_equal(golden, ttnn.to_torch(out))


def test_gather_codegen_strided_split_honours_a_multi_range_core_grid(device):
    # Two ranges, one per core row: the splitter consumes a core set range by range, so its
    # extra-work group is the front of range 0, not whichever core a device-grid sweep reaches
    # second. Ht=100 over 14 cores leaves a remainder of 2, so the two orders disagree, and
    # Wt_index=1 selects the row-buffered factory, whose kernels stride tile-rows from the core
    # ordinal -- an ordinal past the remainder runs one tile-row off the end of the tensor and
    # leaves the row its neighbour gave up unwritten.
    shape, kwargs = ([1, 1, 3200, 64], {"dim": -1, "index": [1, 1, 3200, 32]})
    sub_core_grids = ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (6, 0)), ttnn.CoreRange((0, 1), (6, 1))])
    # Preallocated and poisoned, because the golden's device buffer is freed the moment it is read
    # back and the allocator hands the same DRAM straight to this call: a tile-row the kernels never
    # write would otherwise still read back the golden's own values and match.
    poisoned = ttnn.from_torch(
        torch.full(kwargs["index"], float("nan")), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    golden, out = _codegen_vs_native(device, shape, kwargs, sub_core_grids=sub_core_grids, out=poisoned)
    assert_equal(golden, ttnn.to_torch(out))


def test_gather_codegen_refuses_a_preallocated_output_off_the_created_spec(device, expect_error):
    # A preallocated destination is handed straight back by compute_output_specs, so its page -- not
    # the input's -- sizes every CB and the writer's per-tile transfer, while the readers still emit
    # a 32x32 tile of the input's dtype at a stride derived from that page. Only the spec the op
    # would have created for itself keeps the two in step, so anything else has to be declined at the
    # gate rather than written through.
    shape, kwargs = ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32]})
    xt = ttnn.from_torch(_make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = _materialize_index(shape, kwargs, ttnn.TILE_LAYOUT, device)
    index_shape = [1, 1, 32, 32]
    off_spec_outs = [
        ttnn.from_torch(
            torch.zeros(index_shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        ),
        ttnn.from_torch(
            torch.zeros(index_shape, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        ),
    ]
    for out in off_spec_outs:
        with expect_error(RuntimeError, "does not support"):
            _force_codegen(xt, **kwargs, out=out)


def test_gather_codegen_refuses_a_non_default_tile(device, expect_error):
    # Layout::TILE also covers tiny and transposed tiles. Every reader walks a fixed 2x2 grid of
    # 16x16 faces and the CB descriptors carry no tile override, so the kernels address a 32x32 tile
    # whatever the spec says; the geometry helper also divides both padded widths by the input tile's
    # width, which only yields the right Wt_index when the two tensors carry the same tile.
    shape, kwargs = ([1, 1, 32, 64], {"dim": -1, "index": [1, 1, 32, 32]})
    tiny = ttnn.Tile([16, 32])
    xt = ttnn.from_torch(
        _make_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tiny
    )
    kwargs = _materialize_index(shape, kwargs, ttnn.TILE_LAYOUT, device)
    with expect_error(RuntimeError, "does not support"):
        _force_codegen(xt, **kwargs)
