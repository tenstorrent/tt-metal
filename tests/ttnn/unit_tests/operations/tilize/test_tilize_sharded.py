# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sharded-placement tests for tilize (Refinement 1, op_design.md §5.2 / lamp L1).

Three things are pinned here, and only the third is visible to a numeric check:

1. **The placement regime that was actually selected**, per side. A sharded cell
   passes its identity oracle whether the local shard is consumed zero-copy or
   silently re-read through a `TensorAccessor` — the accessor reads the same
   bytes. So the mechanism is asserted directly: the CB carries the tensor's
   L1 buffer (`CBDescriptor.has_buffer()`), and the kernel is compiled with
   `P_LOCAL_SHARD`, which is what makes the reader/writer issue no NoC traffic
   on that side.
2. **The blocking a shard pins**: `WT_CHUNK` = the shard width when the input is
   aliased (the RM shard's own geometry allows nothing else), and a bounded,
   coarsest-that-fits chunk of it when only the output is aliased — which is
   what keeps a wide-W crossover's streaming CB constant in W instead of OOMing.
3. Identity end-to-end on every scheme × orientation × API, both crossovers, and
   the interleaved path (non-regression).

DO NOT DELETE.
"""

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize, validate
from ttnn.operations.tilize import tilize_program_descriptor as pd
from ttnn.operations._op_contract import ExcludedCell

_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_L1 = ttnn.BufferType.L1
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def _crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


def _legacy(grid, shard_shape, orientation, scheme):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shard_shape, orientation))


def _nd(grid, shard_shape, orientation=_ROW):
    return ttnn.MemoryConfig(_L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid, orientation))


def _shard_grid(memory_config):
    """The shard grid of either sharding API (a host-built nd config has no
    legacy `shard_spec` until a buffer derives one)."""
    spec = memory_config.shard_spec
    return spec.grid if spec is not None else memory_config.nd_shard_spec.grid


def _skip_if_grid_too_small(device, grid):
    live = device.compute_with_storage_grid_size()
    for core_range in grid.ranges():
        if core_range.end.x > live.x - 1 or core_range.end.y > live.y - 1:
            pytest.skip(f"shard grid {core_range} exceeds the live compute grid")


# ---------------------------------------------------------------------------
# 1 + 2. placement regime and derived blocking (host-side, no kernel launch)
# ---------------------------------------------------------------------------


def _descriptor(device, shape, in_mem_config, out_mem_config, dtype=ttnn.bfloat16):
    """Build the ProgramDescriptor the op would run, without launching it."""
    tt_input = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_mem_config,
    )
    plan = validate(tt_input, out_mem_config, dtype=dtype)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.target), plan.out_dtype, ttnn.TILE_LAYOUT, device, plan.out_memory_config
    )
    return pd.create_program_descriptor(tt_input, tt_output, plan)


def _placements(descriptor):
    """(input, output) placement the KERNELS were compiled with."""
    reader, writer = descriptor.kernels[0], descriptor.kernels[1]
    return reader.compile_time_args[1], writer.compile_time_args[0]


_HEIGHT_4 = dict(grid=_crs(((0, 0), (3, 0))), shard_shape=(128, 64), orientation=_ROW, scheme=_HEIGHT)


@pytest.mark.parametrize(
    "shape, in_cfg, out_cfg, expected",
    [
        pytest.param(
            [1, 1, 512, 64],
            _legacy(**_HEIGHT_4),
            _legacy(**_HEIGHT_4),
            (pd.P_LOCAL_SHARD, pd.P_LOCAL_SHARD),
            id="same_spec_is_zero_copy_on_both_sides",
        ),
        pytest.param(
            [1, 1, 128, 64],
            _legacy(_crs(((0, 0), (3, 0))), (32, 64), _ROW, _HEIGHT),
            ttnn.DRAM_MEMORY_CONFIG,
            (pd.P_LOCAL_SHARD, pd.P_ACCESSOR),
            id="sharded_in_interleaved_out",
        ),
        pytest.param(
            # Wide enough that the shard-pinned read transfer clears
            # MIN_STREAM_READ_BYTES (8 tiles = 512 B); see the Refinement-2 gate
            # below for the narrow counterpart.
            [1, 1, 128, 256],
            ttnn.DRAM_MEMORY_CONFIG,
            _legacy(_crs(((0, 0), (3, 0))), (32, 256), _ROW, _HEIGHT),
            (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
            id="interleaved_in_sharded_out",
        ),
        pytest.param(
            [1, 1, 64, 128],
            ttnn.DRAM_MEMORY_CONFIG,
            ttnn.DRAM_MEMORY_CONFIG,
            (pd.P_ACCESSOR, pd.P_ACCESSOR),
            id="interleaved_both_sides_unchanged",
        ),
        pytest.param(
            [1, 1, 128, 256],
            _legacy(_crs(((0, 0), (3, 0))), (32, 256), _ROW, _HEIGHT),
            _legacy(_crs(((0, 0), (1, 0))), (64, 256), _ROW, _HEIGHT),
            (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
            id="cross_spec_gathers_into_the_local_destination",
        ),
    ],
)
def test_placement_regime_selection(device, shape, in_cfg, out_cfg, expected):
    descriptor = _descriptor(device, shape, in_cfg, out_cfg)
    assert _placements(descriptor) == expected

    # The mechanism, not just the flag: an aliased CB carries the tensor's own L1
    # buffer, which is what makes the pack/unpack land in the shard itself.
    cb_input, cb_output = descriptor.cbs[0], descriptor.cbs[1]
    assert cb_input.has_buffer() == (expected[0] == pd.P_LOCAL_SHARD)
    assert cb_output.has_buffer() == (expected[1] == pd.P_LOCAL_SHARD)


def test_local_shard_launches_only_on_cores_that_hold_data(device):
    """A shard pins the core set: 4 shards -> 4 cores, not the whole grid."""
    descriptor = _descriptor(device, [1, 1, 512, 64], _legacy(**_HEIGHT_4), _legacy(**_HEIGHT_4))
    assert descriptor.kernels[0].core_ranges.num_cores() == 4
    assert descriptor.cbs[0].core_ranges.num_cores() == 4


def test_aliased_input_cb_takes_the_whole_shard_width(device):
    """An aliased RM shard admits exactly one block width: the shard's own.

    A block of `WT_CHUNK` pages must be one `tile_h x (WT_CHUNK*32)` row-major
    region, and inside an RM shard that region is the full shard width.
    """
    descriptor = _descriptor(device, [1, 1, 64, 512], _legacy(_crs(((0, 0), (3, 0))), (64, 128), _ROW, _WIDTH), None)
    assert descriptor.kernels[2].compile_time_args[0] == 128 // 32  # WT_CHUNK == shard_wt


def test_wide_w_crossover_keeps_the_cb_bounded_in_w():
    """A wide HEIGHT shard must not size the streaming CB by W.

    The output shard is aliased (it costs no extra L1), so the input CB is the
    only one in the budget — and `derive_shard_blocking` must chunk the shard
    width down to the same `wt_cap()` the interleaved path uses. Pure host
    derivation, so it can sweep W past what a device would allocate.
    """
    in_tile_bytes = 32 * 32 * 2  # bf16 streaming input
    cap = pd.wt_cap(2, in_tile_bytes, 0)  # output aliased -> 0 bytes
    footprints = set()
    for w in (64, 2048, 16384, 65536, 262144):
        wt_chunk, n_chunks = pd.derive_shard_blocking(w // 32, cap)
        assert wt_chunk * n_chunks == w // 32, "WT_CHUNK must divide the shard width exactly"
        l1 = pd.cb_bytes(2, wt_chunk, in_tile_bytes, 0)
        assert l1 <= pd.CB_L1_BUDGET, f"W={w}: streaming CB {l1} exceeds the budget"
        if w >= 16384:
            footprints.add(l1)
    assert len(footprints) == 1, f"CB footprint still grows with W: {footprints}"


def test_chunked_aliased_output_is_correct(device, monkeypatch):
    """`WT_CHUNK < shard_wt` on an aliased output CB — the chunked pack order.

    Squeezing the L1 budget forces `n_chunks > 1`, which is the only regime where
    the push order into the aliased output CB has to be tile-row-major with the W
    chunk innermost (the shard's own linear tile order). A W-chunk-major walk
    still passes every unchunked case, so this is the case that catches it.
    """
    grid = _crs(((0, 0), (1, 0)))
    _skip_if_grid_too_small(device, grid)
    monkeypatch.setattr(pd, "CB_L1_BUDGET", 4 * 2 * (32 * 32 * 2))  # cap = 4 tiles

    shape = [1, 1, 64, 512]
    out_cfg = _legacy(grid, (32, 512), _ROW, _HEIGHT)
    descriptor = _descriptor(device, shape, ttnn.DRAM_MEMORY_CONFIG, out_cfg)
    wt_chunk = descriptor.kernels[2].compile_time_args[0]
    assert wt_chunk == 4 and descriptor.kernels[0].compile_time_args[6] == 4  # n_chunks

    torch_input = torch.arange(64 * 512).reshape(shape).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    got = ttnn.to_torch(tilize(tt_input, out_cfg))
    assert torch.equal(got.to(torch.float32), torch_input.to(torch.float32))


# ---------------------------------------------------------------------------
# 3. identity end-to-end
# ---------------------------------------------------------------------------

_SHARDED_CASES = [
    pytest.param([1, 1, 512, 64], _legacy(**_HEIGHT_4), "same", id="height_row"),
    pytest.param([1, 1, 64, 512], _legacy(_crs(((0, 0), (3, 0))), (64, 128), _ROW, _WIDTH), "same", id="width_row"),
    pytest.param([1, 1, 128, 128], _legacy(_crs(((0, 0), (1, 1))), (64, 64), _COL, _BLOCK), "same", id="block_col"),
    pytest.param([1, 1, 128, 128], _legacy(_crs(((0, 0), (1, 1))), (64, 64), _ROW, _BLOCK), "same", id="block_row"),
    pytest.param([1, 1, 256, 64], _legacy(_crs(((0, 0), (0, 3))), (64, 64), _COL, _HEIGHT), "same", id="height_col"),
    pytest.param([1, 1, 32, 256], _legacy(_crs(((0, 0), (0, 3))), (32, 64), _COL, _WIDTH), "same", id="width_col"),
    pytest.param([1, 1, 128, 128], _nd(_crs(((0, 0), (1, 1))), (1, 1, 64, 64)), "same", id="nd_rank4"),
    pytest.param([4, 32, 64], _nd(_crs(((0, 0), (1, 0))), (2, 32, 64)), "same", id="nd_rank3_batch_fold"),
]


@pytest.mark.parametrize("shape, shard_cfg, kind", _SHARDED_CASES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_same_spec_identity(device, shape, shard_cfg, kind, dtype):
    """Same shard spec in and out: both CBs alias their shard, no NoC either side."""
    _skip_if_grid_too_small(device, _shard_grid(shard_cfg))
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape).to(torch_dtype)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=shard_cfg
    )
    got = ttnn.to_torch(tilize(tt_input, shard_cfg, dtype=dtype))
    assert torch.equal(got.to(torch.float32), torch_input.to(torch.float32))


_CROSSOVER_CASES = [
    pytest.param(
        [1, 1, 128, 64],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(_crs(((0, 0), (3, 0))), (32, 64), _ROW, _HEIGHT),
        id="dram_to_height",
    ),
    pytest.param(
        [1, 1, 128, 64],
        _legacy(_crs(((0, 0), (3, 0))), (32, 64), _ROW, _HEIGHT),
        ttnn.DRAM_MEMORY_CONFIG,
        id="height_to_dram",
    ),
    pytest.param(
        [1, 1, 64, 256],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(_crs(((0, 0), (3, 0))), (64, 64), _ROW, _WIDTH),
        id="dram_to_width",
    ),
    pytest.param(
        [1, 1, 128, 128],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(_crs(((0, 0), (1, 1))), (64, 64), _COL, _BLOCK),
        id="dram_to_block_col",
    ),
    pytest.param(
        [1, 1, 128, 128],
        _legacy(_crs(((0, 0), (1, 1))), (64, 64), _COL, _BLOCK),
        ttnn.DRAM_MEMORY_CONFIG,
        id="block_col_to_dram",
    ),
    pytest.param(
        [4, 32, 64],
        ttnn.DRAM_MEMORY_CONFIG,
        _nd(_crs(((0, 0), (1, 0))), (2, 32, 64)),
        id="dram_to_nd_rank3",
    ),
    pytest.param(
        [1, 1, 128, 64],
        _legacy(_crs(((0, 0), (3, 0))), (32, 64), _ROW, _HEIGHT),
        ttnn.L1_MEMORY_CONFIG,
        id="height_to_l1_interleaved",
    ),
    pytest.param(
        [1, 1, 128, 64],
        _legacy(_crs(((0, 0), (3, 0))), (32, 64), _ROW, _HEIGHT),
        _legacy(_crs(((0, 0), (1, 0))), (64, 64), _ROW, _HEIGHT),
        id="cross_spec_height_4_to_2",
    ),
]


@pytest.mark.parametrize("shape, in_cfg, out_cfg", _CROSSOVER_CASES)
def test_crossover_identity(device, shape, in_cfg, out_cfg):
    """One side sharded, the other interleaved (and one cross-spec reshard)."""
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    torch_input = torch.randn(shape).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    got = ttnn.to_torch(tilize(tt_input, out_cfg))
    assert torch.equal(got.to(torch.float32), torch_input.to(torch.float32))


def test_sharded_output_cast_identity(device):
    """The cast path still reconfigures correctly when the pack target is a shard."""
    shard_cfg = _legacy(**_HEIGHT_4)
    _skip_if_grid_too_small(device, _shard_grid(shard_cfg))
    torch_input = torch.randn([1, 1, 512, 64]).to(torch.float32)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=shard_cfg
    )
    got = ttnn.to_torch(tilize(tt_input, shard_cfg, dtype=ttnn.bfloat16))
    # fp32 -> bf16 is a real (lossy) pack: one bf16 ulp of representation error,
    # no more — the mantissa truncation the precision baseline recorded.
    expected = torch_input.to(torch.bfloat16).to(torch.float32)
    assert torch.allclose(got.to(torch.float32), expected, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_single_core_sharded_is_refused(device, expect_error):
    """A shard pins the core set, so there is no single-core sharded realization."""
    shard_cfg = _legacy(**_HEIGHT_4)
    _skip_if_grid_too_small(device, _shard_grid(shard_cfg))
    tt_input = ttnn.from_torch(
        torch.zeros([1, 1, 512, 64], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=shard_cfg,
    )
    with expect_error(ExcludedCell, "(?i)use_multicore"):
        tilize(tt_input, shard_cfg, use_multicore=False)


# ===========================================================================
# Refinement 2 — cross-spec reshard (general cross-core L1 gather) and padding
# on top of a sharded placement.
#
# Three mechanisms are pinned, in the same spirit as the block above:
#
# a. A source shard NARROWER than a tensor row makes an RM page one SHARD row,
#    not one stick. The gather has to split each row span across pages —
#    addressing it as `page == row` reads the right *number* of bytes from the
#    wrong places, which is a silent PCC failure, so the geometry is asserted
#    (`src_row_pages`) as well as the identity.
# b. Padding no longer disqualifies the OUTPUT side from zero-copy: the fill is
#    materialized into the input CB, so only the input may not be aliased.
# c. The shard grid may split a ROW dim UNEVENLY — the short last shard gets its
#    own block count instead of forcing the whole call onto the accessor path.
# ===========================================================================

# Reader compile-time slots (see tilize_program_descriptor's reader_ct_args).
_CT_REGIME = 0
_CT_SRC_ROW_PAGES = 14

_BLOCK_2x2 = _legacy(_crs(((0, 0), (1, 1))), (64, 64), _ROW, _BLOCK)
_WIDTH_4 = _legacy(_crs(((0, 0), (3, 0))), (64, 128), _ROW, _WIDTH)
_UNEVEN_H3 = _legacy(_crs(((0, 0), (2, 0))), (64, 64), _ROW, _HEIGHT)  # 160 rows -> 64/64/32


def _reshard_cases():
    """(shape, in_cfg, out_cfg, expected_placements, expected_src_row_pages)."""
    return [
        pytest.param(
            # 128 B source pages: below MIN_STREAM_READ_BYTES, so the gate keeps
            # the paged gather but runs it on the whole grid (measured 1.67x).
            [1, 1, 128, 128],
            _BLOCK_2x2,
            _legacy(_crs(((0, 0), (3, 0))), (32, 128), _ROW, _HEIGHT),
            (pd.P_ACCESSOR, pd.P_ACCESSOR),
            2,
            id="narrow_page_block_source_is_gathered_on_the_grid",
        ),
        pytest.param(
            [1, 1, 64, 512],
            _WIDTH_4,
            _legacy(_crs(((0, 0), (1, 0))), (32, 512), _ROW, _HEIGHT),
            (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
            4,
            id="width_in_height_out",
        ),
        pytest.param(
            [1, 1, 64, 512],
            _WIDTH_4,
            _legacy(_crs(((0, 0), (1, 0))), (64, 256), _ROW, _WIDTH),
            (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
            4,
            id="width_in_wider_width_out",
        ),
        pytest.param(
            [1, 1, 64, 512],
            _legacy(_crs(((0, 0), (1, 0))), (32, 512), _ROW, _HEIGHT),
            _WIDTH_4,
            (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
            1,  # a full-width source shard: one page IS one stick
            id="full_width_source_keeps_the_stick_identity",
        ),
        pytest.param(
            [1, 1, 160, 64],
            _UNEVEN_H3,
            _UNEVEN_H3,
            (pd.P_LOCAL_SHARD, pd.P_LOCAL_SHARD),
            1,
            id="uneven_grid_same_spec_is_still_zero_copy",
        ),
        pytest.param(
            [1, 1, 160, 256],
            ttnn.DRAM_MEMORY_CONFIG,
            _legacy(_crs(((0, 0), (2, 0))), (64, 256), _ROW, _HEIGHT),  # 5 tile-rows / 3 shards
            (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
            1,
            id="uneven_grid_destination_is_local",
        ),
    ]


@pytest.mark.parametrize("shape, in_cfg, out_cfg, expected, src_row_pages", _reshard_cases())
def test_reshard_placement_and_source_page_geometry(device, shape, in_cfg, out_cfg, expected, src_row_pages):
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    descriptor = _descriptor(device, shape, in_cfg, out_cfg)
    assert _placements(descriptor) == expected
    assert descriptor.kernels[0].compile_time_args[_CT_SRC_ROW_PAGES] == src_row_pages
    # No DRAM staging and no cross-core combine: two CBs, no semaphores.
    assert len(descriptor.cbs) == 2 and len(descriptor.semaphores) == 0


def test_interleaved_hot_path_keeps_the_stick_identity(device):
    """The general gather must not leak into the interleaved path: one page per
    stick, and still the batched library reader (R_ALIGNED)."""
    descriptor = _descriptor(device, [1, 1, 64, 128], ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
    reader = descriptor.kernels[0]
    assert reader.compile_time_args[_CT_SRC_ROW_PAGES] == 1
    assert reader.compile_time_args[_CT_REGIME] == pd.R_ALIGNED


@pytest.mark.parametrize("shape, in_cfg, out_cfg, expected, src_row_pages", _reshard_cases())
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_reshard_identity(device, shape, in_cfg, out_cfg, expected, src_row_pages, dtype):
    """A reshard is still a permutation: exact identity, L1 -> L1, no DRAM."""
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(shape).to(torch_dtype)
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    got = ttnn.to_torch(tilize(tt_input, out_cfg, dtype=dtype))
    assert torch.equal(got.to(torch.float32), torch_input.to(torch.float32))


def test_uneven_grid_gives_the_short_shard_its_own_block_count(device):
    """160 rows over 3 shards of 64 is 64/64/32: the last core owns 1 tile-row,
    not 2. A uniform per-core count would over-run the tensor."""
    _skip_if_grid_too_small(device, _shard_grid(_UNEVEN_H3))
    descriptor = _descriptor(device, [1, 1, 160, 64], _UNEVEN_H3, _UNEVEN_H3)
    reader = descriptor.kernels[0]
    blocks = [reader.runtime_args[core.x][core.y][2] for core in (ttnn.CoreCoord(x, 0) for x in range(3))]
    assert blocks == [2, 2, 1]


# --- padding on top of a sharded placement ---------------------------------

_PADDED_SHARDED = [
    pytest.param(
        [1, 1, 50, 256],
        [1, 1, 64, 256],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(_crs(((0, 0), (1, 0))), (32, 256), _ROW, _HEIGHT),
        (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
        id="interleaved_in_padded_sharded_out",
    ),
    pytest.param(
        # Both new paths at once: a 256 B-page BLOCK source gathered page-slice by
        # page-slice, filled, and packed into a resident destination shard.
        [1, 1, 100, 256],
        [1, 1, 128, 256],
        _legacy(_crs(((0, 0), (1, 1))), (50, 128), _ROW, _BLOCK),
        _legacy(_crs(((0, 0), (3, 0))), (32, 256), _ROW, _HEIGHT),
        (pd.P_ACCESSOR, pd.P_LOCAL_SHARD),
        id="narrow_page_source_padded_into_a_local_shard",
    ),
    pytest.param(
        [1, 1, 100, 64],
        [1, 1, 128, 64],
        _legacy(_crs(((0, 0), (1, 0))), (50, 64), _ROW, _HEIGHT),
        ttnn.L1_MEMORY_CONFIG,
        (pd.P_ACCESSOR, pd.P_ACCESSOR),
        id="sharded_in_padded_interleaved_out",
    ),
]


@pytest.mark.parametrize("shape, padded_shape, in_cfg, out_cfg, expected", _PADDED_SHARDED)
def test_padded_sharded_keeps_the_destination_local(device, shape, padded_shape, in_cfg, out_cfg, expected):
    """Padding disqualifies only the INPUT side from zero-copy — the fill is
    materialized into the input CB. Compute still packs whole (padded) tiles, so
    a resident destination shard is written in place, not over the NoC."""
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    tt_input = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_cfg,
    )
    plan = validate(tt_input, out_cfg, dtype=ttnn.bfloat16, output_padded_shape=padded_shape, pad_value=10.2)
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.target), plan.out_dtype, ttnn.TILE_LAYOUT, device, plan.out_memory_config
    )
    descriptor = pd.create_program_descriptor(tt_input, tt_output, plan)
    assert _placements(descriptor) == expected
    assert descriptor.cbs[0].has_buffer() is False  # the fill needs a streaming CB
    assert descriptor.cbs[1].has_buffer() == (expected[1] == pd.P_LOCAL_SHARD)
    assert descriptor.kernels[0].compile_time_args[_CT_REGIME] == pd.R_PAD


@pytest.mark.parametrize("shape, padded_shape, in_cfg, out_cfg, expected", _PADDED_SHARDED)
@pytest.mark.parametrize("pad_value", [0.0, -7.5])
def test_padded_sharded_identity(device, shape, padded_shape, in_cfg, out_cfg, expected, pad_value):
    """Data region identical, pad region EXACTLY the fill, on the sharded paths."""
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    torch_input = torch.randn(shape).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    tt_output = tilize(tt_input, out_cfg, dtype=ttnn.bfloat16, output_padded_shape=padded_shape, pad_value=pad_value)
    got = tt_output.cpu().to_torch_with_padded_shape()
    pads = tuple(j for i in reversed(range(len(shape))) for j in (0, padded_shape[i] - shape[i]))
    expected_padded = torch.nn.functional.pad(torch_input, pads, value=pad_value)
    assert torch.equal(got.to(torch.float32), expected_padded.to(torch.float32))
    # The logical view must not have been promoted to the padded shape.
    assert list(ttnn.to_torch(tt_output).shape) == list(shape)


def test_unaddressable_shard_row_is_refused(device, expect_error):
    """A cross-core gather splits every row span at page boundaries, so the page
    itself has to be 32B-alignable. A shard row of 100 B (W=50 bf16) cannot be —
    refuse it rather than return a silently shifted gather."""
    in_cfg = _legacy(_crs(((0, 0), (1, 1))), (50, 50), _ROW, _BLOCK)
    out_cfg = _legacy(_crs(((0, 0), (3, 0))), (32, 128), _ROW, _HEIGHT)
    for cfg in (in_cfg, out_cfg):
        _skip_if_grid_too_small(device, _shard_grid(cfg))
    tt_input = ttnn.from_torch(
        torch.zeros([1, 1, 100, 100], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=in_cfg,
    )
    with expect_error(NotImplementedError, "(?i)aligned page"):
        tilize(tt_input, out_cfg, output_padded_shape=[1, 1, 128, 128], pad_value=1.0)


# --- the read-transfer gate (Refinement 2, measured) ------------------------

_GATED_CASES = [
    pytest.param(
        [1, 1, 1024, 256],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(_crs(((0, 0), (7, 0))), (1024, 32), _ROW, _WIDTH),
        id="one_tile_wide_destination_shard_64B_reads",  # measured 3.25x
    ),
    pytest.param(
        [1, 1, 512, 64],
        ttnn.DRAM_MEMORY_CONFIG,
        _legacy(_crs(((0, 0), (3, 0))), (128, 64), _ROW, _HEIGHT),
        id="two_tile_wide_destination_shard_128B_reads",  # measured 1.75x
    ),
    pytest.param(
        [1, 1, 1024, 256],
        _legacy(_crs(((0, 0), (3, 0))), (1024, 64), _ROW, _WIDTH),
        _legacy(_crs(((0, 0), (7, 0))), (128, 256), _ROW, _HEIGHT),
        id="128B_source_pages",  # measured 1.67x
    ),
]


@pytest.mark.parametrize("shape, in_cfg, out_cfg", _GATED_CASES)
def test_narrow_read_gate_prefers_the_grid_split(device, shape, in_cfg, out_cfg):
    """Aliasing the destination pins WT_CHUNK to the shard's width, which pins the
    reader's per-row transfer. Below MIN_STREAM_READ_BYTES that costs more than
    the NoC write it saves — measured 1.67x / 1.75x / 3.25x in favour of the
    generic full-grid split — so the gate takes the accessor on both sides and
    lets `derive_blocking()` pick a coarse WT_CHUNK again.

    `xfer_gate=0` is the OFF arm (the pre-Refinement-2 choice), pinned here so the
    counterfactual the bench measures stays reachable.
    """
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    descriptor = _descriptor(device, shape, in_cfg, out_cfg)
    assert _placements(descriptor) == (pd.P_ACCESSOR, pd.P_ACCESSOR)
    assert descriptor.kernels[0].core_ranges.num_cores() > out_cfg.shard_spec.grid.num_cores()

    pd.LEVERS["xfer_gate"] = 0
    try:
        off_arm = _descriptor(device, shape, in_cfg, out_cfg)
    finally:
        pd.LEVERS["xfer_gate"] = 1
    assert _placements(off_arm) == (pd.P_ACCESSOR, pd.P_LOCAL_SHARD)


@pytest.mark.parametrize("shape, in_cfg, out_cfg", _GATED_CASES)
def test_narrow_read_gate_keeps_identity(device, shape, in_cfg, out_cfg):
    """Both arms of the gate are correct — it is a perf choice, not a contract."""
    for cfg in (in_cfg, out_cfg):
        if cfg.is_sharded():
            _skip_if_grid_too_small(device, _shard_grid(cfg))
    torch_input = torch.randn(shape).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    for gate in (1, 0):
        pd.LEVERS["xfer_gate"] = gate
        try:
            got = ttnn.to_torch(tilize(tt_input, out_cfg))
        finally:
            pd.LEVERS["xfer_gate"] = 1
        assert torch.equal(got.to(torch.float32), torch_input.to(torch.float32)), f"xfer_gate={gate}"
