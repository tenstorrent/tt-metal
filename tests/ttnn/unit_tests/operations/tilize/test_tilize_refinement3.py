# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — the interleaved <-> sharded crossover.

Three levers, two shipped and one refuted, plus two refuted *sub*-levers. All of
them are tested, because a refuted lever's gate is exactly as load-bearing as a
shipped one: it is what keeps the next implementer from re-enabling a measured
regression.

* **C14 one-sided CB aliasing** (SHIPPED, both directions). On a crossover the
  sharded side's CB is built with `cb_descriptor_from_sharded_tensor`, so its base
  address IS the shard's, and the work split gives each core exactly its own
  shard's tiles. `alias_out` (interleaved RM -> sharded TILE) has the tilize LLK
  pack straight into the output shard; `alias_in` (sharded RM -> interleaved TILE)
  has the unpacker read the input shard in place. Measured, in-run A/B, 7 rounds x
  10 launches:

      regime            | Phase-0 generic | Refinement 3 | ratio
      ------------------|-----------------|--------------|-------
      g_dram_to_sharded |          19 031 |       15 780 | 1.206x
      g_sharded_to_dram |          19 652 |       15 136 | 1.298x

  The correctness risk this carries is NOT a hang or a CB imbalance — it is a
  silent TRANSPOSE: a wrong shard -> global-tile map keeps every CB count balanced
  and writes the right number of tiles to the wrong places (`op_design.md` Risk #2,
  the class of bug that cost Phase 0 26 reference cells). Hence the geometry matrix
  below is asserted with `torch.equal` on `arange` input, where every element is
  unique and a permutation cannot cancel out.

* **B5/B6 coalesced sharded read** (SHIPPED). A ROW_MAJOR-sharded source stores one
  page per row and one page COLUMN per shard, so a chunk-block's 32 pages are
  contiguous in ONE core's L1 — one read of `32 * page_bytes` instead of 32 reads.
  Measured alone (alias forced off): `g_sharded_to_dram` 19 652 -> 17 342 = 1.133x.
  It is the fallback for every sharded-RM input the alias declines.

* **C7 split reader on the alias path** (REFUTED, +11.6 %). The freed BRISC really
  is idle and the split really does halve the reads each RISC-V issues — but the
  read leg is DRAM-BANK bound (10 199 ns for 2.10 MB = 206 GB/s against a 214 GB/s
  best), so a second issuer buys nothing and its hand-off costs. Refinement 3 still
  generalises C7 to depth >= 2 (BRISC now derives the reserved window instead of
  reading `get_write_ptr`), because that is what made the measurement possible at
  the depth the alias actually wants — and that arithmetic is tested here.

* **B8 trid double-issue on the alias path** (REFUTED, +9.9 %) and **B7' read
  grouping** (REFUTED, 1.020 / 1.116 / 1.245 at G = 2 / 4 / 8) — same mechanism:
  there is no read latency left to hide behind.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd
from ttnn.operations.tilize.tilize_program_descriptor import (
    L1_CB_BUDGET_BYTES,
    WT_CHUNK_MAX,
    build_plan,
    create_program_descriptor,
    read_group_pays,
)

_L1 = ttnn.BufferType.L1
_DRAM_BUF = ttnn.BufferType.DRAM
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _crs(end_x, end_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


def _shard(scheme, grid, shape, orientation=_ROW, buffer=_L1):
    return ttnn.MemoryConfig(scheme, buffer, ttnn.ShardSpec(grid, shape, orientation))


def _nd_shard(grid, shard_shape, orientation=_ROW):
    return ttnn.MemoryConfig(_L1, ttnn.NdShardSpec(ttnn.Shape(list(shard_shape)), grid, orientation))


@contextmanager
def _levers(**kwargs):
    """Force the Mode-C counterfactual switches for one test."""
    saved = {}
    try:
        for key, value in kwargs.items():
            name = f"TILIZE_LEVER_{key.upper()}"
            saved[name] = os.environ.get(name)
            os.environ[name] = str(value)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make(device, shape, in_cfg, dtype=ttnn.bfloat16):
    n = 1
    for d in shape:
        n *= d
    if dtype == ttnn.float32:
        torch_input = (torch.arange(n, dtype=torch.float32) % 65536).reshape(shape)
    else:
        torch_input = ((torch.arange(n, dtype=torch.float32) % 4096).reshape(shape)).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    return torch_input, tt_input


def _plan(device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, use_multicore=True, use_double_buffer=None):
    _, tt_input = _make(device, shape, in_cfg, dtype)
    tt_output = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, out_cfg)
    return build_plan(tt_input, tt_output, device, use_multicore=use_multicore, use_double_buffer=use_double_buffer)


def _exact(
    device, shape, in_cfg, out_cfg, *, dtype=ttnn.bfloat16, use_double_buffer=None, repeats=1, use_multicore=True
):
    """tilize is value-preserving, so the oracle is `torch.equal`, not a tolerance."""
    torch_input, tt_input = _make(device, shape, in_cfg, dtype)
    for _ in range(repeats):
        tt_output = tilize(tt_input, out_cfg, use_multicore=use_multicore, use_double_buffer=use_double_buffer)
        assert tt_output.layout == ttnn.TILE_LAYOUT
        result = ttnn.to_torch(tt_output)
        assert torch.equal(result.float(), torch_input.float()), (
            f"not bit-exact on {shape}: {(result.float() - torch_input.float()).abs().max()} max abs, "
            f"{(result.float() != torch_input.float()).sum()} mismatching elements"
        )


# The crossover geometry matrix. Every legacy-2D scheme x both orientations x both
# directions, plus the two shapes that exercise the width-chunked and the widest
# aliasable shard. `expect` is the path the planner must choose.
#   name                         shape             in                out                 expect
GEOMETRIES = [
    ("height_out", (1, 1, 128, 64), None, (_HEIGHT, _crs(3, 0), (32, 64), _ROW), "alias_out"),
    ("height_in", (1, 1, 128, 64), (_HEIGHT, _crs(3, 0), (32, 64), _ROW), None, "alias_in"),
    ("block_out_8x8", (1, 1, 2048, 512), None, (_BLOCK, _crs(7, 7), (256, 64), _ROW), "alias_out"),
    ("block_in_8x8", (1, 1, 2048, 512), (_BLOCK, _crs(7, 7), (256, 64), _ROW), None, "alias_in"),
    ("width_out", (1, 1, 64, 512), None, (_WIDTH, _crs(3, 0), (64, 128), _ROW), "alias_out"),
    ("width_in", (1, 1, 64, 512), (_WIDTH, _crs(3, 0), (64, 128), _ROW), None, "alias_in"),
    ("block_col_out", (1, 1, 128, 128), None, (_BLOCK, _crs(1, 1), (64, 64), _COL), "alias_out"),
    ("block_col_in", (1, 1, 128, 128), (_BLOCK, _crs(1, 1), (64, 64), _COL), None, "alias_in"),
    ("height_col_out", (1, 1, 256, 64), None, (_HEIGHT, _crs(3, 0), (64, 64), _COL), "alias_out"),
    ("wide_height_out", (1, 1, 128, 2048), None, (_HEIGHT, _crs(3, 0), (32, 2048), _ROW), "alias_out"),
    ("wide_height_in", (1, 1, 128, 2048), (_HEIGHT, _crs(3, 0), (32, 2048), _ROW), None, "alias_in"),
]


def _cfgs(entry):
    _, shape, in_spec, out_spec, expect = entry
    in_cfg = DRAM if in_spec is None else _shard(*in_spec)
    out_cfg = DRAM if out_spec is None else _shard(*out_spec)
    return shape, in_cfg, out_cfg, expect


# ---------------------------------------------------------------------------
# C14 — the shard -> global-tile map (the silent-transpose risk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", GEOMETRIES, ids=[e[0] for e in GEOMETRIES])
def test_one_sided_alias_routes_and_covers_the_tensor(device, entry):
    """The plan must take the aliased path AND tile the tensor exactly.

    "Covers exactly" is the property that makes an aliased CB legal: CB page k is
    shard tile k, so the work unit assigned to a core must be precisely that core's
    shard. Asserted three ways — the core count is the shard grid's, the shards tile
    the tile grid with no overlap and no gap, and every unit's rectangle lies inside
    the tensor.
    """
    shape, in_cfg, out_cfg, expect = _cfgs(entry)
    plan = _plan(device, shape, in_cfg, out_cfg)
    assert plan["path"] == expect
    assert plan["alias_in"] == int(expect == "alias_in")
    assert plan["alias_out"] == int(expect == "alias_out")

    assert plan["shard_tiles"] * plan["ncores"] == plan["total_tiles"], "the shards must tile the tensor exactly"
    assert plan["ncores"] == len(plan["work"]) == plan["n_h"] * plan["n_w"]

    covered = set()
    for unit in plan["work"]:
        rows = range(unit["row_start"], unit["row_start"] + unit["row_count"])
        cols = range(
            unit["chunk_start"] * plan["chunk_wt"],
            (unit["chunk_start"] + unit["chunk_count"]) * plan["chunk_wt"],
        )
        assert unit["row_count"] * unit["chunk_count"] * plan["chunk_wt"] == plan["shard_tiles"]
        for r in rows:
            for c in cols:
                assert r < plan["nt_h"] and c < plan["wt"]
                assert (r, c) not in covered, "two cores own the same output tile"
                covered.add((r, c))
    assert len(covered) == plan["total_tiles"], "the work units do not cover the tensor"


@pytest.mark.parametrize("entry", GEOMETRIES, ids=[e[0] for e in GEOMETRIES])
def test_one_sided_alias_is_bit_exact(device, entry):
    """`arange` input, `torch.equal` output: a wrong shard map cannot cancel out.

    This is the test that would have caught Phase 0's tile-grid bug (26 reference
    cells, whole-tensor mismatch) and is the only thing standing between a
    plausible-looking shard map and a silent transpose.
    """
    shape, in_cfg, out_cfg, _ = _cfgs(entry)
    _exact(device, shape, in_cfg, out_cfg, repeats=2)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_one_sided_alias_is_bit_exact_per_dtype(device, dtype):
    """The aliased CB's page size comes from the tensor's tile size, so the map has
    to hold per dtype as well (fp32 doubles both the page and the shard bytes)."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    _exact(device, shape, DRAM, out_cfg, dtype=dtype)
    _exact(device, shape, out_cfg, DRAM, dtype=dtype)


def test_one_sided_alias_casts_through_the_aliased_cb(device):
    """A cast writes the aliased output CB in the OUTPUT dtype (bf8b here), so the
    page size and the shard bytes must both come from the output tensor."""
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    torch_input, tt_input = _make(device, shape, DRAM)
    tt_output = tilize(tt_input, out_cfg, dtype=ttnn.bfloat8_b)
    assert tt_output.dtype == ttnn.bfloat8_b
    result = ttnn.to_torch(tt_output).float()
    ref = torch_input.float()
    denom = (ref - ref.mean()).pow(2).sum().sqrt() * (result - result.mean()).pow(2).sum().sqrt()
    pcc = ((ref - ref.mean()) * (result - result.mean())).sum() / denom
    assert pcc > 0.99, f"bf8b cast through the aliased CB lost accuracy: PCC={pcc}"


def test_alias_out_zero_copy_is_structural_not_incidental(device):
    """The WRITER must be compiled into its no-NoC branch on `alias_out` (and the
    READER on `alias_in`) — that is what "zero traffic on the sharded side" means at
    the kernel level, and it is a compile-time arg, so it is checkable."""
    shape = (1, 1, 2048, 512)
    shard = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)

    _, tt_in = _make(device, shape, DRAM)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, shard)
    plan = build_plan(tt_in, out, device)
    descriptor = create_program_descriptor(tt_in, out, plan)
    reader, writer, _compute = descriptor.kernels
    assert plan["path"] == "alias_out"
    assert list(writer.compile_time_args)[0] == 1, "writer must take the aliased (no-NoC) branch"
    assert list(reader.compile_time_args)[0] == 0, "reader still reads the interleaved input"

    _, tt_in2 = _make(device, shape, shard)
    out2 = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, DRAM)
    plan2 = build_plan(tt_in2, out2, device)
    reader2, writer2, _ = create_program_descriptor(tt_in2, out2, plan2).kernels
    assert plan2["path"] == "alias_in"
    assert list(reader2.compile_time_args)[0] == 1, "reader must take the aliased (no-NoC) branch"
    assert list(writer2.compile_time_args)[0] == 0, "writer still writes the interleaved output"


def test_alias_program_cache_rebinding(device):
    """Two calls, different output shard addresses, one cache entry, both exact.

    With the CB aliased onto a tensor there is no `Buffer*` runtime arg forcing a
    re-patch, so this is the property that says the second call does not write into
    the first call's shard (verified for Path B in Phase 0; the one-sided paths need
    their own witness because the OTHER side's accessor address is a runtime arg).
    """
    shape = (1, 1, 256, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (128, 64), _ROW)
    torch_input, tt_input = _make(device, shape, DRAM)

    first = tilize(tt_input, out_cfg)
    entries = device.num_program_cache_entries()
    second = tilize(tt_input, out_cfg)
    assert device.num_program_cache_entries() == entries, "second call must hit the cache"
    assert first.buffer_address() != second.buffer_address(), "the two calls must land in different shards"
    for out in (first, second):
        assert torch.equal(ttnn.to_torch(out).float(), torch_input.float())


# ---------------------------------------------------------------------------
# C14 — where the alias DECLINES (every one of these is a correctness boundary)
# ---------------------------------------------------------------------------


ND_GEOMETRIES = [
    ("nd_2d", (1, 1, 128, 128), _crs(1, 1), (1, 1, 64, 64)),
    ("nd_rank3", (4, 32, 64), _crs(1, 0), (2, 32, 64)),
    # The case that looks dangerous: a LEADING dim is split while the row dim is
    # whole, so "row-major over the ND shard grid" and "row-major over the folded
    # rows" could disagree. Measured, it cannot: the allocator normalises this to
    # HEIGHT_SHARDED over the folded rows (see the assert below).
    ("nd_split_batch", (2, 64, 64), _crs(1, 1), (1, 32, 64)),
]


@pytest.mark.parametrize("entry", ND_GEOMETRIES, ids=[e[0] for e in ND_GEOMETRIES])
@pytest.mark.parametrize("direction", ["out", "in"])
def test_nd_requests_normalise_to_the_2d_map_and_alias_correctly(device, entry, direction):
    """An ND *request* whose shard is 2D-representable is NORMALISED to a legacy
    layout at allocation, and the buffer then uses the legacy page mapping — which is
    exactly the map `_one_sided_shard_split` implements. So these DO alias, and the
    thing to verify is that the normalisation really happened and the result is
    bit-exact (probe_024 established both; this pins them).

    The `nd` guard in `_shard_geometry` is therefore about specs that stay ND
    (`test_alias_declines_a_genuinely_nd_spec`), not about the ND *API*.
    """
    _, shape, grid, shard_shape = entry
    nd = _nd_shard(grid, shard_shape)
    in_cfg, out_cfg = (DRAM, nd) if direction == "out" else (nd, DRAM)
    plan = _plan(device, shape, in_cfg, out_cfg)
    sharded_side = out_cfg if direction == "out" else in_cfg
    del sharded_side  # the tensor's config is what matters, checked below
    assert plan["path"] == f"alias_{direction}"
    _exact(device, shape, in_cfg, out_cfg)


def test_nd_normalisation_is_what_makes_that_safe(device):
    """The load-bearing fact behind the test above: the allocator hands back a LEGACY
    2D layout, so the buffer's shard -> core order is `corerange_to_cores` and its
    within-shard page order is row-major (`core_to_host_pages`). If a future build
    stopped normalising, `_shard_geometry`'s `nd` flag catches it and the alias
    declines instead of guessing."""
    nd = _nd_shard(_crs(1, 1), (1, 32, 64))
    tensor = ttnn.allocate_tensor_on_device(ttnn.Shape([2, 64, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, nd)
    config = tensor.memory_config()
    assert config.memory_layout != ttnn.TensorMemoryLayout.ND_SHARDED
    assert config.shard_spec is not None
    assert tpd._shard_geometry(tensor)["nd"] is False


def test_alias_declines_a_genuinely_nd_spec(device):
    """More shards than cores is the ND round-robin the 2D map cannot express (a
    legacy spec is one shard per core). It must fall back, not guess."""
    shape = (1, 1, 512, 64)
    nd = _nd_shard(_crs(1, 0), (1, 1, 32, 64))  # 16 shards on 2 cores
    try:
        plan = _plan(device, shape, DRAM, nd)
    except Exception as exc:  # pragma: no cover - build-dependent
        pytest.skip(f"round-robin ND output not constructible here: {exc}")
    assert plan["path"] == "generic", "a shard map the 2D rules cannot express must decline"
    _exact(device, shape, DRAM, nd)


def test_alias_declines_a_cross_spec_reshard(device):
    """Both sides sharded with different specs: the generic path is already correct
    on both, and this refinement deliberately does not touch it."""
    shape = (1, 1, 128, 64)
    in_cfg = _shard(_HEIGHT, _crs(3, 0), (32, 64), _ROW)
    out_cfg = _shard(_HEIGHT, _crs(1, 0), (64, 64), _ROW)
    plan = _plan(device, shape, in_cfg, out_cfg)
    assert plan["path"] == "generic"
    _exact(device, shape, in_cfg, out_cfg)


def test_alias_declines_single_core(device):
    """`use_multicore=False` keeps meaning EXACTLY one core (Refinement 1). The
    aliased path is inherently one-shard-per-core, so it must not hijack it."""
    shape = (1, 1, 128, 64)
    out_cfg = _shard(_HEIGHT, _crs(3, 0), (32, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg, use_multicore=False)
    assert plan["path"] == "generic"
    assert plan["ncores"] == 1
    _exact(device, shape, DRAM, out_cfg, use_multicore=False)


def test_alias_declines_a_dram_sharded_side(device):
    """A CB can only be aliased onto L1. A DRAM-sharded side keeps the accessor."""
    shape = (1, 1, 128, 64)
    try:
        out_cfg = _shard(_HEIGHT, _crs(3, 0), (32, 64), _ROW, buffer=_DRAM_BUF)
        plan = _plan(device, shape, DRAM, out_cfg)
    except Exception as exc:  # pragma: no cover - not all builds accept DRAM shards
        pytest.skip(f"DRAM-sharded output not constructible on this build: {exc}")
    assert plan["path"] == "generic"


def test_alias_in_declines_a_shard_too_wide_for_one_block(device):
    """`alias_in` reads the RM shard IN PLACE, so a block is 32 whole shard rows and
    `chunk_wt == shard_wt` is forced — there is no chunking available. When that
    block's output CB would blow the L1 budget the alias must decline rather than
    OOM, and the generic path's bounded-CB chunking takes over.

    (`alias_out` has no such limit: its plain side is the INPUT CB, which it chunks —
    see `test_alias_out_chunks_a_wide_shard`.)
    """
    shape = (1, 1, 64, 8192)  # HEIGHT shard 32 x 8192 => shard_wt = 256 tiles
    in_cfg = _shard(_HEIGHT, _crs(1, 0), (32, 8192), _ROW)
    plan = _plan(device, shape, in_cfg, DRAM)
    assert plan["path"] == "generic", "a 256-tile-wide block cannot be held in the CB budget"
    assert plan["cb_bytes_per_core"] <= L1_CB_BUDGET_BYTES
    _exact(device, shape, in_cfg, DRAM)


def test_alias_out_chunks_a_wide_shard(device):
    """A wide aliased OUTPUT keeps its per-core input CB bounded by chunking the
    width — and that flips the reader to tile-row-outer / chunk-inner order, because
    the aliased CB's pages are the shard's tiles in row-major order. Getting that
    order wrong transposes the shard while keeping every CB count balanced."""
    shape = (1, 1, 128, 2048)
    out_cfg = _shard(_HEIGHT, _crs(3, 0), (32, 2048), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["path"] == "alias_out"
    assert plan["chunks_per_core"] > 1 and plan["blocks_row_major"] == 1
    assert plan["chunk_wt"] <= WT_CHUNK_MAX
    assert plan["cb_bytes_per_core"] <= L1_CB_BUDGET_BYTES
    # The levers that also own the block sequence must be off (they are chunk-outer).
    assert plan["split_read"] == 0 and plan["prefetch_blocks"] == 1 and plan["read_group"] == 1
    _exact(device, shape, DRAM, out_cfg)


@pytest.mark.parametrize("width", [64, 256, 1024, 2048])
def test_bounded_cb_on_every_alias_width(device, width):
    """`PROPERTIES["bounded_cb"]` on the crossover: the ALLOCATED CB stays inside the
    budget at every width, because only the plain side is allocated (the aliased side
    is the tensor's own shard)."""
    shape = (1, 1, 128, width)
    out_cfg = _shard(_HEIGHT, _crs(3, 0), (32, width), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["cb_bytes_per_core"] <= L1_CB_BUDGET_BYTES
    assert (
        plan["cb_bytes_per_core"] == plan["depth"] * plan["chunk_wt"] * plan["tile_in"]
    ), "on alias_out only the INPUT CB is allocated; the output CB is the shard"
    assert plan["alias_cb_bytes"] == plan["shard_tiles"] * plan["tile_out"]


def test_alias_costs_less_cb_l1_than_the_generic_path(device):
    """The alias does not buy its speed with L1: on `g_dram_to_sharded` it allocates
    ONE side of the CB pair instead of two (8 192 vs 65 536 B/core)."""
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    aliased = _plan(device, shape, DRAM, out_cfg)
    with _levers(r3=0):
        generic = _plan(device, shape, DRAM, out_cfg)
    assert aliased["path"] == "alias_out" and generic["path"] == "generic"
    assert aliased["cb_bytes_per_core"] < generic["cb_bytes_per_core"]


# ---------------------------------------------------------------------------
# B5/B6 — the coalesced sharded read
# ---------------------------------------------------------------------------


def test_coalesced_read_fires_on_a_sharded_source(device):
    """With the alias declined (forced off here), a sharded RM source folds each
    block's 32 same-shard page reads into ONE transaction."""
    shape = (1, 1, 2048, 512)
    in_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    with _levers(r3=0):
        plan = _plan(device, shape, in_cfg, DRAM)
        assert plan["path"] == "generic"
        assert plan["coalesce_rows"] == 1
        assert plan["chunk_row_bytes"] == plan["source_page_bytes"]
        # The levers that reshape the same row loop must yield to it.
        assert plan["stateful_read"] == 0 and plan["prefetch_blocks"] == 1 and plan["split_read"] == 0
        _exact(device, shape, in_cfg, DRAM)


def test_coalesced_read_is_bit_exact_across_schemes(device):
    """One 32-page transaction per block rests on "a shard's pages are contiguous in
    its owner's L1, row-major". If that were false the block would be assembled from
    the wrong rows — which `arange` + `torch.equal` catches immediately."""
    cases = [
        ((1, 1, 512, 64), _shard(_HEIGHT, _crs(3, 0), (128, 64), _ROW)),
        ((1, 1, 128, 128), _shard(_BLOCK, _crs(1, 1), (64, 64), _COL)),
        ((1, 1, 64, 512), _shard(_WIDTH, _crs(3, 0), (64, 128), _ROW)),
        ((1, 1, 2048, 512), _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)),
    ]
    with _levers(r3=0):
        for shape, in_cfg in cases:
            _exact(device, shape, in_cfg, DRAM)


def test_coalesce_declines_when_a_chunk_is_narrower_than_a_page(device):
    """The fold is only legal when the chunk covers a WHOLE source page: otherwise
    the 32 pages it would merge are not contiguous with the L1 destination stride."""
    shape = (1, 1, 2048, 512)
    in_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)  # 128 B pages
    with _levers(r3=0), _pin_chunk(1):  # force chunk_wt = 1 => 64 B chunk, 128 B page
        plan = _plan(device, shape, in_cfg, DRAM)
        assert plan["chunk_row_bytes"] < plan["source_page_bytes"]
        assert plan["coalesce_rows"] == 0
        _exact(device, shape, in_cfg, DRAM)


@contextmanager
def _pin_chunk(chunk):
    tpd.CHUNK_CAP_OVERRIDE = chunk
    try:
        yield
    finally:
        tpd.CHUNK_CAP_OVERRIDE = None


def test_coalesce_declines_on_an_interleaved_source(device):
    """Consecutive pages of an interleaved tensor live in DIFFERENT banks, so there
    is nothing to coalesce (this is the R1c finding, now a structural gate)."""
    plan = _plan(device, (1, 1, 2048, 512), DRAM, DRAM)
    assert plan["coalesce_rows"] == 0


# ---------------------------------------------------------------------------
# The refuted levers: gates pinned to their counterfactual numbers
# ---------------------------------------------------------------------------


def test_c7_is_gated_off_on_the_alias_path(device):
    """C7 on `alias_out`, measured (in-run A/B, 7 rounds x 10 launches):

        depth | no C7  | with C7 | ratio
        ------|--------|---------|-------
          2   | 15 866 |  17 699 | 1.116
          1   | 16 480 |  18 113 | 1.099

    The freed BRISC is genuinely idle, so this is not an implementation failure — it
    is the read leg being DRAM-BANK bound (206 GB/s of a 214 GB/s best), so a second
    issuer has nothing to win and its per-block hand-off costs. Re-enabling it needs
    a new sweep, not a hunch.
    """
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["path"] == "alias_out"
    assert plan["split_read"] == 0, "C7 is measured +11.6 % here"
    assert plan["stateful_read"] == 1, "B13 is what ships on this path (-5.5 %)"


@pytest.mark.parametrize("depth", [False, True], ids=["depth1", "depth2"])
def test_forced_c7_is_bit_exact_at_either_depth(device, depth):
    """The refutation must be a PERF verdict on a working implementation, so the
    forced lever is asserted bit-exact — and at depth 2 that specifically tests
    Refinement 3's window arithmetic (BRISC derives `cb_base + (block % depth) *
    window_bytes` instead of reading `get_write_ptr`, which is only the reserved
    window at depth 1). A wrong window here corrupts data without hanging."""
    shape = (1, 1, 512, 128)
    out_cfg = _shard(_BLOCK, _crs(1, 1), (256, 64), _ROW)
    with _levers(c7=2, b13=0, b8=0):
        plan = _plan(device, shape, DRAM, out_cfg, use_double_buffer=depth)
        assert plan["path"] == "alias_out" and plan["split_read"] == 1
        assert plan["depth"] == (2 if depth else 1)
        _exact(device, shape, DRAM, out_cfg, use_double_buffer=depth, repeats=2)


def test_b8_is_gated_off_on_the_alias_path(device):
    """B8's own size clause (<= 128 B reads) would fire on every crossover shard
    narrower than 8 tiles, but measured on `alias_out` it is **17 440 vs 15 866
    (+9.9 %)** — same mechanism as C7: no read latency left to hide."""
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["blocks_per_core"] >= 2 and plan["chunk_row_bytes"] <= 128, "B8's clauses DO hold here"
    assert plan["prefetch_blocks"] == 1, "...and it is still gated off, by measurement"


def test_forced_b8_is_bit_exact_on_the_alias_path(device):
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    with _levers(b8=2, b13=0, c7=0):
        plan = _plan(device, shape, DRAM, out_cfg)
        assert plan["path"] == "alias_out" and plan["prefetch_blocks"] == 2
        assert plan["depth"] == tpd.PREFETCH_DEPTH
        _exact(device, shape, DRAM, out_cfg)


def test_read_group_gate_is_identity_false(device):
    """Lever B7' (one barrier per GROUP of blocks), measured on `alias_out`:

        group | depth | ns     | vs the shipped 15 866
        ------|-------|--------|----------------------
          1   |   2   | 15 866 | 1.000
          2   |   3   | 16 187 | 1.020
          4   |   5   | 17 706 | 1.116
          8   |   9   | 19 750 | 1.245

    Monotone in G, which is the signature of the real mechanism: grouping delays the
    first push by G blocks and serializes compute behind the reads, while the barrier
    drain it was meant to hide does not exist (the read leg is already at the DRAM
    read rate). Lowering this to 2 costs 2 % and 4 096 B/core of L1.
    """
    for blocks in (1, 2, 4, 8, 16, 64):
        assert read_group_pays(blocks) == 1
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["read_group"] == 1 and plan["depth"] <= 2


@pytest.mark.parametrize("group", [2, 4, 8])
def test_forced_read_group_is_bit_exact_and_bounded(device, group):
    """Forced groups must still be correct (the group writes G windows before ONE
    barrier, so a wrong window index would corrupt without hanging) and must keep
    the CB bounded: G+1 windows, never G+1 * anything that grows with W."""
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    tpd.READ_GROUP_OVERRIDE = group
    try:
        plan = _plan(device, shape, DRAM, out_cfg)
        assert plan["read_group"] == group
        assert plan["depth"] == group + 1, "the group needs one window more than its size"
        assert plan["cb_bytes_per_core"] == (group + 1) * plan["chunk_wt"] * plan["tile_in"]
        _exact(device, shape, DRAM, out_cfg)
    finally:
        tpd.READ_GROUP_OVERRIDE = None


def test_address_probe_is_unreachable_by_default(device):
    """The address-generation probe is a bench instrument with garbage output (it
    sends all 32 reads of a block to ONE bank). It must never be on by default — and
    its number is kept because it is a third confirmation that the read leg is
    bank-bound: 46 851 ns vs 16 743 = 2.80x SLOWER."""
    shape = (1, 1, 2048, 512)
    out_cfg = _shard(_BLOCK, _crs(7, 7), (256, 64), _ROW)
    plan = _plan(device, shape, DRAM, out_cfg)
    assert plan["addr_probe"] == 0
    with _levers(addr=1):
        forced = _plan(device, shape, DRAM, out_cfg)
        assert forced["addr_probe"] == 1  # the hook exists, so the row is re-measurable


# ---------------------------------------------------------------------------
# Non-regression: the interleaved plans this refinement must not touch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,chunk,stagger",
    [
        ((1, 1, 2048, 2048), 16, 0),  # a_square
        ((1, 1, 32, 16384), 8, 3),  # b_wide_short (Refinement 2b's rotation)
        ((1, 1, 2048, 32), 1, 0),  # d_tall_narrow
    ],
    ids=["a_square", "b_wide_short", "d_tall_narrow"],
)
def test_interleaved_plans_are_untouched(device, shape, chunk, stagger):
    """Refinement 3 only fires when exactly one side is sharded. The three
    interleaved bench regimes must keep the plan every prior phase measured."""
    plan = _plan(device, shape, DRAM, DRAM)
    assert plan["path"] == "generic"
    assert plan["alias_in"] == 0 and plan["alias_out"] == 0
    assert plan["chunk_wt"] == chunk
    assert plan["stagger"] == stagger
    assert plan["coalesce_rows"] == 0 and plan["read_group"] == 1 and plan["addr_probe"] == 0


def test_same_spec_sharded_still_takes_path_b(device):
    """Path B (both sides aliased) must win over the one-sided paths — it has zero
    traffic on BOTH sides."""
    shape = (1, 1, 512, 64)
    cfg = _shard(_HEIGHT, _crs(3, 0), (128, 64), _ROW)
    plan = _plan(device, shape, cfg, cfg)
    assert plan["path"] == "alias"
    assert plan["alias_in"] == 1 and plan["alias_out"] == 1
    _exact(device, shape, cfg, cfg)
