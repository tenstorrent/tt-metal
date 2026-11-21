# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""to_memory_config's own tests: what the golden suite cannot reach (op_design.md, Work Distribution).

The golden suite sizes a cell's input and its sharded destination alike, so it has no reshard between two
different grids (rule R6), no COL_MAJOR shard grid, no partial last shard, and never turns the ring knobs.
Each test pins the regime it means to exercise by reading the core table off the host program.
"""

import pytest
import torch

import ttnn
from eval import memory
from ttnn.operations.to_memory_config import to_memory_config
from ttnn.operations.to_memory_config import to_memory_config_program_artifacts as artifacts

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def _height(device, shard_shape, num_cores, *, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=shard_shape, core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT, orientation=orientation,
        use_height_and_width_as_shard_shape=True)


def _put(device, shape, layout, config, seed=0):
    torch.manual_seed(seed)
    t = torch.randn(shape).to(torch.bfloat16)
    t[t == 0] = 1.0  # nonzero everywhere: an unwritten page cannot pass as a copied one
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=config)


def _regime(x, config):
    """The core table the op would use for this call (the output is allocated the same way)."""
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(x.shape, x.dtype, x.layout, config, tile=x.tile), x.device())
    work = artifacts.split_work(x, out)
    ring = artifacts.ring_geometry(x, out, max(work.counts.values()))
    ttnn.deallocate(out)
    return work, ring


def _check(t, y, config):
    got = ttnn.to_torch(y)
    assert torch.equal(got, t), f"max diff {(got.float() - t.float()).abs().max()}"
    assert memory.memory_axes(y) == memory.memory_axes(_Placed(config))
    if config.is_sharded():
        want, have = config.shard_spec, y.memory_config().shard_spec
        assert tuple(have.shape) == tuple(want.shape)
        assert have.grid == want.grid
        assert have.orientation == want.orientation


class _Placed:
    def __init__(self, config):
        self._config = config

    def memory_config(self):
        return self._config


# --- R-bal: interleaved <-> interleaved, including the entry_bytes = min(aligned slots) cap -------------------

@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize("shape", [(1, 1, 17, 50), (2, 1, 100, 47), (32, 17)], ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("route", [("DRAM", "L1"), ("L1", "DRAM"), ("DRAM", "DRAM"), ("L1", "L1")],
                         ids=lambda r: "%s-to-%s" % r)
def test_balanced(device, shape, layout, route):
    configs = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}
    t, x = _put(device, shape, layout, configs[route[0]])
    work, _ = _regime(x, configs[route[1]])
    assert work.regime == "R-bal"
    _check(t, to_memory_config(x, configs[route[1]]), configs[route[1]])


def test_balanced_covers_the_grid(device):
    """Every engaged core and both kernels ran: 400 tiles over 32 clusters in two groups (13 and 12 pages);
    a core or a kernel that did not run leaves its pages unwritten, which the nonzero input exposes."""
    t, x = _put(device, (1, 1, 1600, 256), TILE, ttnn.DRAM_MEMORY_CONFIG)
    work, ring = _regime(x, ttnn.L1_MEMORY_CONFIG)
    grid = device.compute_with_storage_grid_size()
    assert work.regime == "R-bal" and len(work.cores) == grid.x * grid.y
    assert sorted(set(work.counts.values())) == [12, 13]
    assert (ring.block_pages, ring.depth) == (13, 1)
    _check(t, to_memory_config(x, ttnn.L1_MEMORY_CONFIG), ttnn.L1_MEMORY_CONFIG)


def test_multi_block_double_buffered(device, monkeypatch):
    """The depth-2 path: a ring budget of 16 KiB holds 8 tiles, so 13 pages per core become blocks of 4, the
    last one short."""
    monkeypatch.setattr(artifacts, "RING_BYTES_CAP", 16384)
    t, x = _put(device, (1, 1, 1600, 256), TILE, ttnn.DRAM_MEMORY_CONFIG)
    _, ring = _regime(x, ttnn.L1_MEMORY_CONFIG)
    assert (ring.block_pages, ring.depth, ring.num_entries) == (4, 2, 8)
    _check(t, to_memory_config(x, ttnn.L1_MEMORY_CONFIG), ttnn.L1_MEMORY_CONFIG)


# --- R-in / R-out ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dst", ["DRAM", "L1"])
@pytest.mark.parametrize("case", [((1, 1, 64, 512), TILE, (32, 512), 2), ((1, 4, 4, 64), RM, (4, 64), 4)],
                         ids=["tile", "rm"])
def test_sharded_to_interleaved(device, case, dst):
    shape, layout, shard, cores = case
    src = _height(device, shard, cores)
    dst = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[dst]
    t, x = _put(device, shape, layout, src)
    work, _ = _regime(x, dst)
    assert work.regime == "R-in" and len(work.cores) == cores
    _check(t, to_memory_config(x, dst), dst)


@pytest.mark.parametrize("src", ["DRAM", "L1"])
@pytest.mark.parametrize("case", [((1, 1, 64, 512), TILE, (32, 512), 2), ((1, 4, 4, 64), RM, (4, 64), 4)],
                         ids=["tile", "rm"])
def test_interleaved_to_sharded(device, case, src):
    shape, layout, shard, cores = case
    src = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[src]
    dst = _height(device, shard, cores)
    t, x = _put(device, shape, layout, src)
    work, _ = _regime(x, dst)
    assert work.regime == "R-out" and len(work.cores) == cores
    _check(t, to_memory_config(x, dst), dst)


# --- R6: resharding between different grids, both layouts -----------------------------------------------------

RESHARD = {
    # name: (shape, layout, shard on 2 cores, shard on 1 core)
    "tile": ((1, 1, 64, 512), TILE, (32, 512), (64, 512)),
    "rm": ((1, 4, 4, 64), RM, (8, 64), (16, 64)),
}


@pytest.mark.parametrize("name", RESHARD)
def test_reshard_two_cores_to_one(device, name):
    shape, layout, two, one = RESHARD[name]
    t, x = _put(device, shape, layout, _height(device, two, 2))
    dst = _height(device, one, 1)
    work, _ = _regime(x, dst)
    assert work.regime == "R-in" and len(work.cores) == 2
    _check(t, to_memory_config(x, dst), dst)


@pytest.mark.parametrize("name", RESHARD)
def test_reshard_one_core_to_two(device, name):
    shape, layout, two, one = RESHARD[name]
    t, x = _put(device, shape, layout, _height(device, one, 1))
    dst = _height(device, two, 2)
    work, _ = _regime(x, dst)
    assert work.regime == "R-out" and len(work.cores) == 2
    _check(t, to_memory_config(x, dst), dst)


@pytest.mark.parametrize("name", RESHARD)
def test_reshard_equal_config_is_a_local_copy(device, name):
    """R7: the same config in and out is still one dispatch and a copy; R-in makes it core-local."""
    shape, layout, two, _ = RESHARD[name]
    config = _height(device, two, 2)
    t, x = _put(device, shape, layout, config)
    work, _ = _regime(x, config)
    assert work.regime == "R-in"
    y = to_memory_config(x, config)
    _check(t, y, config)
    assert y.buffer_address() != x.buffer_address()


def test_col_major_orientation_and_partial_shard(device):
    """Shards dealt column-major over a 2 x 2 grid, and a last shard holding half its tiles."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    col_major = ttnn.create_sharded_memory_config(
        shape=(64, 64), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.COL_MAJOR, use_height_and_width_as_shard_shape=True)
    t, x = _put(device, (1, 1, 224, 64), TILE, ttnn.DRAM_MEMORY_CONFIG)  # 7 tile-rows: shards of 2, 2, 2, 1
    work, _ = _regime(x, col_major)
    assert work.regime == "R-out"
    assert [(c.x, c.y) for c in work.cores] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [work.counts[c] for c in work.cores] == [4, 4, 4, 2]
    y = to_memory_config(x, col_major)
    _check(t, y, col_major)
    t2 = ttnn.to_torch(to_memory_config(y, ttnn.L1_MEMORY_CONFIG))
    assert torch.equal(t2, t)


# --- The thread plan knob -------------------------------------------------------------------------------------

@pytest.mark.parametrize("plan", [(2, 0, 2)], ids=lambda p: "R%dC%dW%d" % p)
def test_thread_plan_knob(device, monkeypatch, plan):
    """Turning THREAD_PLAN is one constant; every hart's pages land (a missing hart leaves pages unwritten).
    16 pages per core so the ring divides over two harts per side."""
    monkeypatch.setattr(artifacts, "THREAD_PLAN", plan)
    t, x = _put(device, (1, 1, 2048, 256), TILE, ttnn.DRAM_MEMORY_CONFIG)  # 512 tiles: 16 per cluster
    _check(t, to_memory_config(x, ttnn.L1_MEMORY_CONFIG), ttnn.L1_MEMORY_CONFIG)


# --- One device session: programs before and after ------------------------------------------------------------

def test_one_session_before_and_after(device):
    """Our program after another program, and a tt-metal program (to_layout to row-major, plain NoC
    transfers) after ours: the kernels must hand the sticky transaction tag back (target page)."""
    t, x = _put(device, (1, 1, 64, 512), TILE, ttnn.DRAM_MEMORY_CONFIG)
    first = to_memory_config(x, _height(device, (32, 512), 2))
    second = to_memory_config(first, ttnn.L1_MEMORY_CONFIG)
    _check(t, second, ttnn.L1_MEMORY_CONFIG)
    rm = ttnn.to_layout(second, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(rm)), t)
    third = to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    _check(t, third, ttnn.DRAM_MEMORY_CONFIG)
    rm = ttnn.to_layout(third, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(rm)), t)


# --- Refusals (X1, X2, R3, R5) --------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["L1", None], ids=["word", "none"])
def test_memory_config_must_be_a_config(device, bad):
    _, x = _put(device, (1, 1, 32, 64), TILE, ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(TypeError):
        to_memory_config(x, bad)


def _block(device):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    return ttnn.create_sharded_memory_config(
        shape=(32, 32), core_grid=grid, strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


def test_block_sharded_destination_is_refused(device):
    _, x = _put(device, (1, 1, 64, 64), TILE, ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError, match="to='L1/block'"):
        to_memory_config(x, _block(device))


def test_block_sharded_input_is_refused(device):
    _, x = _put(device, (1, 1, 64, 64), TILE, _block(device))
    with pytest.raises(NotImplementedError, match="sharding='block'"):
        to_memory_config(x, ttnn.L1_MEMORY_CONFIG)


def test_row_major_partial_width_shard_is_refused(device):
    """A row-major width shard that does not divide W leaves a short last page in every row (op_design.md,
    Mechanism caps)."""
    _, x = _put(device, (1, 1, 32, 96), RM, ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError, match="does not divide"):
        to_memory_config(x, _width(device, (32, 64), 2))


# --- Width sharding (Refinement 0) ----------------------------------------------------------------------------
#
# In tile layout a width shard's page is a tile, as everywhere. In row-major layout it is a shard_w-wide run of
# one row, so an entry is a run of lcm(Pi, Po) elements: in_pieces / out_pieces pages per entry on each side
# (the program artifacts' docstring). The cases below cover one piece per side (1, 1), a row split into shard
# pages (1, n), a row assembled from them (n, 1) and both at once (n, m).

def _width(device, shard_shape, num_cores, *, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=shard_shape, core_grid=cores, strategy=ttnn.ShardStrategy.WIDTH, orientation=orientation,
        use_height_and_width_as_shard_shape=True)


def _pieces(x, config):
    out = ttnn.allocate_tensor_on_device(
        ttnn.TensorSpec(x.shape, x.dtype, x.layout, config, tile=x.tile), x.device())
    space = artifacts.entry_space(x, out)
    ttnn.deallocate(out)
    return space.in_pieces, space.out_pieces


WIDTH = {
    # name: (shape, layout, width shard on 2 cores, on 4 cores, height shard on 2 cores)
    "tile": ((1, 1, 64, 512), TILE, (64, 256), (64, 128), (32, 512)),
    "rm": ((1, 4, 8, 64), RM, (32, 32), (32, 16), (16, 64)),
}


@pytest.mark.parametrize("buffer", ["DRAM", "L1"])
@pytest.mark.parametrize("name", WIDTH)
def test_interleaved_to_width_and_back(device, name, buffer):
    shape, layout, two, _, _ = WIDTH[name]
    interleaved = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[buffer]
    width = _width(device, two, 2)
    t, x = _put(device, shape, layout, interleaved)
    work, _ = _regime(x, width)
    assert _pieces(x, width) == ((1, 1) if layout == TILE else (1, 2))
    # A tile is inside one width shard, so its owner pulls it; a row-major row spans both shards (R-bal).
    assert work.regime == ("R-out" if layout == TILE else "R-bal")
    y = to_memory_config(x, width)
    _check(t, y, width)
    assert _pieces(y, interleaved) == ((1, 1) if layout == TILE else (2, 1))
    _check(t, to_memory_config(y, interleaved), interleaved)


@pytest.mark.parametrize("name", WIDTH)
def test_height_to_width_and_back(device, name):
    shape, layout, two, _, height_two = WIDTH[name]
    height, width = _height(device, height_two, 2), _width(device, two, 2)
    t, x = _put(device, shape, layout, height)
    y = to_memory_config(x, width)
    _check(t, y, width)
    z = to_memory_config(y, height)
    _check(t, z, height)


@pytest.mark.parametrize("direction", ["2-to-4", "4-to-2"])
@pytest.mark.parametrize("name", WIDTH)
def test_reshard_between_width_grids(device, name, direction):
    """R6 for width: two width grids that differ (two cores to four and back)."""
    shape, layout, two, four, _ = WIDTH[name]
    src, dst = (_width(device, two, 2), _width(device, four, 4))[:: 1 if direction == "2-to-4" else -1]
    t, x = _put(device, shape, layout, src)
    work, _ = _regime(x, dst)
    if layout == RM:
        # Runs of 32 and 16 columns: an entry is 32 wide, one 2-core page and two 4-core pages.
        assert _pieces(x, dst) == ((1, 2) if direction == "2-to-4" else (2, 1))
        assert work.regime == ("R-in" if direction == "2-to-4" else "R-out") and len(work.cores) == 2
    else:
        assert work.regime == ("R-out" if direction == "2-to-4" else "R-in") and len(work.cores) == 4
    _check(t, to_memory_config(x, dst), dst)


@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_reshard_width_grids_that_cut_rows_apart(device, layout):
    """Width shards of 64 and 96 columns over 3 and 2 cores of W = 192: in row-major layout an entry is the whole
    row, assembled from three pages and split into two (explicit sync on both sides); in tile layout the shards
    are 2 and 3 tiles wide."""
    shape = (1, 1, 32, 192)
    src, dst = _width(device, (32, 64), 3), _width(device, (32, 96), 2)
    t, x = _put(device, shape, layout, src)
    if layout == RM:
        assert _pieces(x, dst) == (3, 2)
    y = to_memory_config(x, dst)
    _check(t, y, dst)
    _check(t, to_memory_config(y, src), src)


def test_width_col_major_and_partial_tile_shard(device):
    """Width shards dealt column-major over a 2 x 2 grid, the last one holding one of its two tile columns."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    col_major = ttnn.create_sharded_memory_config(
        shape=(64, 64), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.COL_MAJOR, use_height_and_width_as_shard_shape=True)
    t, x = _put(device, (1, 1, 64, 224), TILE, ttnn.DRAM_MEMORY_CONFIG)  # 7 tile columns: shards of 2, 2, 2, 1
    work, _ = _regime(x, col_major)
    assert work.regime == "R-out"
    assert [(c.x, c.y) for c in work.cores] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [work.counts[c] for c in work.cores] == [4, 4, 4, 2]
    y = to_memory_config(x, col_major)
    _check(t, y, col_major)
    assert torch.equal(ttnn.to_torch(to_memory_config(y, ttnn.L1_MEMORY_CONFIG)), t)


def test_width_thread_plan_knob(device, monkeypatch):
    """(2, 0, 2) on the explicit row-major sides: 32 rows per cluster, every hart's entries land."""
    monkeypatch.setattr(artifacts, "THREAD_PLAN", (2, 0, 2))
    t, x = _put(device, (1, 1, 1024, 64), RM, ttnn.DRAM_MEMORY_CONFIG)
    width = _width(device, (1024, 32), 2)
    y = to_memory_config(x, width)
    _check(t, y, width)
    _check(t, to_memory_config(y, ttnn.L1_MEMORY_CONFIG), ttnn.L1_MEMORY_CONFIG)


def test_width_one_session_before_and_after(device):
    """The explicit-sync sides in one device session: after our implicit program, and before tt-metal programs
    (to_layout) that issue untagged transfers."""
    t, x = _put(device, (1, 1, 32, 256), RM, ttnn.DRAM_MEMORY_CONFIG)
    first = to_memory_config(x, ttnn.L1_MEMORY_CONFIG)          # implicit on both sides
    width = to_memory_config(first, _width(device, (32, 64), 4))  # explicit writer
    back = to_memory_config(width, ttnn.L1_MEMORY_CONFIG)        # explicit reader
    _check(t, back, ttnn.L1_MEMORY_CONFIG)
    tiled = ttnn.to_layout(back, ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(tiled)), t)
    width2 = to_memory_config(x, _width(device, (32, 64), 4))
    tiled2 = ttnn.to_layout(to_memory_config(width2, ttnn.DRAM_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(tiled2)), t)
