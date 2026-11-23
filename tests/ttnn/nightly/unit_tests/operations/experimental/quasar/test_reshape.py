# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

QUASAR_ONLY = "Quasar-native kernels (multi-hart DM kernel, byte-granular NOC transfers)"

TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT

# ttnn dtype -> (torch dtype of the reference, value generator)
DTYPES = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.uint8: torch.int32,
}


def _random(shape, dtype):
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return torch.randn(shape, dtype=DTYPES[dtype])
    low, high = {
        ttnn.int32: (-(2**31), 2**31 - 1),
        ttnn.uint32: (0, 2**31 - 1),
        ttnn.uint16: (0, 2**16 - 1),
        ttnn.uint8: (0, 2**8 - 1),
    }[dtype]
    return torch.randint(low, high, shape, dtype=torch.int32)


def _run(device, in_shape, out_shape, layout, dtype=ttnn.bfloat16, memory_config=None, expect_kernel=True, **kwargs):
    """Reshape on device and compare bit-exactly with torch.reshape. `expect_kernel` states whether the
    call must run a device program (True) or be a metadata-only view (False)."""
    x = _random(in_shape, dtype)
    expected = x.reshape(out_shape)
    tx = ttnn.from_torch(
        x, dtype=dtype, layout=layout, device=device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG
    )
    entries = device.num_program_cache_entries()
    out = ttnn.experimental.quasar.reshape(tx, out_shape, **kwargs)
    new_entries = device.num_program_cache_entries() - entries
    assert (new_entries > 0) == expect_kernel, f"{new_entries} new programs, expect_kernel={expect_kernel}"
    assert tuple(out.shape) == tuple(expected.shape)
    assert out.layout == layout and out.dtype == dtype
    got = ttnn.to_torch(out)
    assert torch.equal(got.to(torch.float64), expected.to(torch.float64)), "reshape is not bit-exact"
    return out


def _quasar(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    torch.manual_seed(0)


# TILE reshapes that really re-tile: the last dim changes, or the second-to-last changes across padding.
TILE_CASES = [
    ((1, 1, 64, 128), (1, 1, 128, 64)),  # tile-aligned, last two dims swap
    ((2048,), (32, 64)),  # rank 1 -> 2
    ((64, 96), (2, 48, 64)),  # rank 2 -> 3
    ((4, 40, 48), (2, 2, 5, 16, 24)),  # rank 3 -> 5, nothing tile-aligned
    ((2, 2, 3, 20, 24), (72, 80)),  # rank 5 -> 2
    ((1, 3, 50, 70), (1, 5, 30, 70)),  # same last dim, padded rows move: whole face-column runs
    ((2, 3, 17, 33), (3, 2, 33, 17)),  # odd sizes: runs split at input row ends and faces
    ((1, 1, 40, 7), (280, 1)),  # last dim -> 1: every element is its own run
    ((280, 1), (1, 1, 7, 40)),  # last dim from 1: every element comes from another input row
    ((1, 2, 64, 64), (1, 1, 128, 64)),  # tile-aligned rows, same last dim: a view
    ((3, 100), (2, 150)),
    ((1, 1, 33, 96), (1, 1, 99, 32)),
]
# Of these, the pure metadata reshapes (last dim kept, no padding on the second-to-last dims).
TILE_VIEWS = {((1, 2, 64, 64), (1, 1, 128, 64))}


@pytest.mark.parametrize("in_shape, out_shape", TILE_CASES)
def test_quasar_reshape_tile(device, in_shape, out_shape):
    _quasar(device)
    _run(device, in_shape, out_shape, TILE, expect_kernel=(in_shape, out_shape) not in TILE_VIEWS)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8])
@pytest.mark.parametrize("layout", [TILE, RM])
def test_quasar_reshape_dtypes(device, dtype, layout):
    _quasar(device)
    _run(device, (2, 3, 20, 44), (4, 55, 24), layout, dtype=dtype)


# ROW_MAJOR reshapes that change the last dim (anything else is a view).
RM_CASES = [
    ((32, 17), (544, 1)),  # last dim -> 1
    ((544, 1), (17, 32)),  # last dim from 1
    ((240,), (4, 6, 10)),  # rank 1 -> 3
    ((2, 3, 4, 5, 6), (6, 5, 4, 3, 2)),  # rank 5 -> 5
    ((2, 3, 4, 5, 6), (720,)),  # rank 5 -> 1
    ((7, 13), (13, 7)),  # odd page sizes
    ((1, 1, 64, 128), (1, 1, 128, 64)),
    ((64, 2501), (32, 5002)),  # 10 KB pages: a page is moved in several windows
    ((3, 20000), (1, 60000)),  # one 120 KB page spread over nodes and harts
    ((1, 60000), (20000, 3)),  # a wide page into many tiny ones
]


@pytest.mark.parametrize("in_shape, out_shape", RM_CASES)
def test_quasar_reshape_row_major(device, in_shape, out_shape):
    _quasar(device)
    _run(device, in_shape, out_shape, RM)


@pytest.mark.parametrize("layout", [TILE, RM])
def test_quasar_reshape_odd_bytes(device, layout):
    """1-byte elements with odd page sizes: every transfer starts and ends off any word boundary."""
    _quasar(device)
    _run(device, (5, 77), (11, 35), layout, dtype=ttnn.uint8)


@pytest.mark.parametrize("layout", [TILE, RM])
@pytest.mark.parametrize(
    "in_shape, out_shape, resolved",
    [
        ((2, 64, 64), (-1, 128), (64, 128)),
        ((4, 6, 20), (2, -1, 8), (2, 30, 8)),
        ((3, 5, 7), (-1,), (105,)),
    ],
)
def test_quasar_reshape_inferred_dim(device, layout, in_shape, out_shape, resolved):
    _quasar(device)
    out = _run(device, in_shape, out_shape, layout)
    assert tuple(out.shape) == resolved


@pytest.mark.parametrize("layout", [TILE, RM])
@pytest.mark.parametrize(
    "in_shape, out_shape",
    [
        ((2, 3, 32, 64), (6, 32, 64)),  # leading dims merge
        ((1, 2, 64, 64), (1, 1, 128, 64)),  # tile-aligned rows, same last dim
        ((4, 32, 64), (4, 32, 64)),  # no-op
    ],
)
def test_quasar_reshape_views_run_no_program(device, layout, in_shape, out_shape):
    """The host fast paths: a reshape that keeps the page contents is a metadata-only view."""
    _quasar(device)
    _run(device, in_shape, out_shape, layout, expect_kernel=False)


@pytest.mark.parametrize("layout", [TILE, RM])
@pytest.mark.parametrize("in_mem, out_mem", [("l1", "l1"), ("dram", "l1"), ("l1", "dram")])
def test_quasar_reshape_l1(device, layout, in_mem, out_mem):
    _quasar(device)
    mem = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
    x = _random((2, 48, 80), ttnn.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem[in_mem])
    out = ttnn.experimental.quasar.reshape(tx, (4, 30, 64), memory_config=mem[out_mem])
    assert out.memory_config().buffer_type == mem[out_mem].buffer_type
    assert torch.equal(ttnn.to_torch(out).float(), x.reshape(4, 30, 64).float())


def _height_sharded(num_nodes, shard_shape):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_nodes - 1, 0))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


# TILE tensors go to the kernel sharded (it addresses pages through TensorAccessor); ROW_MAJOR tensors are
# converted to interleaved and back by the host entry.
@pytest.mark.parametrize("layout", [TILE, RM])
@pytest.mark.parametrize(
    "in_mem, out_shape, out_mem, out_layout",
    [
        ("sharded", (1, 1, 64, 128), None, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),  # output inherits the sharding
        ("sharded", (1, 1, 256, 32), "dram", ttnn.TensorMemoryLayout.INTERLEAVED),
        ("dram", (1, 1, 64, 128), "sharded", ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ],
)
def test_quasar_reshape_sharded(device, layout, in_mem, out_shape, out_mem, out_layout):
    _quasar(device)
    mem = {
        "dram": ttnn.DRAM_MEMORY_CONFIG,
        "sharded": _height_sharded(4, [32, 64]) if out_mem != "sharded" else _height_sharded(2, [32, 128]),
    }
    x = _random((1, 1, 128, 64), ttnn.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem[in_mem])
    kwargs = {} if out_mem is None else {"memory_config": mem[out_mem]}
    out = ttnn.experimental.quasar.reshape(tx, out_shape, **kwargs)
    assert out.memory_config().memory_layout == out_layout
    assert torch.equal(ttnn.to_torch(out).float(), x.reshape(out_shape).float())


# Work split. On the full grid a small tensor gives each node at most one window, so only hart 0 of a
# node works; pinning the op to one or two nodes hands every hart several windows.
@pytest.mark.parametrize("layout", [TILE, RM])
@pytest.mark.parametrize("num_nodes", [1, 2])
@pytest.mark.parametrize("num_threads", [1, 2, 3, 4, 5, 6])
def test_quasar_reshape_threads(device, monkeypatch, layout, num_nodes, num_threads):
    _quasar(device)
    monkeypatch.setenv("TTNN_QSR_RESHAPE_THREADS", str(num_threads))
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_nodes - 1, 0))])
    # TILE: 32 x 4 output tiles = 32 windows of 4. ROW_MAJOR: 1000 pages of 208 B = 26 windows of 39 pages.
    _run(device, (1, 1, 208, 500), (1, 1, 1000, 104), layout, sub_core_grids=grid)


@pytest.mark.parametrize("layout", [TILE, RM])
def test_quasar_reshape_all_nodes(device, layout):
    """More windows than nodes: every node of the grid moves data, with a ragged last window."""
    _quasar(device)
    grid = device.compute_with_storage_grid_size()
    nodes = grid.x * grid.y
    # TILE: 5 * nodes + 3 output tiles of bf16 (4 per window). ROW_MAJOR: wide rows, 4 windows per page.
    if layout == TILE:
        tiles = 5 * nodes + 3
        _run(device, (1, tiles * 32, 48), (1, tiles * 48, 32), layout)
    else:
        _run(device, (nodes, 2 * 7000), (2 * nodes, 7000), layout, dtype=ttnn.float32)


def test_quasar_reshape_program_cache(device):
    """A second call with the same shapes reuses the program and still moves the new tensor's data."""
    _quasar(device)
    for layout in (TILE, RM):
        _run(device, (2, 40, 72), (5, 24, 48), layout)
        entries = device.num_program_cache_entries()
        x = _random((2, 40, 72), ttnn.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device)
        out = ttnn.experimental.quasar.reshape(tx, (5, 24, 48))
        assert device.num_program_cache_entries() == entries
        assert torch.equal(ttnn.to_torch(out).float(), x.reshape(5, 24, 48).float())


def test_quasar_reshape_tile_mode_is_accepted(device):
    """reshape_tile_mode stays in the signature for compatibility; there is no mapping tensor to recreate."""
    _quasar(device)
    _run(
        device,
        (1, 1, 64, 96),
        (1, 1, 96, 64),
        TILE,
        reshape_tile_mode=ttnn._ttnn.operations.experimental.quasar.TileReshapeMapMode.RECREATE,
    )


def _padding_of(out, logical_shape):
    """The tile-padding elements of a TILE tensor, as float64."""
    padded = out.cpu().to_torch_with_padded_shape()
    mask = torch.ones(padded.shape, dtype=torch.bool)
    mask[tuple(slice(0, n) for n in logical_shape)] = False
    assert mask.any()
    return padded[mask].to(torch.float64)


# The kernel fills the tile padding of the output itself: pad_value when given, zero otherwise. The first
# reshape of each pair leaves non-zero data in the hart stages, so stale stage bytes would show up.
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8])
@pytest.mark.parametrize("pad_value", [None, 3], ids=["default_zero", "explicit_3"])
@pytest.mark.parametrize(
    "in_shape, out_shape",
    [((1, 1, 40, 50), (1, 1, 50, 40)), ((1, 32, 32), (1, 16, 64)), ((3, 70, 33), (1, 99, 70))],
    ids=["both_ragged", "short_rows", "many_tiles"],
)
def test_quasar_reshape_tile_padding(device, dtype, pad_value, in_shape, out_shape):
    _quasar(device)
    _run(device, (1, 1, 64, 96), (1, 1, 96, 64), TILE, dtype=dtype)
    kwargs = {} if pad_value is None else {"pad_value": float(pad_value)}
    out = _run(device, in_shape, out_shape, TILE, dtype=dtype, **kwargs)
    assert torch.all(_padding_of(out, out_shape) == (pad_value or 0))


@pytest.mark.parametrize("num_threads", [1, 6])
def test_quasar_reshape_tile_padding_threads(device, monkeypatch, num_threads):
    _quasar(device)
    monkeypatch.setenv("TTNN_QSR_RESHAPE_THREADS", str(num_threads))
    out = _run(device, (4, 100, 90), (6, 60, 100), TILE, pad_value=-2.5)
    assert torch.all(_padding_of(out, (6, 60, 100)) == -2.5)


def test_quasar_reshape_tile_padding_height_sharded(device):
    _quasar(device)
    x = torch.randn((1, 1, 128, 40), dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=TILE, device=device, memory_config=_height_sharded(4, (32, 64)))
    out = ttnn.experimental.quasar.reshape(tx, (1, 1, 64, 80), pad_value=7.0)
    assert torch.equal(ttnn.to_torch(out).float(), x.reshape(1, 1, 64, 80).float())
    assert torch.all(_padding_of(out, (1, 1, 64, 80)) == 7.0)


def test_quasar_reshape_bfloat8_is_rejected(device, expect_error):
    """A block-float reshape that would move data fails up front; its metadata-only reshapes still work."""
    _quasar(device)
    x = torch.randn((1, 1, 64, 96))
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=TILE, device=device)
    with expect_error(RuntimeError, "Quasar reshape cannot re-tile block-float tensors"):
        ttnn.experimental.quasar.reshape(tx, (1, 1, 96, 64))
    entries = device.num_program_cache_entries()
    view = ttnn.experimental.quasar.reshape(tx, (1, 2, 32, 96))
    assert device.num_program_cache_entries() == entries
    assert tuple(view.shape) == (1, 2, 32, 96)
    assert torch.equal(ttnn.to_torch(view), ttnn.to_torch(tx).reshape(1, 2, 32, 96))
