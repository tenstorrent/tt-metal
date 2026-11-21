# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""concat: the tests op_design.md's Work Distribution requires beside the acceptance test.

- regime-pinned cases (page_copy, run_assembly row-, face- and element-granular), with the host's choice asserted;
- a ragged last block (the block knob turned to a value that does not divide a core's share);
- the work split: every output page on exactly one cluster;
- one device session: the op after another program, after a tt-metal program, and before one (to_layout);
- placement (R5), refusals (R4) and malformed calls (X1..X4).

Values are moved, never computed: every check is exact.
"""

import pytest
import torch

import ttnn
from ttnn.operations.concat import concat
from ttnn.operations.concat.concat import concat_with_block_tiles
from ttnn.operations.concat.concat_program_artifacts import BLOCK_TILES, select_regime, split_tiles

# (dim, shapes, regime the host must choose)
PINNED = [
    ((-1, (1, 1, 64, 64), (1, 1, 64, 80)), "page_copy"),
    ((-2, (1, 64, 144), (1, 16, 144), (1, 4, 144)), "run_assembly"),  # row-granular, seam at row 80
    ((-2, (1, 2, 400), (1, 2, 400)), "run_assembly"),  # row-granular, seam at row 2
    ((-1, (1, 1, 32, 48), (1, 1, 32, 16)), "run_assembly"),  # face-granular, seam at column 48
    ((-1, (1, 1, 32, 50), (1, 1, 32, 14)), "run_assembly"),  # element-granular
    ((-1, (4, 8, 32, 47), (4, 8, 32, 100), (4, 8, 32, 50)), "run_assembly"),  # element-granular, many tiles
    ((-2, (1, 1, 17, 64), (1, 1, 15, 64)), "run_assembly"),  # 17 + 15 rows in one tile
    ((-2, (2, 4, 32, 256), (2, 4, 96, 256), (2, 4, 32, 256)), "page_copy"),
    ((-1, (2, 1, 100, 32), (2, 1, 100, 96)), "page_copy"),  # H non-aligned, seam aligned
    ((-1, (1, 1, 64, 17), (1, 1, 64, 47)), "run_assembly"),
]


def _id(entry):
    dim, *shapes = entry
    return "+".join("x".join(map(str, s)) for s in shapes) + f"_dim{dim}"


def _make(device, shapes, memory_config=ttnn.DRAM_MEMORY_CONFIG, seed=0):
    torch.manual_seed(seed)
    host = [torch.randn(*s, dtype=torch.bfloat16) for s in shapes]
    dev = [
        ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=memory_config)
        for t in host
    ]
    return host, dev


def _check(out, host, dim):
    got = ttnn.to_torch(out)
    expected = torch.cat(host, dim=dim)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert torch.equal(got, expected), (got.float() - expected.float()).abs().max()


@pytest.mark.parametrize("entry,regime", PINNED, ids=[_id(e) for e, _ in PINNED])
def test_regime_selection(entry, regime):
    dim, *shapes = entry
    canonical = dim if dim < 0 else dim - len(shapes[0])
    assert select_regime(shapes, canonical) == regime


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["DRAM", "L1"])
@pytest.mark.parametrize("entry", [e for e, _ in PINNED], ids=[_id(e) for e, _ in PINNED])
def test_pinned(device, entry, memory_config):
    dim, *shapes = entry
    host, dev = _make(device, shapes, memory_config)
    _check(concat(dev, dim=dim), host, dim)


# A block knob that does not divide a core's share: every core's last block is ragged.
RAGGED = [
    (-1, (4, 8, 32, 47), (4, 8, 32, 100), (4, 8, 32, 50)),  # run_assembly: 224 tiles, 7 per cluster
    (-2, (2, 4, 50, 64), (2, 4, 50, 64), (2, 4, 28, 64)),  # run_assembly: 8 images x 4 x 2 = 64 tiles
    (1, (2, 512, 1024), (2, 512, 1024)),  # page_copy: 2048 tiles, 64 per cluster
]


@pytest.mark.parametrize("block_tiles", [1, 3, 5, BLOCK_TILES, 64])
@pytest.mark.parametrize("entry", RAGGED, ids=[_id(e) for e in RAGGED])
def test_ragged_last_block(device, entry, block_tiles):
    dim, *shapes = entry
    host, dev = _make(device, shapes)
    _check(concat_with_block_tiles(dev, dim=dim, block_tiles=block_tiles), host, dim)


@pytest.mark.parametrize("num_tiles", [1, 10, 13, 15, 31, 32, 33, 224, 2048])
def test_split_covers_every_page_once(device, num_tiles):
    """The writer of cluster c stores pages [start, start + n): together they are every page, once."""
    _, counts, starts = split_tiles(num_tiles, device.compute_with_storage_grid_size())
    pages = sorted(p for core, n in counts.items() for p in range(starts[core], starts[core] + n))
    assert pages == list(range(num_tiles))


def test_output_is_dram_interleaved(device):
    """R5: the output is in DRAM, interleaved, whichever buffer the inputs are in."""
    host, dev = _make(device, [(1, 1, 64, 64), (1, 1, 64, 80)], ttnn.L1_MEMORY_CONFIG)
    out = concat(dev, dim=3)
    config = out.memory_config()
    assert config.buffer_type == ttnn.BufferType.DRAM
    assert config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _check(out, host, 3)


def test_programs_share_a_device_session(device):
    """Both regimes after each other, after a tt-metal program, and followed by one, in one device session.

    page_copy uses implicit reads, run_assembly explicit tagged reads; the writer is implicit in both. A sticky
    transaction tag left by one program would hang or corrupt the next.
    """
    steps = PINNED[:6] + PINNED[:2]
    for seed, (entry, _) in enumerate(steps):
        dim, *shapes = entry
        host, dev = _make(device, shapes, seed=seed)
        # A tt-metal program before ours (an untilize with plain transfers).
        rm = ttnn.to_layout(dev[0], ttnn.ROW_MAJOR_LAYOUT)
        assert torch.equal(ttnn.to_torch(rm), host[0])
        out = concat(dev, dim=dim)
        _check(out, host, dim)
        # A tt-metal program after ours.
        rm_out = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)))
        assert torch.equal(rm_out, torch.cat(host, dim=dim))


def test_op_twice_in_a_row(device):
    for entry in [PINNED[0][0], PINNED[4][0], PINNED[0][0], PINNED[1][0], PINNED[1][0]]:
        dim, *shapes = entry
        host, dev = _make(device, shapes)
        _check(concat(dev, dim=dim), host, dim)


# --- refusals (R4) and malformed calls (X1..X4) ---

REFUSED = [
    ("float32", ttnn.float32, (-1, (1, 1, 32, 64), (1, 1, 32, 64))),
    ("dim -3", ttnn.bfloat16, (-3, (1, 4, 32, 64), (1, 8, 32, 64))),
    ("count 1", ttnn.bfloat16, (-1, (1, 1, 32, 64))),
    ("count 5", ttnn.bfloat16, (-1, *[(1, 1, 32, 32)] * 5)),
    ("rank 2", ttnn.bfloat16, (-1, (32, 64), (32, 64))),
]


@pytest.mark.parametrize("dtype,entry", [r[1:] for r in REFUSED], ids=[r[0] for r in REFUSED])
def test_refused(device, dtype, entry):
    dim, *shapes = entry
    dev = [ttnn.from_torch(torch.zeros(*s), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype) for s in shapes]
    with pytest.raises(NotImplementedError):
        concat(dev, dim=dim)


def test_sharded_refused_by_name(device):
    """R6: a sharded tensor is refused through the `sharding` axis (eval.memory.memory_axes), by name.

    The tensors are bfloat16 TILE and otherwise a PHASE0 call, so `sharding` is the only axis outside SUPPORTED.
    YOLOv8's untraced calls: [1,1,64,64] + [1,1,64,32] height-sharded in L1, shard height 32 on two cores.
    """
    shapes = [(1, 1, 64, 64), (1, 1, 64, 32)]
    dev = []
    for s in shapes:
        config = ttnn.create_sharded_memory_config(
            shape=(32, s[-1]),
            core_grid=ttnn.CoreGrid(y=1, x=2),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        dev.append(
            ttnn.from_torch(
                torch.zeros(*s), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=config
            )
        )
    with pytest.raises(NotImplementedError, match="sharding"):
        concat(dev, dim=-1)


def test_row_major_refused(device):
    dev = [ttnn.from_torch(torch.zeros(1, 1, 32, 64), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)] * 2
    with pytest.raises(NotImplementedError):
        concat(dev, dim=-1)


@pytest.mark.parametrize(
    "entry,error",
    [
        ((-1, (1, 1, 32, 64), (1, 1, 64, 64)), ValueError),  # X1
        ((-1, (1, 1, 32, 64), (1, 32, 64)), ValueError),  # X2
        ((4, (1, 1, 32, 64), (1, 1, 32, 64)), IndexError),  # X3
        ((0,), ValueError),  # X4
    ],
    ids=["X1", "X2", "X3", "X4"],
)
def test_malformed(device, entry, error):
    dim, *shapes = entry
    _, dev = _make(device, shapes)
    with pytest.raises(error):
        concat(dev, dim=dim)
