# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design's Work Distribution asks for, beyond the acceptance test:

- every wrap of the reader's (c, h, r, b) walk, on shapes whose per-core ranges cut a token row;
- the knobs turned (thread plan, block size, ring depth, cores engaged): each value must give the same bytes;
- engaged-core counts of the host split;
- the op in one device session after another program and before a tt-metal program (`ttnn.to_layout`):
  the sticky NoC tag must be handed back.
"""

import pytest
import torch

import ttnn
from ttnn.operations.nlp_concat_heads import nlp_concat_heads
from ttnn.operations.nlp_concat_heads.nlp_concat_heads_program_artifacts import (
    create_program_artifacts,
    head_geometry,
    split_pages,
)


def _reference(x):
    b, h, s, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, 1, s, h * d)


def _input(device, shape, seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype)
    return x, ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _run(device, x_dev, **knobs):
    """The op's program with knobs set, through the same dispatch as the entry point."""
    b, h, s, d = (int(v) for v in x_dev.shape)
    out_spec = ttnn.TensorSpec(ttnn.Shape([b, 1, s, h * d]), x_dev.dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(out_spec, device)
    spec, run_args, idx = create_program_artifacts(x_dev, out, **knobs)
    return ttnn.generic_op([x_dev, out], spec, run_args, idx)


def _exact(got, expected):
    assert torch.equal(got, expected), (got.float() - expected.float()).abs().max()


@pytest.mark.parametrize(
    "shape", [(1, 8, 64, 64), (1, 32, 128, 64), (1, 64, 32, 128), (1, 2, 512, 64)], ids=lambda s: "x".join(map(str, s))
)
def test_walk_wraps(device, shape):
    x, x_dev = _input(device, shape)
    _exact(ttnn.to_torch(nlp_concat_heads(x_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)), _reference(x))


# [1, 37, 32, 32]: 37 tiles, one or two per cluster (a short last round for a 2-hart plan).
# [1, 5, 64, 960]: 5 * 2 * 30 = 300 tiles, 9 or 10 per cluster, ranges cut mid-row and mid-head.
KNOBS = [
    {"threads": (2, 0, 2)},
    {"threads": (2, 0, 1)},
    {"threads": (1, 0, 2)},
    {"threads": (4, 0, 2)},
    {"block_tiles": 1},
    {"block_tiles": 3},
    {"block_tiles": 4, "threads": (2, 0, 2)},
    {"ring_depth_tiles": 4},
    {"ring_depth_tiles": 16},
    {"min_tiles_per_core": 4},
    {"min_tiles_per_core": 8, "threads": (2, 0, 2), "block_tiles": 3},
]


@pytest.mark.parametrize("knobs", KNOBS, ids=lambda k: "-".join(f"{n}={v}" for n, v in k.items()).replace(" ", ""))
@pytest.mark.parametrize("shape", [(1, 37, 32, 32), (1, 5, 64, 960)], ids=lambda s: "x".join(map(str, s)))
def test_knobs_give_the_same_bytes(device, shape, knobs):
    x, x_dev = _input(device, shape)
    _exact(ttnn.to_torch(_run(device, x_dev, **knobs)), _reference(x))


@pytest.mark.parametrize(
    "shape,cores",
    [((1, 4, 32, 64), 8), ((1, 4, 32, 128), 16), ((1, 8, 64, 64), 32), ((1, 32, 128, 64), 32), ((1, 2, 32, 32), 2)],
)
def test_engaged_cores(device, shape, cores):
    grid = device.compute_with_storage_grid_size()
    b, h, st, dt = head_geometry(shape)
    pages = b * h * st * dt
    _, counts, starts = split_pages(pages, grid)
    assert len(counts) == min(grid.x * grid.y, pages) == min(grid.x * grid.y, cores)
    assert sum(counts.values()) == pages
    assert max(counts.values()) - min(counts.values()) <= 1
    # contiguous ranges, in core order
    ranges = sorted((starts[c], counts[c]) for c in counts)
    assert all(s0 + n0 == s1 for (s0, n0), (s1, _) in zip(ranges, ranges[1:]))


def test_one_device_session_with_tt_metal_programs(device):
    """After another program (a tt-metal tilize, and the op itself), and before ttnn.to_layout (an untilize with
    plain NoC transfers), all in one device session. (ttnn.add is a Metal 1
    program and does not run on Quasar.)"""

    def tt_metal_tilize(seed):
        torch.manual_seed(100 + seed)
        a = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
        a_rm = ttnn.from_torch(a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
        _exact(ttnn.to_torch(ttnn.to_layout(a_rm, ttnn.TILE_LAYOUT)), a)

    tt_metal_tilize(0)

    for seed, shape in enumerate([(1, 4, 32, 64), (1, 16, 32, 64), (1, 4, 32, 128)]):
        x, x_dev = _input(device, shape, seed)
        out = nlp_concat_heads(x_dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _exact(ttnn.to_torch(out), _reference(x))
        rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        _exact(ttnn.to_torch(ttnn.from_device(rm)), _reference(x))

    # And a plan with more harts, then tt-metal again.
    x, x_dev = _input(device, (1, 5, 64, 960), 7)
    out = _run(device, x_dev, threads=(4, 0, 2))
    _exact(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))), _reference(x))
    tt_metal_tilize(1)


def test_rejections(device):
    x, x_dev = _input(device, (1, 4, 32, 64))
    with pytest.raises(TypeError):
        nlp_concat_heads(x_dev, memory_config="DRAM")
    with pytest.raises(NotImplementedError):
        nlp_concat_heads(x_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    rm = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16)
    with pytest.raises(NotImplementedError):
        nlp_concat_heads(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    r3 = ttnn.from_torch(x[0], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    with pytest.raises(ValueError):
        nlp_concat_heads(r3, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def test_tile_relay_refuses_a_head_dim_off_the_tile_grid(device):
    """The program builder itself refuses D % 32 != 0 (regime `subtile_assemble`, not built), so a validate()
    widened ahead of that regime cannot dispatch the whole-tile walk on a head that ends inside a tile."""
    x = torch.randn(1, 4, 32, 50, dtype=torch.bfloat16)
    x_dev = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    out_spec = ttnn.TensorSpec(ttnn.Shape([1, 1, 32, 200]), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(out_spec, device)
    with pytest.raises(NotImplementedError):
        create_program_artifacts(x_dev, out)
