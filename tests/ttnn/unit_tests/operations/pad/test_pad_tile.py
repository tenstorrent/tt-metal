# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 0 of pad: tile layout and rank 2.

In tile layout a page is a tile. With elements added after the data, an input element keeps its tile and its
place in the tile, so an input tile is copied whole and the reader writes `value` over the input's tile padding
in the last tile row / column (those places are added elements of the output). The inputs here carry non-zero
tile padding (7.0), which a missing fill would leak into the output, and value 1.0 runs through the program entry
(validate() bypassed: the kernels are value-general), which a missing pad-tile write would show on the
zero-initialized simulator DRAM.
"""

import pytest
import torch

import ttnn
from ttnn.operations.pad import pad
from ttnn.operations.pad.pad import _run
from ttnn.operations.pad.pad_program_artifacts import PAD_THREADS

GARBAGE = 7.0  # what the input's tile padding holds


def _reference(t, padding, value):
    flat = [amount for pair in reversed(list(padding)) for amount in pair]
    return torch.nn.functional.pad(t, flat, mode="constant", value=value)


def _tile_input(device, shape, seed=0):
    """A bfloat16 tile-layout DRAM tensor of logical `shape` whose tile padding holds GARBAGE."""
    torch.manual_seed(seed)
    t = torch.randn(shape, dtype=torch.bfloat16)
    padded = list(shape[:-2]) + [-(-shape[-2] // 32) * 32, -(-shape[-1] // 32) * 32]
    full = torch.full(padded, GARBAGE, dtype=torch.bfloat16)
    full[tuple(slice(0, d) for d in shape)] = t
    host = ttnn.from_torch(full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    host = ttnn.reshape(host, ttnn.Shape(list(shape)), ttnn.Shape(padded))
    return t, host.to(device)


def _assert_exact(got, expected):
    assert got.shape == expected.shape, (got.shape, expected.shape)
    bad = (got != expected).sum().item()
    assert bad == 0, f"{bad} elements differ"


def _case_id(case):
    shape, padding = case
    return "x".join(map(str, shape)) + "_p" + "_".join(f"{b}+{a}" for b, a in padding)


def test_qwen_model_call(device):
    """The model's call: [70, 512] bfloat16 tile layout in DRAM, 58 rows of zeros after the data -> [128, 512].
    Rows 70..95 of the input's third tile row are tile padding (GARBAGE); in the output they are added elements."""
    t, x = _tile_input(device, (70, 512))
    y = pad(x, [[0, 58], [0, 0]], 0)
    assert y.layout == ttnn.TILE_LAYOUT and y.dtype == ttnn.bfloat16
    mc = y.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(y)
    assert tuple(got.shape) == (128, 512)
    assert torch.equal(got[:70], t), "data rows changed"
    assert (got[70:] == 0).all(), f"{(got[70:] != 0).sum().item()} added elements are not 0"


TILE_CASES = [
    ((70, 512), ((0, 58), (0, 0))),  # Qwen, rank 2
    ((32, 17), ((0, 0), (0, 15))),  # rank 2, W non-aligned, to one tile
    ((1, 3, 20, 20), ((0, 0), (0, 13), (0, 0), (0, 0))),  # YOLO stem: C, tile padding inside each image
    ((1, 1, 17, 50), ((0, 0), (0, 1), (0, 20), (0, 33))),  # H and W cross tile boundaries, C
    ((4, 8, 47, 256), ((0, 2), (0, 3), (0, 1), (0, 4))),  # every dimension
    ((1, 1, 64, 128), ((0, 0), (0, 0), (0, 17), (0, 40))),  # aligned input, whole new tile rows and columns
]


@pytest.mark.parametrize("threads", [PAD_THREADS, (1, 2), (2, 2)], ids=lambda t: "R%dW%d" % t)
@pytest.mark.parametrize("case", TILE_CASES, ids=_case_id)
@pytest.mark.parametrize("value", [0.0, 1.0])
def test_tile_coverage(device, case, threads, value):
    shape, padding = case
    t, x = _tile_input(device, shape)
    y = _run(x, padding, value, threads=threads)
    assert y.layout == ttnn.TILE_LAYOUT
    _assert_exact(ttnn.to_torch(y), _reference(t, padding, value))


@pytest.mark.parametrize("block_rows", [1, 3])
@pytest.mark.parametrize("case", [TILE_CASES[0], TILE_CASES[4]], ids=_case_id)
def test_tile_streamed_regime(device, case, block_rows):
    """Several blocks per core, a ring that wraps: the tile-padding fill runs on every lap, not the first only."""
    shape, padding = case
    t, x = _tile_input(device, shape)
    y = _run(x, padding, 1.0, block_rows_override=block_rows)
    _assert_exact(ttnn.to_torch(y), _reference(t, padding, 1.0))


RANK2_RM_CASES = [
    ((32, 64), ((0, 0), (0, 64))),
    ((17, 64), ((0, 15), (0, 0))),
    ((32, 17), ((0, 0), (0, 15))),
    ((70, 512), ((0, 58), (0, 0))),
]


@pytest.mark.parametrize("case", RANK2_RM_CASES, ids=_case_id)
def test_rank2_row_major(device, case):
    shape, padding = case
    torch.manual_seed(3)
    t = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _assert_exact(ttnn.to_torch(pad(x, padding, 0.0)), _reference(t, padding, 0.0))
    _assert_exact(ttnn.to_torch(_run(x, padding, 1.0)), _reference(t, padding, 1.0))


def test_tile_session_between_programs(device):
    """In one device session: a tt-metal program, then pad on a tile tensor, then tt-metal's untilize of pad's
    output (untagged transfers after the writer's implicit, ring-tagged writes)."""
    torch.manual_seed(1)
    a = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    rm = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tiled = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)  # a program before pad
    y = pad(tiled, ((0, 0), (0, 1), (0, 0), (0, 32)), 0.0)
    expected = _reference(a, ((0, 0), (0, 1), (0, 0), (0, 32)), 0.0)
    _assert_exact(ttnn.to_torch(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)), expected)  # a program after pad
    t, x = _tile_input(device, (70, 512))
    y = pad(x, ((0, 58), (0, 0)), 0.0)
    _assert_exact(ttnn.to_torch(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)), _reference(t, ((0, 58), (0, 0)), 0.0))


def test_tile_before_padding_is_refused(device):
    """Elements before the data are outside SUPPORTED (padding_side); in tile layout a shift in H or W is not a
    page copy, and the program refuses it too."""
    _, x = _tile_input(device, (1, 1, 32, 64))
    with pytest.raises(NotImplementedError):
        pad(x, ((0, 0), (0, 0), (1, 0), (0, 0)), 0.0)
    with pytest.raises(NotImplementedError):
        _run(x, ((0, 0), (0, 0), (1, 0), (0, 0)), 0.0)
