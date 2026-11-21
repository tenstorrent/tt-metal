# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import itertools

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

QUASAR_ONLY = "Quasar-native kernel (six DM threads per cluster moving blocks through L1 windows)"

_LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
_LAYOUT_IDS = ["TILE", "RM"]

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int64,  # torch has no uint32; values stay below 2**31
    ttnn.uint16: torch.int32,
    ttnn.uint8: torch.uint8,
}


# Every tensor of the module gets its own data: a test that reruns a shape must not be satisfied by the previous
# run's result still sitting at the address the allocator hands out again.
_seed = itertools.count(1)


def _random(shape, dtype):
    torch.manual_seed(next(_seed))
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return torch.randn(shape, dtype=_TORCH_DTYPE[dtype])
    high = {ttnn.int32: 2**31 - 1, ttnn.uint32: 2**31 - 1, ttnn.uint16: 2**16 - 1, ttnn.uint8: 255}[dtype]
    low = -(2**31) if dtype == ttnn.int32 else 0
    return torch.randint(low, high, shape, dtype=_TORCH_DTYPE[dtype])


def _run_permute(device, shape, dims, layout, dtype=ttnn.bfloat16, memory_config=None, pad_value=None):
    x = _random(shape, dtype)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    kwargs = {}
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    if pad_value is not None:
        kwargs["pad_value"] = pad_value
    out = ttnn.experimental.quasar.permute(tx, dims, **kwargs)

    ref = torch.permute(x, dims)
    assert tuple(out.shape) == tuple(ref.shape)
    assert out.layout == layout
    assert out.dtype == dtype
    got = ttnn.to_torch(out)
    # A permute moves bits: the result must be exact.
    assert torch.equal(got.to(ref.dtype), ref), f"permute {shape} {dims} is not bit-exact"
    return out


# Ranks 2 to 5, shapes that are not tile aligned, and every way the page-forming dimensions can move:
# pages that move unchanged, tiles transposed in place, one or both tile dimensions leaving the tile, the
# ROW_MAJOR row dimension leaving the row, identity (as a copy) and the full reversal.
_CASES = [
    ((5, 7), (1, 0)),
    ((33, 70), (1, 0)),
    ((33, 70), (0, 1)),
    ((3, 40, 50), (1, 0, 2)),
    ((3, 40, 50), (0, 2, 1)),
    ((3, 40, 50), (2, 0, 1)),
    ((3, 40, 50), (1, 2, 0)),
    ((3, 40, 50), (2, 1, 0)),
    ((2, 3, 36, 70), (1, 0, 2, 3)),
    ((2, 3, 36, 70), (0, 1, 3, 2)),
    ((2, 3, 36, 70), (0, 2, 3, 1)),
    ((2, 3, 36, 70), (0, 3, 1, 2)),
    ((2, 3, 36, 70), (0, 2, 1, 3)),
    ((2, 3, 36, 70), (2, 3, 0, 1)),
    ((2, 3, 36, 70), (3, 2, 1, 0)),
    ((2, 34, 3, 5), (2, 3, 0, 1)),
    ((2, 3, 4, 20, 33), (0, 2, 1, 3, 4)),
    ((2, 3, 4, 20, 33), (3, 0, 4, 1, 2)),
    ((2, 3, 4, 20, 33), (4, 3, 2, 1, 0)),
    ((1, 1, 1, 1, 37), (4, 1, 2, 3, 0)),
]
_CASE_IDS = ["x".join(map(str, s)) + "_" + "".join(map(str, d)) for s, d in _CASES]


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize("shape, dims", _CASES, ids=_CASE_IDS)
def test_quasar_permute(device, shape, dims, layout):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    # The identity is exercised as a copy into the other memory: with the input's own config it returns the input.
    memory_config = ttnn.L1_MEMORY_CONFIG if tuple(dims) == tuple(range(len(dims))) else None
    _run_permute(device, shape, dims, layout, memory_config=memory_config)


# 1, 2 and 4 byte elements: the reorder is a typed element copy, the page move a byte copy.
@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8],
    ids=["float32", "int32", "uint32", "uint16", "uint8"],
)
@pytest.mark.parametrize(
    "shape, dims",
    [
        ((2, 3, 36, 70), (1, 0, 2, 3)),
        ((2, 3, 36, 70), (0, 1, 3, 2)),
        ((2, 3, 36, 70), (0, 2, 3, 1)),
        ((2, 3, 36, 70), (3, 2, 1, 0)),
        ((3, 5, 7), (2, 0, 1)),
    ],
    ids=["page_move", "last_two_swapped", "nchw_to_nhwc", "reversal", "rank3_small"],
)
def test_quasar_permute_dtypes(device, shape, dims, layout, dtype):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    _run_permute(device, shape, dims, layout, dtype=dtype)


# Work split: each shape cuts into at least 6 blocks per cluster of an 8x4 grid in both layouts (192 or more
# blocks), so every one of the six DM threads of every cluster moves at least one block, for each kind of block.
@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "shape, dims",
    [
        ((8, 16, 40, 70), (1, 0, 2, 3)),
        ((16, 12, 20, 64), (0, 1, 3, 2)),
        ((48, 64, 64, 3), (0, 3, 1, 2)),
        ((2, 3, 40, 160), (3, 2, 1, 0)),
    ],
    ids=["page_move", "last_two_swapped", "nhwc_to_nchw", "reversal"],
)
def test_quasar_permute_all_clusters(device, shape, dims, layout):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    _run_permute(device, shape, dims, layout)


# 1 to 6 DM threads per cluster; the shape leaves every thread of every cluster at least one block.
@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize("threads", [1, 2, 3, 4, 5, 6])
def test_quasar_permute_thread_counts(device, monkeypatch, layout, threads):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    monkeypatch.setenv("TTNN_QSR_PERMUTE_THREADS", str(threads))
    _run_permute(device, (8, 24, 6, 40), (1, 0, 3, 2), layout)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
def test_quasar_permute_l1_tensors(device, layout):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    x = _random((2, 3, 36, 70), ttnn.bfloat16)
    tx = ttnn.from_torch(x, layout=layout, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.experimental.quasar.permute(tx, (0, 3, 1, 2))
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert torch.equal(ttnn.to_torch(out), torch.permute(x, (0, 3, 1, 2)))
    out = ttnn.experimental.quasar.permute(tx, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert torch.equal(ttnn.to_torch(out), torch.permute(x, (0, 2, 3, 1)))


# A TILE output whose tiles are rebuilt element by element carries pad_value in its tile padding.
@pytest.mark.parametrize(
    "dims", [(0, 1, 3, 2), (0, 2, 3, 1), (2, 3, 0, 1)], ids=["last_two_swapped", "one_leaves", "both_leave"]
)
@pytest.mark.parametrize("pad_value", [0.0, 3.0])
def test_quasar_permute_tile_padding(device, dims, pad_value):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    shape = (2, 3, 36, 70)
    out = _run_permute(device, shape, dims, ttnn.TILE_LAYOUT, pad_value=pad_value)
    padded = out.cpu().to_torch_with_padded_shape()
    ref = torch.permute(torch.zeros(shape), dims)
    mask = torch.ones(padded.shape, dtype=torch.bool)
    mask[tuple(slice(0, n) for n in ref.shape)] = False
    assert mask.any()
    assert torch.all(padded[mask].float() == pad_value)


def test_quasar_permute_negative_dims_and_cache(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    for _ in range(2):
        _run_permute(device, (2, 3, 36, 70), (0, -1, 1, -2), ttnn.ROW_MAJOR_LAYOUT)
    # Same permutation and rank, another shape: its own program, its own block plan.
    _run_permute(device, (3, 2, 20, 45), (0, -1, 1, -2), ttnn.ROW_MAJOR_LAYOUT)


def test_quasar_permute_identity_returns_input(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    x = _random((3, 40, 50), ttnn.bfloat16)
    tx = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.experimental.quasar.permute(tx, (0, 1, 2))
    assert torch.equal(ttnn.to_torch(out), x)


def test_quasar_permute_rank1(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)
    for layout in _LAYOUTS:
        _run_permute(device, (70,), (0,), layout, memory_config=ttnn.L1_MEMORY_CONFIG)


# ---- contract rejections (host-side validation, run on every arch) ----


def test_quasar_permute_rejects_bad_dims(device, expect_error):
    x = ttnn.from_torch(torch.zeros((2, 3, 4), dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "dims given for a rank 3 tensor"):
        ttnn.experimental.quasar.permute(x, (1, 0))
    with expect_error(RuntimeError, "is out of range for rank 3"):
        ttnn.experimental.quasar.permute(x, (0, 1, 3))
    with expect_error(RuntimeError, "is repeated in the permutation"):
        ttnn.experimental.quasar.permute(x, (0, 1, 1))


def test_quasar_permute_rejects_sharded(device, expect_error):
    x = ttnn.from_torch(torch.zeros((1, 1, 64, 64), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    sharded = ttnn.create_sharded_memory_config(
        (1, 1, 64, 64), core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
    )
    with expect_error(RuntimeError, "supports interleaved tensors only"):
        ttnn.experimental.quasar.permute(x, (0, 1, 3, 2), memory_config=sharded)
