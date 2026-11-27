# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_embedding.

out[i, :] = W[idx[i], :] as a TILE tensor idx.shape + [D]. Every shape below is a production case
of the golden suite (BB-22). The op copies values and computes none, so every logical value is
compared bit for bit (helpers.TOLERANCES holds ("exact", None)), special bf16 patterns included.
"""

import functools

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.g6_embedding import g6_embedding, select_plan

L1_BOUND_BYTES = 1347776  # BB-05: what the model leaves free per core

QWEN_SMALL = (262144, 64)
QWEN_LARGE = (248320, 1280)
FLUX_LARGE = (131072, 2560)
FLUX_43 = (32, 5120)
FLUX_44 = (1, 256, 128)

# bfloat16 patterns of spec rule 2: -0.0, subnormals 2^-133 and -2^-130, +inf, -inf, quiet NaN,
# NaN with sign and payload set, signalling NaN.
SPECIAL_BITS = (0x8000, 0x0001, 0x8008, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81)

# How the model calls the op: Qwen passes an explicit DRAM memory_config, FLUX passes none and
# sets runtime_rows=False.
QWEN_KWARGS = "qwen"
FLUX_KWARGS = "flux"


def _call_kwargs(kind):
    if kind == QWEN_KWARGS:
        return {"layout": ttnn.TILE_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
    return {"layout": ttnn.TILE_LAYOUT, "runtime_rows": False}


@functools.lru_cache(maxsize=2)
def _table(weight_shape):
    """Random table; cached because the Qwen and FLUX tables are 0.6-0.7 GB. Do not mutate."""
    generator = torch.Generator().manual_seed(42)
    return torch.randn(weight_shape, generator=generator, dtype=torch.bfloat16)


def _table_2d(weight):
    return weight if weight.dim() == 2 else weight[0]


def _reference(index, weight):
    # Gather in the table's own dtype: a float32 detour would rewrite NaN payloads.
    return F.embedding(index.to(torch.int64), _table_2d(weight))


def _index(index_shape, vocab):
    return torch.randint(0, vocab, index_shape, dtype=torch.int32)


def _device_table(weight, device):
    return ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _device_index(index, device, dtype=ttnn.uint32):
    return ttnn.from_torch(
        index,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_bit_exact(output, expected):
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.dtype == ttnn.bfloat16
    placement = output.memory_config()
    assert placement.buffer_type == ttnn.BufferType.DRAM
    assert placement.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    actual = ttnn.to_torch(output)
    assert tuple(actual.shape) == tuple(expected.shape)
    assert actual.dtype == torch.bfloat16
    got = actual.contiguous().view(torch.int16)
    want = expected.contiguous().view(torch.int16)
    differs = got != want
    if bool(differs.any()):
        first = tuple(int(i) for i in differs.nonzero()[0])
        raise AssertionError(
            f"{int(differs.sum())} of {differs.numel()} values differ; first at {first}: "
            f"got {int(got[first]) & 0xFFFF:#06x} expected {int(want[first]) & 0xFFFF:#06x}"
        )


def _run_and_check(device, index, weight, kind):
    tt_weight = _device_table(weight, device)
    tt_index = _device_index(index, device)
    output = g6_embedding(tt_index, tt_weight, **_call_kwargs(kind))
    _assert_bit_exact(output, _reference(index, weight))
    for tensor in (output, tt_index, tt_weight):
        ttnn.deallocate(tensor)


# Ordered by table so the two-entry table cache is hit.
SHAPE_CASES = [
    pytest.param((1, 1), QWEN_SMALL, QWEN_KWARGS, id="qwen231-idx1x1-one_tile_row"),
    pytest.param((1, 31), QWEN_SMALL, QWEN_KWARGS, id="qwen207-idx1x31-h_non_aligned"),
    pytest.param((16, 4096), QWEN_SMALL, QWEN_KWARGS, id="qwen191-idx16x4096-d64"),
    pytest.param((1, 128), QWEN_LARGE, QWEN_KWARGS, id="qwen234-idx1x128-column_pieces"),
    pytest.param((2, 128), QWEN_LARGE, QWEN_KWARGS, id="qwen204-idx2x128-multi_batch"),
    pytest.param((1, 3844), QWEN_LARGE, QWEN_KWARGS, id="qwen196-p4-idx1x3844-h_non_aligned"),
    pytest.param((8, 4096), QWEN_LARGE, QWEN_KWARGS, id="qwen194-p2-idx8x4096-multi_block"),
    pytest.param((16, 4096), QWEN_LARGE, QWEN_KWARGS, id="qwen192-p1-idx16x4096"),
    pytest.param((1, 512), FLUX_LARGE, FLUX_KWARGS, id="flux41-idx1x512-d2560"),
    pytest.param((1, 1), FLUX_43, FLUX_KWARGS, id="flux43-idx1x1-d5120"),
    pytest.param((1, 1), FLUX_44, FLUX_KWARGS, id="flux44-idx1x1-rank3_weight"),
]


@pytest.mark.parametrize("index_shape,weight_shape,kind", SHAPE_CASES)
def test_g6_embedding(device, index_shape, weight_shape, kind):
    weight = _table(weight_shape)
    torch.manual_seed(42)
    index = _index(index_shape, weight_shape[-2])
    _run_and_check(device, index, weight, kind)


def _with_special_rows(weight, index):
    """A copy of `weight` whose rows named by `index` hold SPECIAL_BITS, shifted per row so every
    column sees every pattern. Written through an int16 view: no float conversion touches them."""
    special = weight.clone()
    table = _table_2d(special)
    rows = torch.unique(index.to(torch.int64))
    bits = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    width = table.shape[-1]
    pattern = bits[(torch.arange(width).unsqueeze(0) + rows.unsqueeze(1)) % len(bits)]
    table.view(torch.int16)[rows] = pattern
    return special


@pytest.mark.parametrize(
    "index_shape,weight_shape",
    [
        pytest.param((1, 128), QWEN_SMALL, id="qwen233-d64"),
        pytest.param((1, 128), QWEN_LARGE, id="qwen234-d1280"),
        pytest.param((16, 1024), QWEN_LARGE, id="qwen198-p3-d1280-whole_rows"),
    ],
)
def test_special_values_bit_exact(device, index_shape, weight_shape):
    """-0.0, subnormals, infinities and NaN patterns must come back unchanged (spec rule 2)."""
    torch.manual_seed(42)
    index = _index(index_shape, weight_shape[-2])
    weight = _with_special_rows(_table(weight_shape), index)
    _run_and_check(device, index, weight, QWEN_KWARGS)


def test_new_indices_same_program(device):
    """Each call reads its own index values on the device (spec rule 1, BB-06, BB-07)."""
    index_shape, weight_shape = (1, 4096), QWEN_SMALL
    weight = _table(weight_shape)
    tt_weight = _device_table(weight, device)
    torch.manual_seed(42)
    first = _index(index_shape, weight_shape[-2])
    second = _index(index_shape, weight_shape[-2])
    for index in (first, second, first):
        tt_index = _device_index(index, device)
        output = g6_embedding(tt_index, tt_weight, **_call_kwargs(QWEN_KWARGS))
        _assert_bit_exact(output, _reference(index, weight))
        ttnn.deallocate(output)
        ttnn.deallocate(tt_index)
    ttnn.deallocate(tt_weight)


def test_first_and_last_table_row(device):
    """Indices alternate 0 and V-1: an off-by-one page id at either end of the table."""
    index_shape, weight_shape = (1, 128), QWEN_LARGE
    vocab = weight_shape[-2]
    index = torch.zeros(index_shape, dtype=torch.int32)
    index[..., 1::2] = vocab - 1
    _run_and_check(device, index, _table(weight_shape), QWEN_KWARGS)


def test_repeated_index(device):
    """One row at every position: each position gets its own copy."""
    index_shape, weight_shape = (1, 128), QWEN_SMALL
    torch.manual_seed(42)
    row = int(torch.randint(0, weight_shape[-2], (1,)))
    index = torch.full(index_shape, row, dtype=torch.int32)
    _run_and_check(device, index, _table(weight_shape), QWEN_KWARGS)


_QWEN_INDEX_SHAPES = [
    (16, 4096),
    (8, 4096),
    (1, 3844),
    (16, 1024),
    (1, 16),
    (16, 128),
    (2, 128),
    (1, 64),
    (1, 31),
    (1, 33),
    (1, 256),
    (1, 77),
    (1, 1000),
    (1, 129),
    (8, 1024),
    (1, 8),
    (8, 128),
    (1, 4096),
    (1, 100),
    (1, 1024),
    (1, 1),
    (1, 128),
]
PRODUCTION_PAIRS = [(i, w) for i in _QWEN_INDEX_SHAPES for w in (QWEN_SMALL, QWEN_LARGE)] + [
    ((1, 512), FLUX_LARGE),
    ((1, 256), FLUX_LARGE),
    ((1, 1), FLUX_43),
    ((1, 1), FLUX_44),
    ((1,), FLUX_LARGE),
]


def _pair_id(pair):
    index_shape, weight_shape = pair
    return "idx" + "x".join(map(str, index_shape)) + "-w" + "x".join(map(str, weight_shape))


@pytest.mark.parametrize("index_shape,weight_shape", PRODUCTION_PAIRS, ids=[_pair_id(p) for p in PRODUCTION_PAIRS])
def test_plan_invariants(device, index_shape, weight_shape):
    """The plan every production shape selects: within the L1 bound (BB-05), pieces cover D exactly,
    the grid is filled when the tile rows alone can fill it, and a wide one-tile-row result is not
    left on one core (spec rule 6)."""
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    plan = select_plan(index_shape, weight_shape, grid)
    d_tiles = weight_shape[-1] // 32
    batch = index_shape[0] if len(index_shape) == 2 else 1
    tile_rows = batch * -(-index_shape[-1] // 32)
    assert plan.piece_count * plan.block_width_tiles == d_tiles
    assert 1 <= plan.num_cores <= grid_cores
    assert plan.l1_bytes_per_core <= L1_BOUND_BYTES
    if tile_rows >= grid_cores:
        assert plan.num_cores == grid_cores
    if tile_rows == 1 and d_tiles >= 40:
        assert plan.num_cores > 1


def test_runtime_rows_true_refused(device, expect_error):
    """runtime_rows=True is outside TARGET: refused as an unsupported axis value."""
    torch.manual_seed(42)
    weight = torch.randn(FLUX_44, dtype=torch.bfloat16)
    index = torch.zeros((1, 1), dtype=torch.int32)
    with expect_error(UnsupportedAxisValue, "runtime_rows"):
        g6_embedding(_device_index(index, device), _device_table(weight, device), runtime_rows=True)


# Each refusal names the input it rejects (op_design.md, Host Contract).
INVALID_INPUT_CASES = {
    "weight_leading_extent_2": "weight",
    "d_not_multiple_of_32": "32",
    "index_int32": "uint32",
    "index_rank3": "rank",
    "l1_memory_config": "memory_config",
}


@pytest.mark.parametrize("case", list(INVALID_INPUT_CASES))
def test_invalid_inputs_refused(device, expect_error, case):
    torch.manual_seed(42)
    weight_shape = {"weight_leading_extent_2": (2, 256, 128), "d_not_multiple_of_32": (256, 48)}.get(case, FLUX_44)
    index_shape = (1, 1, 1) if case == "index_rank3" else (1, 1)
    weight = torch.randn(weight_shape, dtype=torch.bfloat16)
    index = torch.zeros(index_shape, dtype=torch.int32)
    tt_index = _device_index(index, device, ttnn.int32 if case == "index_int32" else ttnn.uint32)
    tt_weight = _device_table(weight, device)
    kwargs = {"memory_config": ttnn.L1_MEMORY_CONFIG} if case == "l1_memory_config" else {}
    with expect_error((ValueError, RuntimeError), INVALID_INPUT_CASES[case]):
        g6_embedding(tt_index, tt_weight, **kwargs)
