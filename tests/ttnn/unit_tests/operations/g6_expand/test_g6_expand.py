# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_expand: x (1, 1, 1, W) -> (1, 1, W, W), bit-exact.

The op is an exact copy (helpers.TOLERANCES[bfloat16] == ("exact", None)), so every
value check compares 16-bit patterns; a PCC gate would pass a nearly right row.
Cases are limited to the five production widths and to what the spec's rules require
(BB-22): bit identity on special values, independence from x's tile padding, repeat
determinism, the program cache across widths, plan reporting (BB-08) and host-side
validation. Single-tile, non-square and multi-batch shapes are not production calls.
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.g6_expand import g6_expand

PRODUCTION_WIDTHS = [512, 256, 196, 185, 207]
L1_OP_BUDGET = 1347776

# NaN (canonical; quiet and signalling payloads; both signs), +-inf, -0.0, +0.0,
# denormals of both signs, largest finite values.
SPECIAL_BF16_BITS = [
    0x7FC0,
    0x7FA1,
    0xFFC5,
    0x7F81,
    0x7F80,
    0xFF80,
    0x8000,
    0x0000,
    0x0001,
    0x0040,
    0x807F,
    0x7F7F,
    0xFF7F,
]


def torch_g6_expand(x, sizes):
    return x.expand(*sizes).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _row(width, *, special_values=False):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    if special_values:
        n = len(SPECIAL_BF16_BITS)
        pattern = torch.tensor(
            [v - (1 << 16) if v >= (1 << 15) else v for v in SPECIAL_BF16_BITS], dtype=torch.int16
        ).view(torch.bfloat16)
        # Both ends of the row: the first tile and the last (partial when W % 32 != 0).
        x[..., :n] = pattern
        x[..., width - n :] = pattern
    return x


def _to_device(x, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _check(out, expected, x_tt):
    assert list(out.shape) == list(expected.shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    memory_config = out.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.buffer_address() != x_tt.buffer_address(), "out must own its storage, not alias x"
    got = ttnn.to_torch(out)
    assert got.dtype == torch.bfloat16
    diff = _bits(got) != _bits(expected)
    if bool(diff.any()):
        first = [int(i) for i in diff.nonzero()[0]]
        pytest.fail(
            f"{int(diff.sum())} of {diff.numel()} elements differ in bits; first at {first}: "
            f"got {got[tuple(first)].item()!r} expected {expected[tuple(first)].item()!r}"
        )


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_g6_expand_production(device, width):
    x = _row(width)
    sizes = [1, 1, width, width]
    x_tt = _to_device(x, device)
    out = g6_expand(x_tt, sizes)
    _check(out, torch_g6_expand(x, sizes), x_tt)


def test_g6_expand_sizes_tuple(device):
    width = 256
    x = _row(width)
    sizes = (1, 1, width, width)
    x_tt = _to_device(x, device)
    out = g6_expand(x_tt, sizes)
    _check(out, torch_g6_expand(x, sizes), x_tt)


def test_g6_expand_sizes_minus_one(device):
    width = 196
    x = _row(width)
    sizes = [-1, -1, width, -1]
    x_tt = _to_device(x, device)
    out = g6_expand(x_tt, sizes)
    _check(out, torch_g6_expand(x, sizes), x_tt)


@pytest.mark.parametrize("width", [512, 196])
def test_g6_expand_special_values(device, width):
    x = _row(width, special_values=True)
    sizes = [1, 1, width, width]
    x_tt = _to_device(x, device)
    if not torch.equal(_bits(ttnn.to_torch(x_tt)), _bits(x)):
        pytest.fail("harness: ttnn.from_torch did not carry the bfloat16 bit patterns unchanged")
    out = g6_expand(x_tt, sizes)
    _check(out, torch_g6_expand(x, sizes), x_tt)


@pytest.mark.parametrize("width", [512, 196])
def test_g6_expand_poisoned_input_padding(device, width):
    x = _row(width)
    sizes = [1, 1, width, width]
    # Rows 1-31 of every tile of x and its columns beyond W hold a loud value; the
    # expected output comes from the logical row only.
    x_tt = ttnn.fill_implicit_tile_padding(_to_device(x, device), 1000.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = g6_expand(x_tt, sizes)
    _check(out, torch_g6_expand(x, sizes), x_tt)


def test_g6_expand_repeat_determinism(device):
    width = 512
    x = _row(width)
    sizes = [1, 1, width, width]
    x_tt = _to_device(x, device)
    first = g6_expand(x_tt, sizes)
    second = g6_expand(x_tt, sizes)
    expected = torch_g6_expand(x, sizes)
    _check(first, expected, x_tt)
    _check(second, expected, x_tt)
    assert torch.equal(_bits(ttnn.to_torch(first)), _bits(ttnn.to_torch(second)))


def test_g6_expand_program_cache_across_widths(device):
    order = [512, 196, 512, 185, 207, 256]
    rows = {width: _row(width) for width in set(order)}
    inputs = {width: _to_device(row, device) for width, row in rows.items()}
    outputs = [(width, g6_expand(inputs[width], [1, 1, width, width])) for width in order]
    entries = device.num_program_cache_entries()
    repeats = [(width, g6_expand(inputs[width], [1, 1, width, width])) for width in order]
    assert device.num_program_cache_entries() == entries, "a repeat call with an identical key added a program"
    for width, out in outputs + repeats:
        _check(out, torch_g6_expand(rows[width], [1, 1, width, width]), inputs[width])


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_g6_expand_plan_report(device, width):
    from ttnn.operations.g6_expand import g6_expand_plan

    x_tt = _to_device(_row(width), device)
    sizes = [1, 1, width, width]
    plan = g6_expand_plan(x_tt, sizes)
    assert plan == g6_expand_plan(x_tt, sizes), "the plan must be a function of the call"
    assert plan["regime"] == "column_group_broadcast"

    grid = device.compute_with_storage_grid_size()
    src_col_tiles = math.ceil(width / 32)
    out_row_tiles = math.ceil(width / 32)
    num_col_groups, num_row_groups = plan["num_col_groups"], plan["num_row_groups"]
    core_src_col_tiles, core_out_row_tiles = plan["core_src_col_tiles"], plan["core_out_row_tiles"]
    assert (num_col_groups - 1) * core_src_col_tiles < src_col_tiles <= num_col_groups * core_src_col_tiles
    assert (num_row_groups - 1) * core_out_row_tiles < out_row_tiles <= num_row_groups * core_out_row_tiles
    assert 1 <= plan["block_src_col_tiles"] <= core_src_col_tiles
    assert 1 <= plan["block_out_row_tiles"] <= core_out_row_tiles
    assert plan["num_cores"] == num_col_groups * num_row_groups <= grid.x * grid.y
    assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET

    roles = plan["core_roles"]
    assert len(roles) == plan["num_cores"]
    assert set(roles.values()) <= {"sender", "receiver"}
    assert sum(1 for role in roles.values() if role == "sender") == num_col_groups
    assert all(0 <= x < grid.x and 0 <= y < grid.y for x, y in roles)


@pytest.mark.parametrize(
    "sizes",
    [
        [1, 256, 256],
        [1, 1, -2, 256],
        [1, 1, 0, 256],
        [1, 1, 256, 128],
        [2, 1, 256, 256],
        [1, 3, 256, 256],
    ],
    ids=["length", "negative", "zero", "mismatch_on_non_unit_dim", "expand_dim0", "expand_dim1"],
)
def test_g6_expand_rejects_bad_sizes(device, sizes, expect_error):
    x_tt = _to_device(_row(256), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(x_tt, sizes)


def test_g6_expand_rejects_rank_3(device, expect_error):
    x_tt = ttnn.from_torch(
        torch.randn((1, 1, 256), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(x_tt, [1, 256, 256])


def test_g6_expand_rejects_host_tensor(expect_error):
    x_host = ttnn.from_torch(_row(256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(x_host, [1, 1, 256, 256])


def test_g6_expand_rejects_l1_input(device, expect_error):
    x_tt = _to_device(_row(256), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), ""):
        g6_expand(x_tt, [1, 1, 256, 256])
