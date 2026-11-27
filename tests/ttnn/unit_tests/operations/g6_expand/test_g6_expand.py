# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_expand: x (1, 1, 1, W) -> (1, 1, W, W), every logical element bit-identical.

The op computes nothing, so the gate is bit equality on the int16 view of the bfloat16 values
(NaN payloads, signed zeros and denormals included), never a PCC / RMS tolerance.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand, g6_expand_with_plan, select_plan

PRODUCTION_WIDTHS = [512, 256, 196, 185, 207]
L1_OP_BUDGET_BYTES = 1347776

# NaN with payloads (quiet and signalling, both signs), +-inf, -0.0, denormals, largest finite.
SPECIAL_BF16_BITS = [0x7FC0, 0x7FA1, 0xFFC5, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x0040, 0x7F7F, 0xFF7F]


def torch_expand(x, sizes):
    return x.expand(*[int(s) for s in sizes]).contiguous()


def _bits(t):
    return t.contiguous().view(torch.int16)


def make_row(width, *, special_values=False):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
    if special_values:
        n = len(SPECIAL_BF16_BITS)
        row = torch.tensor([v - (1 << 16) if v >= (1 << 15) else v for v in SPECIAL_BF16_BITS], dtype=torch.int16).view(
            torch.bfloat16
        )
        x[..., :n] = row
        x[..., width - n :] = row
    return x


def to_device(x, device, *, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def check_bit_exact(out, expected, x_tt):
    assert list(out.shape) == list(expected.shape)
    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    mem = out.memory_config()
    assert mem.buffer_type == ttnn.BufferType.DRAM
    assert mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.buffer_address() != x_tt.buffer_address(), "out must own its storage, not alias the input"
    got = ttnn.to_torch(out)
    assert got.dtype == torch.bfloat16
    diff = _bits(got) != _bits(expected)
    if bool(diff.any()):
        first = diff.nonzero()[0].tolist()
        pytest.fail(f"{int(diff.sum())} of {diff.numel()} elements differ in bits; first at {first}")
    return got


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_production(device, width):
    x = make_row(width)
    sizes = [1, 1, width, width]
    x_tt = to_device(x, device)
    out = g6_expand(x_tt, sizes)
    check_bit_exact(out, torch_expand(x, sizes), x_tt)


@pytest.mark.parametrize(
    "width,sizes",
    [(256, (1, 1, 256, 256)), (512, [-1, -1, 512, -1]), (196, (-1, 1, 196, -1))],
    ids=["tuple", "minus_one_list", "minus_one_tuple"],
)
def test_sizes_forms(device, width, sizes):
    x = make_row(width)
    x_tt = to_device(x, device)
    out = g6_expand(x_tt, sizes)
    check_bit_exact(out, torch_expand(x, sizes), x_tt)


@pytest.mark.parametrize("width", [196, 512])
def test_special_values_bit_identity(device, width):
    x = make_row(width, special_values=True)
    x_tt = to_device(x, device)
    if not torch.equal(_bits(ttnn.to_torch(x_tt)), _bits(x)):
        pytest.fail("harness: ttnn.from_torch did not carry the bfloat16 special bit patterns unchanged")
    sizes = [1, 1, width, width]
    out = g6_expand(x_tt, sizes)
    check_bit_exact(out, torch_expand(x, sizes), x_tt)


@pytest.mark.parametrize("width", [512, 196])
def test_input_padding_does_not_leak(device, width):
    # Rows 1..31 of every input tile and the columns beyond W hold a loud value; out must not see it.
    x = make_row(width)
    x_tt = ttnn.fill_implicit_tile_padding(to_device(x, device), 1000.0)
    sizes = [1, 1, width, width]
    out = g6_expand(x_tt, sizes)
    check_bit_exact(out, torch_expand(x, sizes), x_tt)


def test_repeat_and_program_cache(device):
    sequence = [512, 196, 512, 185, 207, 256]
    rows = {w: make_row(w) for w in sorted(set(sequence))}
    inputs = {w: to_device(rows[w], device) for w in rows}
    first = []
    for w in sequence:
        sizes = [1, 1, w, w]
        first.append(check_bit_exact(g6_expand(inputs[w], sizes), torch_expand(rows[w], sizes), inputs[w]))
    entries = device.num_program_cache_entries()
    for w, previous in zip(sequence, first):
        again = ttnn.to_torch(g6_expand(inputs[w], [1, 1, w, w]))
        assert torch.equal(_bits(again), _bits(previous)), f"W={w}: a repeated call returned different bits"
    assert device.num_program_cache_entries() == entries, "a repeated call built a new device program"


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_plan_report(device, width):
    x_tt = to_device(make_row(width), device)
    sizes = [1, 1, width, width]
    plan = select_plan(x_tt, sizes)
    assert plan == select_plan(x_tt, sizes), "plan selection must be deterministic"
    assert plan.regime == "column_group_bcast"
    grid = device.compute_with_storage_grid_size()
    assert 1 < plan.num_cores <= grid.x * grid.y
    assert plan.num_cores == plan.num_groups * plan.row_splits
    assert len(plan.core_roles) == plan.num_cores
    assert sum(role == "injector" for role in plan.core_roles.values()) == plan.num_groups
    assert plan.l1_bytes_per_core <= L1_OP_BUDGET_BYTES


# Knob settings the production plan does not reach on every grid: no multicast (row_splits = 1),
# multi-page groups (strips_per_column = 1), and several blocks per group with depth 2.
PINNED_PLANS = [
    (512, {"row_splits": 1}),
    (512, {"strips_per_column": 1}),
    (512, {"strips_per_column": 1, "block_pages": 1}),
    (196, {"row_splits": 1}),
]


@pytest.mark.parametrize(
    "width,overrides",
    PINNED_PLANS,
    ids=["512-no_mcast", "512-one_strip", "512-one_strip-multi_block", "196-no_mcast"],
)
def test_pinned_plans(device, width, overrides):
    x = make_row(width)
    x_tt = to_device(x, device)
    sizes = [1, 1, width, width]
    plan = select_plan(x_tt, sizes, **overrides)
    if "row_splits" in overrides:
        assert plan.row_splits == overrides["row_splits"]
    if "strips_per_column" in overrides:
        assert plan.strips_per_column == overrides["strips_per_column"]
    if "block_pages" in overrides:
        assert plan.block_pages == overrides["block_pages"]
        if plan.group_pages_max > plan.block_pages:
            assert plan.t_block_depth == 2
    out = g6_expand_with_plan(x_tt, sizes, plan)
    check_bit_exact(out, torch_expand(x, sizes), x_tt)


def test_validation_errors(device, expect_error):
    x = make_row(512)
    x_tt = to_device(x, device)
    bad_sizes = [
        [1, 512, 512],  # length differs from the rank of x
        [1, 1, 0, 512],  # a size of 0
        [1, 1, -2, 512],  # negative and not -1
        [1, 1, 512, 256],  # differs along a dim whose size is not 1
        [2, 1, 512, 512],  # expands dim 0
        [1, 2, 512, 512],  # expands dim 1
    ]
    for sizes in bad_sizes:
        with expect_error((ValueError, RuntimeError), "g6_expand"):
            g6_expand(x_tt, sizes)

    rank3 = to_device(torch.randn((1, 1, 512), dtype=torch.float32).to(torch.bfloat16), device)
    with expect_error((ValueError, RuntimeError), "g6_expand"):
        g6_expand(rank3, [1, 512, 512])

    on_host = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), "g6_expand"):
        g6_expand(on_host, [1, 1, 512, 512])

    in_l1 = to_device(x, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), "g6_expand"):
        g6_expand(in_l1, [1, 1, 512, 512])


def test_unsupported_cells_refused(device, expect_error):
    x = make_row(512)
    as_fp32 = to_device(x.to(torch.float32), device, dtype=ttnn.float32)
    with expect_error(NotImplementedError, "g6_expand"):
        g6_expand(as_fp32, [1, 1, 512, 512])

    x_tt = to_device(x, device)
    with expect_error(NotImplementedError, "g6_expand"):
        g6_expand(x_tt, [1, 1, 100, 512])  # h_non_aligned: outside TARGET
