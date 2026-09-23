# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 5: the grid_2d_split regime (short_wide / square_large) and low_l1.

- The assignment rule (host-only, no device): makespan, tie-breaks, alignment units.
- Bit-exact tilize on the work-geometry shapes, with the Tensix core count the
  program descriptor reaches (must be ~ the full grid on short_wide / square_large).
- low_l1=True vs low_l1=False readbacks are bit-identical (same data path, smaller CBs).
- A 2-D split program hits the program cache on fresh allocations.
"""

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import (
    CB_BUDGET_BYTES,
    create_program_descriptor,
    grid_2d_split,
)


# ---------------------------------------------------------------- host rule


@pytest.mark.parametrize(
    "R, C, N, align, expect",
    [
        (1, 64, 64, 1, (1, 64)),  # short_wide_canonical: every tile-column on its own core
        (1, 256, 64, 1, (1, 64)),  # short_wide_l1_forcing: 4 tile-columns per core
        (2, 128, 64, 1, (2, 32)),  # tie on makespan -> wider column groups
        (64, 64, 64, 1, (64, 1)),  # square_large on 64 cores: row split already optimal (tie -> wide)
        (64, 64, 130, 1, (64, 2)),  # the same on a Blackhole-sized grid: 2-D
        (512, 2, 64, 1, (64, 1)),  # perf-focus shape keeps the row split
        (64, 2, 130, 1, (64, 2)),  # tall_narrow_grid_scale with R < N
        (1, 64, 64, 2, (1, 32)),  # col_align_tiles = 2: columns move in pairs
    ],
)
def test_grid_2d_rule(R, C, N, align, expect):
    assert grid_2d_split(R, C, N, col_align_tiles=align, min_group_col_tiles=1) == expect


def test_grid_2d_rule_row_align():
    # retile: row units of 2 tile-rows, R = 4 -> at most 2 row groups
    g_r, g_c = grid_2d_split(4, 64, 64, col_align_tiles=1, row_align=2, min_group_col_tiles=1)
    assert g_r <= 2 and g_r * g_c <= 64


# ---------------------------------------------------------------- device


def _rm(device, x, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)


def _num_cores(t, *, low_l1=False):
    """Tensix cores the program descriptor dispatches on for an unpadded interleaved call."""
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(t.shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, t.device(), t.memory_config()
    )
    desc = create_program_descriptor(t, out, low_l1=low_l1)
    return desc.kernels[0].core_ranges.num_cores()


WORK_GEOMETRY = [
    pytest.param([1, 1, 32, 2048], True, id="short_wide_canonical"),
    pytest.param([1, 1, 64, 4096], True, id="short_wide_two_tile_rows"),
    pytest.param([1, 1, 32, 8192], True, id="short_wide_l1_forcing"),
    pytest.param([1, 1, 2048, 2048], True, id="square_large"),
    pytest.param([1, 1, 2048, 64], True, id="tall_narrow_grid_scale"),
    pytest.param([1, 1, 96, 64], False, id="small_3x2"),
    pytest.param([2, 3, 64, 1024], True, id="rank4_fold_12x32"),
]


@pytest.mark.parametrize("shape, fills_grid", WORK_GEOMETRY)
def test_grid_2d_bit_exact(device, shape, fills_grid):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = _rm(device, x)
    grid = device.compute_with_storage_grid_size()
    n = _num_cores(t)
    print(f"{shape}: {n} of {grid.x * grid.y} Tensix cores")
    if fills_grid:
        assert n >= (grid.x * grid.y) * 3 // 4, f"{shape} reaches {n} of {grid.x * grid.y} Tensix cores"
    out = tilize(t)
    assert out.layout == ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(out), x)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param([1, 1, 1, 2048], id="short_wide_single_stick"),
        pytest.param([1, 1, 32, 4090], id="short_wide_w_tail"),
        pytest.param([8, 1, 249, 2048], id="square_large_from_leading_dims"),
    ],
)
def test_grid_2d_padded(device, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    out = tilize(_rm(device, x), pad_value=0.0)
    assert torch.equal(ttnn.to_torch(out), x)


def test_grid_2d_l1_interleaved(device):
    torch.manual_seed(0)
    x = torch.randn([1, 1, 32, 2048], dtype=torch.float32).to(torch.bfloat16)
    out = tilize(_rm(device, x, ttnn.L1_MEMORY_CONFIG), ttnn.L1_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(out), x)


@pytest.mark.parametrize(
    "shape, pad",
    [
        pytest.param([1, 1, 64, 256], False, id="low_l1_request"),
        pytest.param([1, 1, 32, 4096], False, id="short_wide_low_l1"),
        pytest.param([1, 1, 32, 8192], False, id="low_l1_forcing_width"),
        pytest.param([1, 1, 50, 50], True, id="padded_low_l1"),
        pytest.param([1, 1, 4096, 1024], False, id="tall_wide_low_l1"),
    ],
)
def test_low_l1_ab_bit_identical(device, shape, pad):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    kwargs = {"pad_value": 0.0} if pad else {}
    t = _rm(device, x)
    a = ttnn.to_torch(tilize(t, **kwargs))
    b = ttnn.to_torch(tilize(t, low_l1=True, **kwargs))
    assert torch.equal(a, x)
    assert torch.equal(a, b), "low_l1 must not change the result"


def test_low_l1_bounds_the_cbs(device):
    x = torch.zeros([1, 1, 4096, 1024], dtype=torch.bfloat16)
    t = _rm(device, x)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 4096, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    for low_l1 in (False, True):
        desc = create_program_descriptor(t, out, low_l1=low_l1)
        total = sum(cb.total_size for cb in desc.cbs)
        assert total <= CB_BUDGET_BYTES[low_l1], f"low_l1={low_l1}: {total} bytes of CBs"


@pytest.mark.parametrize("shape", [[1, 1, 32, 2048], [1, 1, 2048, 2048]], ids=["short_wide", "square_large"])
def test_grid_2d_program_cache(device, shape):
    device.enable_program_cache()
    n0 = device.num_program_cache_entries()
    keep = []
    first = None
    for i in range(3):
        x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        t = _rm(device, x)
        out = tilize(t)
        keep += [t, out]
        assert torch.equal(ttnn.to_torch(out), x)
        delta = device.num_program_cache_entries() - n0
        if i == 0:
            first = delta
            assert first <= 1
        else:
            assert delta == first, f"call {i + 1} added a program"
