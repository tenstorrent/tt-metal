# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Work-geometry coverage for `tilize` — the shapes the acceptance test does not
carry, and the perf-focus targets.

DO NOT DELETE. The `tile_grid` axis is trivially CORRECT under any distribution
scheme, so these cases are not here to prove the values; they are here so the
geometries the golden `work_geometry` group and `LOOSE_CASES` name are exercised
on device at all (an L1-forcing width, the two big `square_large` forms, and the
[1,1,32,16384] / [1,1,16384,32] transposed pair at an identical tile count).

Shapes with a non-tile-aligned H (the golden group's padded members —
[1,1,1,2048], [1,1,32,4090], [8,1,249,2048], [1,1,1,50304]) are Phase 0
support refusals, not failures, and are deliberately absent.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.tilize import tilize

PCC = 0.995

# (shape, id) — every one is bf16 / rank 4 / tile-aligned / interleaved DRAM,
# i.e. inside the Phase 0 rectangle, differing in GEOMETRY alone.
GEOMETRIES = [
    # --- golden work_geometry group (the tile-aligned members) --------------
    ((1, 1, 32, 2048), "short_wide_canonical"),
    ((1, 1, 64, 4096), "short_wide_two_tile_rows"),
    ((1, 1, 32, 8192), "short_wide_l1_forcing"),  # a full-width block would be 1 MB/side at fp32
    ((1, 1, 32, 4096), "short_wide_low_l1_shape"),
    ((1, 1, 2048, 64), "tall_narrow_grid_scale"),
    ((1, 1, 2048, 2048), "square_large"),
    # --- LOOSE_CASES perf-focus geometries ---------------------------------
    ((1, 1, 32, 16384), "perf_focus_attention"),  # R=1, C=512 — the mandatory target
    ((1, 1, 16384, 32), "perf_focus_transposed"),  # R=512, C=1 — identical tile count
    ((1, 1, 32, 32768), "short_wide_32768"),
    ((1, 1, 64, 12288), "short_wide_12288"),
    ((1, 1, 1024, 1024), "square_large_1024"),
    # --- leading-dim fold carrying a wide C --------------------------------
    ((8, 1, 32, 2048), "leading_fold_wide"),
]


@pytest.mark.parametrize("shape", [s for s, _ in GEOMETRIES], ids=[i for _, i in GEOMETRIES])
def test_tilize_geometry(device, shape):
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == ttnn.bfloat16
    assert list(tt_output.shape) == list(shape)
    assert_with_pcc(torch_input.float(), ttnn.to_torch(tt_output).float(), PCC)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 16384), (1, 1, 16384, 32)],
    ids=["perf_focus_attention", "perf_focus_transposed"],
)
def test_tilize_perf_focus(device, shape):
    """The mandatory perf target and its transposed counterpart at an identical
    tile count (512 tiles each). Run under `--profile` for the device kernel ns;
    the core count the distribution reached is asserted here so the number can
    never be reported off a fraction of the grid.
    """
    import ttnn.operations.tilize.tilize_program_descriptor as pd

    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(7)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)

    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    cores_reached = len(plan.assignment)
    print(
        f"\n[tilize perf] {shape}: R={plan.tensor_row_blocks} C={plan.tensor_col_tiles} "
        f"block={plan.tensor_row_blocks // plan.num_row_groups}x{plan.block_width_tiles} tiles, "
        f"num_w_chunks={plan.num_w_chunks} num_row_groups={plan.num_row_groups}, "
        f"cores={cores_reached}/{grid.x * grid.y}, "
        f"read={plan.block_row_bytes} B/stick, L1/core={plan.l1_per_core_bytes // 1024} KiB"
    )
    assert cores_reached == grid.x * grid.y, (
        f"tilize reached {cores_reached} of {grid.x * grid.y} cores on {shape} — "
        f"a duration measured here would describe the split, not the kernel"
    )
    assert_with_pcc(torch_input.float(), ttnn.to_torch(tt_output).float(), PCC)
