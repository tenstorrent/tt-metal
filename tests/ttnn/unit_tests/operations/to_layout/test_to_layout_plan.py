# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests op_design.md's Work Distribution asks for, beside the golden suite:

1. the thread plan read off every built ProgramSpec;
2. the op in one device session with tt-metal's own programs, before and after them (ring protocol hygiene:
   an untagged transfer or a ring id left in a hart's tag register only shows across programs);
3. our three regimes back to back in one session;
4. a host unit test of select_block_width_tiles on the design's worked table, plus the 2-D work split.
"""

import pytest
import torch

import ttnn
from ttnn.operations.to_layout import to_layout
from ttnn.operations.to_layout import to_layout_program_artifacts as pa

TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def _put(x, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)


def _exact(t, expected):
    got = ttnn.to_torch(t)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert torch.equal(got, expected), (got.float() - expected.float()).abs().max()


@pytest.mark.parametrize("src,dst", [(RM, TILE), (TILE, RM), (TILE, TILE), (RM, RM)], ids=["tilize", "untilize", "copy_tile", "copy_rm"])
def test_thread_plan_on_every_kernel(device, src, dst):
    """Every KernelSpec of the built program carries THREAD_PLAN's count for its role."""
    x = torch.randn(1, 1, 400, 512, dtype=torch.bfloat16)
    t = _put(x, src, device)
    out_spec = ttnn.TensorSpec(t.shape, t.dtype, dst, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(out_spec, device)
    spec, _, _ = pa.create_program_artifacts(t, out)
    R, C, W = pa.THREAD_PLAN
    want = {"reader": R, "compute": C, "writer": W}
    kernels = {k.unique_id: k for k in spec.kernels}
    assert set(kernels) == ({"reader", "writer"} if src == dst else {"reader", "compute", "writer"})
    for name, k in kernels.items():
        assert k.num_threads == want[name], (name, k.num_threads)
        assert k.compile_time_args[f"{name}_threads"] == want[name]


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["DRAM", "L1"])
def test_one_session_with_tt_metal_programs(device, memory_config):
    """tt-metal to_layout -> ours (each regime) -> tt-metal to_layout of our output, values checked each step."""
    torch.manual_seed(1)
    shape = (1, 1, 400, 512)  # h_non_aligned, column split: bw = 4 on 32 clusters
    x = torch.randn(shape, dtype=torch.bfloat16)

    # A tt-metal program first (an untilize: explicit sync, untagged transfers), then our tilize.
    rm = ttnn.to_layout(_put(x, TILE, device, memory_config), RM)
    _exact(rm, x)
    ours_tile = to_layout(rm, TILE)
    assert ours_tile.layout == TILE
    assert ours_tile.memory_config().buffer_type == memory_config.buffer_type  # R5
    _exact(ours_tile, x)
    # tt-metal's untilize after our tilize.
    _exact(ttnn.to_layout(ours_tile, RM), x)

    # Our untilize, then tt-metal's tilize of our row-major output and its untilize again.
    ours_rm = to_layout(ours_tile, RM)
    assert ours_rm.layout == RM
    assert ours_rm.memory_config().buffer_type == memory_config.buffer_type
    _exact(ours_rm, x)
    back = ttnn.to_layout(ours_rm, TILE)
    _exact(back, x)
    _exact(ttnn.to_layout(back, RM), x)

    # Our identity copies, each followed by a tt-metal program.
    same_tile = to_layout(back, TILE)
    _exact(same_tile, x)
    _exact(ttnn.to_layout(same_tile, RM), x)
    same_rm = to_layout(ours_rm, RM)
    _exact(same_rm, x)
    _exact(ttnn.to_layout(same_rm, TILE), x)


def test_our_regimes_back_to_back(device):
    """Our tilize, untilize and identity copies, one after another in one session, on several shapes."""
    torch.manual_seed(2)
    for shape in [(1, 1, 1600, 256), (4, 8, 47, 256), (2, 4, 128, 512), (1, 1, 32, 4096)]:
        x = torch.randn(shape, dtype=torch.bfloat16)
        t = to_layout(_put(x, RM, device), TILE)
        _exact(t, x)
        r = to_layout(t, RM)
        _exact(r, x)
        _exact(to_layout(t, TILE), x)
        _exact(to_layout(r, RM), x)


# op_design.md, "Worked Phase 0 values (Quasar, 32 clusters, bf16)": shape -> (total_tile_rows, tensor_w_tiles, bw)
WORKED = {
    (1, 1, 400, 512): (13, 16, 4),
    (1, 1, 1600, 256): (50, 8, 8),
    (1, 1, 32, 64): (1, 2, 2),
    (1, 1, 17, 64): (1, 2, 2),
    (1, 1, 64, 128): (2, 4, 4),
    (1, 1, 50, 128): (2, 4, 4),
    (4, 8, 32, 256): (32, 8, 8),
    (2, 4, 128, 512): (32, 16, 16),
    (1, 1, 2048, 256): (64, 8, 8),
    (4, 8, 47, 256): (64, 8, 8),
    (4, 1, 512, 512): (64, 16, 16),
    (1, 1, 32, 4096): (1, 128, 4),
    (1, 1, 32, 8192): (1, 256, 8),
    (1, 1, 128, 4096): (4, 128, 16),
    (2, 1, 64, 4096): (4, 128, 16),
}


@pytest.mark.parametrize("shape", list(WORKED), ids=lambda s: "x".join(map(str, s)))
def test_select_block_width_tiles_worked_table(shape):
    rows, wt, bw = WORKED[shape]
    n_images = shape[0] * shape[1]
    assert n_images * -(-shape[2] // 32) == rows and -(-shape[3] // 32) == wt
    got = pa.select_block_width_tiles(
        total_row_bands=-(-rows // pa.BLOCK_HEIGHT_TILES),
        tensor_w_tiles=wt,
        tile_bytes=2048,
        elem_size=2,
        num_cores=32,
        dram_alignment=64,
        l1_budget=4 * 1024 * 1024 - pa.L1_RESERVE_BYTES,
    )
    assert got == bw
    assert wt % got == 0  # no ragged column block (Quasar tilize MOP width = init width)


def test_split_covers_every_unit_once(device):
    """The 2-D split hands out every (band, column block) unit exactly once, in contiguous ranges."""
    grid = device.compute_with_storage_grid_size()
    for units in (1, 13, 32, 52, 64, 100):
        _, counts, starts = pa.split_units(units, grid)
        covered = sorted(u for core in counts for u in range(starts[core], starts[core] + counts[core]))
        assert covered == list(range(units))
        assert len(counts) == min(units, grid.x * grid.y)
