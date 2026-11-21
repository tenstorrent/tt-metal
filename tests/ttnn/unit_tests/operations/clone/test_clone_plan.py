# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests op_design.md requires beyond the acceptance test.

- Every thread of the plan ran (a sentinel-prefilled output; a missing hart leaves its stride unwritten).
- Device-session order: a tt-metal program before clone, and one after it (the sticky trid hand-back).
- Placement (R3), a fresh buffer (R5), and refusals (X1, X2, R7, R8) the cells do not cover.
"""

import pytest
import torch

import ttnn
from ttnn.operations.clone import clone
from ttnn.operations.clone.clone_program_artifacts import create_program_artifacts, split_pages

SENTINEL = -3.0e38  # bf16-representable, absent from a standard-normal input


def _put(t, layout, mem=ttnn.DRAM_MEMORY_CONFIG, device=None):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


def _run_with_plan(device, torch_input, layout, threads, grid=None, in_mem=ttnn.DRAM_MEMORY_CONFIG):
    tt_in = _put(torch_input, layout, in_mem, device)
    tt_out = _put(torch.full(torch_input.shape, SENTINEL, dtype=torch.bfloat16), layout, ttnn.L1_MEMORY_CONFIG, device)
    spec, run_args, idx = create_program_artifacts(tt_in, tt_out, threads=threads, grid=grid)
    return ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], spec, run_args, idx))


@pytest.mark.parametrize("R", [1, 2, 3])
@pytest.mark.parametrize(
    "shape,layout",
    [
        ((1, 1, 224, 32), ttnn.TILE_LAYOUT),  # 7 tiles on one cluster: not divisible by 2 or 3
        ((1, 1, 7, 64), ttnn.ROW_MAJOR_LAYOUT),  # 7 sticks on one cluster
    ],
    ids=["tile7", "rm7"],
)
def test_every_thread_ran_one_cluster(device, shape, layout, R):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    got = _run_with_plan(device, x, layout, (R, R), grid=ttnn.CoreCoord(1, 1))
    assert not (got == SENTINEL).any(), "a hart's stride of pages was never written"
    assert torch.equal(got, x)


@pytest.mark.parametrize("R", [1, 2, 3])
@pytest.mark.parametrize(
    "shape,layout",
    [((1, 1, 400, 64), ttnn.ROW_MAJOR_LAYOUT), ((1, 1, 400, 256), ttnn.TILE_LAYOUT), ((2, 1, 100, 47), ttnn.ROW_MAJOR_LAYOUT)],
    ids=["rm400x64", "tile400x256", "rm2x100x47"],
)
def test_every_thread_ran_full_grid(device, shape, layout, R):
    torch.manual_seed(1)
    x = torch.randn(shape, dtype=torch.bfloat16)
    got = _run_with_plan(device, x, layout, (R, R))
    assert not (got == SENTINEL).any()
    assert torch.equal(got, x)


def test_split_deals_whole_rounds(device):
    grid = device.compute_with_storage_grid_size()
    for pages, R in [(400, 1), (400, 3), (7, 3), (104, 2), (1, 3)]:
        cores, counts, starts = split_pages(pages, grid, R)
        assert sum(counts.values()) == pages
        ordered = sorted(counts, key=lambda c: starts[c])
        assert [starts[c] for c in ordered] == [sum(counts[d] for d in ordered[:i]) for i in range(len(ordered))]
        # only the last engaged cluster may end on a short round
        assert all(counts[c] % R == 0 for c in ordered[:-1])
        assert len(counts) <= grid.x * grid.y


def test_tt_metal_program_before_clone(device):
    """A tt-metal program (untilize) runs first in this session, then clone."""
    torch.manual_seed(2)
    y = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    _ = ttnn.to_torch(ttnn.to_layout(_put(y, ttnn.TILE_LAYOUT, device=device), ttnn.ROW_MAJOR_LAYOUT))
    x = torch.randn((1, 1, 400, 512), dtype=torch.bfloat16)
    out = clone(_put(x, ttnn.TILE_LAYOUT, device=device), dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(out), x)


@pytest.mark.parametrize(
    "layout,in_mem",
    [(ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG), (ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG)],
    ids=["tile_dram", "rm_l1"],
)
def test_tt_metal_program_after_clone(device, layout, in_mem):
    """clone, then a tt-metal program on its output (untagged NoC transfers) in the same session."""
    torch.manual_seed(3)
    shape = (1, 1, 1600, 512) if layout == ttnn.TILE_LAYOUT else (1, 1, 400, 64)
    x = torch.randn(shape, dtype=torch.bfloat16)
    out = clone(_put(x, layout, in_mem, device), memory_config=ttnn.L1_MEMORY_CONFIG)
    other = ttnn.ROW_MAJOR_LAYOUT if layout == ttnn.TILE_LAYOUT else ttnn.TILE_LAYOUT
    assert torch.equal(ttnn.to_torch(ttnn.from_device(ttnn.to_layout(out, other))), x)
    # and clone again after it
    out2 = clone(out, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(out2), x)


@pytest.mark.parametrize("in_mem", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_placement_and_fresh_buffer(device, in_mem):
    """R3: the output is L1 interleaved; R5: it is a new buffer, also for an L1 input."""
    torch.manual_seed(4)
    x = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    tt_in = _put(x, ttnn.TILE_LAYOUT, in_mem, device)
    out = clone(tt_in, memory_config=ttnn.L1_MEMORY_CONFIG)
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert out.buffer_address() != tt_in.buffer_address()
    assert torch.equal(ttnn.to_torch(out), x)
    assert torch.equal(ttnn.to_torch(tt_in), x)  # input untouched


def test_refusals(device):
    x = _put(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(TypeError):  # X1
        clone(x, memory_config="L1")
    with pytest.raises(NotImplementedError):  # X2
        clone(x, dtype=ttnn.float32)
    with pytest.raises(NotImplementedError):  # R8: not in Phase 0
        clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with pytest.raises(NotImplementedError):  # R8: memory_config=None is not in Phase 0
        clone(x)


def test_sharded_refused_by_name(device):
    """R7: a sharded memory_config argument, and a sharded input, are refused."""
    x_torch = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    x = _put(x_torch, ttnn.TILE_LAYOUT, device=device)
    sharded = ttnn.create_sharded_memory_config(
        (1, 1, 64, 64), ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.HEIGHT, use_height_and_width_as_shard_shape=False
    )
    with pytest.raises(NotImplementedError, match="output_memory"):
        clone(x, memory_config=sharded)
    try:
        xs = _put(x_torch, ttnn.TILE_LAYOUT, sharded, device)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot build a sharded input on this backend: {e}")
    with pytest.raises(NotImplementedError, match="sharding"):
        clone(xs, memory_config=ttnn.L1_MEMORY_CONFIG)
