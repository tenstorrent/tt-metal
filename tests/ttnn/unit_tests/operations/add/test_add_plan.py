# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design (op_design.md, Work Distribution) requires beyond the acceptance cells:

1. the thread plans (1,1,1), (2,2,2), (4,4,2), (1,4,1) on a shape where every NEO of every engaged cluster
   gets pages; the output is pre-filled with NaN, so a thread that did not run leaves NaN pages behind;
2. one device session: a tt-metal program before add, add twice, a tt-metal program (to_layout) after;
3. block_tiles in {1, 2, 4} with ragged per-NEO tails;
4. the structural count: clusters engaged = min(num_tiles, grid);
5. refusals: a number operand, broadcasting, non-broadcastable shapes, float32.
"""

import pytest
import torch

import ttnn
from ttnn.operations.add import add, default_compute_kernel_config
from ttnn.operations.add.add_program_artifacts import create_program_artifacts, split_pages


def _put(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _inputs(device, shape, seed=0, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(seed)
    ta = torch.randn(*shape, dtype=torch.bfloat16)
    tb = torch.randn(*shape, dtype=torch.bfloat16)
    return ta, tb, _put(ta, device, memory_config), _put(tb, device, memory_config)


def _expected(ta, tb):
    # The fp32 DEST sum of two bf16 values, rounded once to bf16.
    return (ta.float() + tb.float()).to(torch.bfloat16).float()


def _check(got, ta, tb):
    exp = _expected(ta, tb)
    assert not torch.isnan(got).any(), "NaN pages left: some thread did not write its pages"
    assert torch.allclose(got, exp, rtol=1e-2, atol=1e-2), (got - exp).abs().max()


def _run_with_knobs(a, b, **knobs):
    """The op's program with knob overrides, into a NaN-prefilled output."""
    out = _put(torch.full(tuple(a.shape), float("nan"), dtype=torch.bfloat16), a.device())
    spec, run_args, idx = create_program_artifacts(a, b, out, default_compute_kernel_config(), **knobs)
    return ttnn.generic_op([a, b, out], spec, run_args, idx)


PLANS = [(1, 1, 1), (2, 2, 2), (4, 4, 2), (1, 4, 1)]


@pytest.mark.parametrize("threads", PLANS, ids=lambda p: "R%dC%dW%d" % p)
def test_thread_plan(device, threads):
    shape = (1, 3, 320, 320)  # 300 tiles: 9 or 10 per cluster, so every NEO of a C=4 plan gets pages
    _, counts, _ = split_pages(300, device.compute_with_storage_grid_size(), threads[1])
    assert min(counts.values()) >= threads[1]
    ta, tb, a, b = _inputs(device, shape)
    _check(ttnn.to_torch(_run_with_knobs(a, b, threads=threads)).float(), ta, tb)


@pytest.mark.parametrize("threads", [(1, 1, 1), (4, 4, 2)], ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("block_tiles", [1, 2, 4])
def test_block_tiles_ragged_tails(device, threads, block_tiles):
    # 900 tiles: 28 or 29 per cluster; at C=4 a NEO takes 7 or 8 pages, so blocks of 2 and 4 end short.
    ta, tb, a, b = _inputs(device, (1, 4, 480, 480))
    _check(ttnn.to_torch(_run_with_knobs(a, b, threads=threads, block_tiles=block_tiles)).float(), ta, tb)


@pytest.mark.parametrize("ring_depth", [2, 3, 4])
def test_ring_depth(device, ring_depth):
    ta, tb, a, b = _inputs(device, (1, 2, 8400))  # 263 tiles, the model's dist2bbox shape
    _check(ttnn.to_torch(_run_with_knobs(a, b, ring_depth=ring_depth)).float(), ta, tb)


@pytest.mark.parametrize("min_tiles", [1, 4, 8])
def test_min_tiles_per_core(device, min_tiles):
    num_tiles = 52  # [1, 1, 400, 128]
    core_set, counts, starts = split_pages(num_tiles, device.compute_with_storage_grid_size(), 1, min_tiles)
    assert core_set.num_cores() == min(32, -(-num_tiles // min_tiles))
    assert sum(counts.values()) == num_tiles
    ta, tb, a, b = _inputs(device, (1, 1, 400, 128))
    _check(ttnn.to_torch(_run_with_knobs(a, b, min_tiles_per_core=min_tiles)).float(), ta, tb)


def test_one_device_session_with_tt_metal_programs(device):
    """A tt-metal program before add, add twice, and a tt-metal program on add's output after it."""
    ta, tb, a, b = _inputs(device, (1, 2, 400), seed=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    # before: tt-metal's own program (untilize) on an input
    a_rm = ttnn.to_layout(a, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(a_rm), ta)
    out1 = add(a, b)
    out2 = add(out1, b)
    # after: tt-metal's own program on add's output, issuing untagged NoC transfers
    out_rm = ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT)
    _check(ttnn.to_torch(out1).float(), ta, tb)
    exp2 = (_expected(ta, tb) + tb.float()).to(torch.bfloat16).float()
    got2 = ttnn.to_torch(ttnn.from_device(out_rm)).float()
    assert torch.allclose(got2, exp2, rtol=1e-2, atol=1e-2), (got2 - exp2).abs().max()
    assert out2.memory_config().buffer_type == ttnn.BufferType.L1  # R9


@pytest.mark.parametrize(
    "shape,tiles", [((1, 2, 400), 13), ((1, 1, 400, 128), 52), ((1, 2, 8400), 263), ((1, 2, 33600), 1050)]
)
def test_clusters_engaged(device, shape, tiles):
    ta, tb, a, b = _inputs(device, shape)
    out = ttnn.allocate_tensor_on_device(a.spec, device)
    assert out.buffer_num_pages() == tiles
    grid = device.compute_with_storage_grid_size()
    core_set, counts, _ = split_pages(tiles, grid, 1)
    assert core_set.num_cores() == min(tiles, grid.x * grid.y)
    assert max(counts.values()) - min(counts.values()) <= 1
    _check(ttnn.to_torch(add(a, b)).float(), ta, tb)
    assert add(a, b).memory_config().buffer_type == ttnn.BufferType.DRAM  # R9


def test_refusals(device):
    _, _, a, b = _inputs(device, (1, 1, 32, 64))
    with pytest.raises(NotImplementedError):
        add(a, 2.0)
    _, _, _, b_row = _inputs(device, (1, 1, 1, 64))
    with pytest.raises(NotImplementedError):
        add(a, b_row)
    _, _, _, b_bad = _inputs(device, (1, 1, 32, 50))
    with pytest.raises(ValueError):
        add(a, b_bad)
    a32 = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(NotImplementedError):
        add(a32, a32)
    with pytest.raises(NotImplementedError):
        add(a, b, dtype=ttnn.float32)
