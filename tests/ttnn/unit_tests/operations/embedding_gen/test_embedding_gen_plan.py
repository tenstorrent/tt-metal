# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-carried tests of embedding_gen (op_design.md, Work Distribution): the thread plan and the dealing read off
the built program, the assignment sub-modes no PHASE0 entry reaches (several blocks per core, the ring-cap
column split), the R2 guard, and the op in one device session with other programs."""

import math

import pytest
import torch

import ttnn
from ttnn.operations.embedding_gen import embedding_gen
from ttnn.operations.embedding_gen import embedding_gen_program_artifacts as artifacts
from ttnn.operations.embedding_gen.embedding_gen import output_shape, run

LAYOUTS = {"TILE": ttnn.TILE_LAYOUT, "RM": ttnn.ROW_MAJOR_LAYOUT}


def _make(device, indices_shape, table_shape, table_layout, *, seed=0, high=None):
    torch.manual_seed(seed)
    V = table_shape[-2]
    idx = torch.randint(0, high or V, indices_shape, dtype=torch.int32)
    table = torch.randn(table_shape, dtype=torch.bfloat16)
    t_idx = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    t_table = ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=table_layout, device=device)
    return idx, table, t_idx, t_table


def _expected(idx, table):
    rows = idx.long().reshape(idx.shape[0], idx.shape[-1]) if idx.dim() > 1 else idx.long()
    return torch.nn.functional.embedding(rows.clamp(max=table.shape[-2] - 1), table.reshape(-1, table.shape[-1]))


def _assert_exact(out, expected):
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(expected.shape)
    assert torch.equal(got.to(torch.bfloat16), expected.to(torch.bfloat16)), (got.float() - expected.float()).abs().max()


def _out_tensor(device, t_idx, t_table, out_layout):
    shape = output_shape([int(d) for d in t_idx.shape], [int(d) for d in t_table.shape])
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), t_table.dtype, out_layout, device, ttnn.DRAM_MEMORY_CONFIG)


# --- the plan, read off the built program -------------------------------------------------------------------


@pytest.mark.parametrize("out_layout", ["TILE", "RM"])
@pytest.mark.parametrize(
    "indices_shape,table_shape",
    [((1, 32), (1, 1, 8192, 64)), ((2, 1, 1, 1024), (1, 1, 64, 128)), ((1, 8), (1, 1, 32, 8192))],
)
def test_plan_is_dealt_to_every_unit(device, indices_shape, table_shape, out_layout):
    """Reader and writer run THREAD_PLAN's thread counts, there is no compute kernel, and the per-core unit
    counts cover every unit of plan_blocks with no core left empty."""
    _, _, t_idx, t_table = _make(device, indices_shape, table_shape, ttnn.TILE_LAYOUT)
    out = _out_tensor(device, t_idx, t_table, LAYOUTS[out_layout])
    spec, run_args, _ = artifacts.create_program_artifacts(t_idx, t_table, out)

    R, C, W = artifacts.THREAD_PLAN
    threads = {k.unique_id: k.num_threads for k in spec.kernels}
    assert threads == {"reader": R, "writer": W}  # no compute kernel: C == 0
    assert C == 0

    B, S = indices_shape[0], indices_shape[-1]
    Dt = math.ceil(table_shape[-1] / 32)
    grid = device.compute_with_storage_grid_size()
    plan = artifacts.plan_blocks(B, S, Dt, 2, out_layout == "TILE", grid,
                                 ttnn.get_max_worker_l1_unreserved_size() - artifacts.L1_RESERVE_BYTES)
    for kernel_args in run_args.kernel_run_args:
        counts = kernel_args.runtime_arg_values["num_units"]
        starts = kernel_args.runtime_arg_values["start_unit"]
        assert sum(counts.values()) == plan.num_units
        assert all(n > 0 for n in counts.values())
        assert sorted(starts.values())[0] == 0
        assert len(counts) == min(plan.num_units, grid.x * grid.y)


def test_plan_blocks_ring_cap():
    """Host-only: a 32-token tile-layout block of a D = 8192 table is cut by the ring cap into 2 chunks of 128."""
    grid = ttnn.CoreCoord(8, 4)
    plan = artifacts.plan_blocks(1, 1024, 256, 2, True, grid, 4 * 1024 * 1024)
    assert (plan.num_col_chunks, plan.block_cols) == (2, 128)
    assert plan.entry_bytes * plan.ring_entries <= artifacts.DFB_MAX_BYTES
    forced = artifacts.plan_blocks(2, 1024, 4, 2, True, grid, 4 * 1024 * 1024, dfb_max_bytes=2 * 32 * 64)
    assert (forced.num_col_chunks, forced.block_cols) == (4, 1)


# --- assignment sub-modes no PHASE0 entry reaches ---------------------------------------------------------


@pytest.mark.parametrize("out_layout", ["TILE", "RM"])
@pytest.mark.parametrize("table_layout", ["TILE", "RM"])
def test_several_blocks_per_core(device, table_layout, out_layout):
    """Token split only: 64 token groups on 32 cores, so the ring cycles and the index prefetch crosses blocks."""
    idx, table, t_idx, t_table = _make(device, (2, 1, 1, 1024), (1, 1, 64, 128), LAYOUTS[table_layout])
    out = embedding_gen(t_idx, t_table, layout=LAYOUTS[out_layout])
    _assert_exact(out, _expected(idx, table))


@pytest.mark.parametrize("out_layout", ["TILE", "RM"])
@pytest.mark.parametrize("table_layout", ["TILE", "RM"])
def test_ring_cap_column_split(device, table_layout, out_layout):
    """A ring cap of one column unit per block forces the cap path: num_col_chunks = Dt, block_cols = 1."""
    idx, table, t_idx, t_table = _make(device, (2, 1, 1, 1024), (1, 1, 64, 128), LAYOUTS[table_layout])
    out = run(t_idx, t_table, out_tiled=out_layout == "TILE", dfb_max_bytes=2 * 32 * 64)
    _assert_exact(out, _expected(idx, table))


@pytest.mark.parametrize("threads", [(2, 0, 2), (2, 0, 1), (1, 0, 2)], ids=lambda p: "R%dC%dW%d" % p)
def test_other_thread_plans(device, threads):
    """The kernels deal their blocks by the plan's counts (thread t takes blocks t, t+N): a thread that did not
    run would leave its blocks unwritten. Several blocks per core, so every thread has work."""
    idx, table, t_idx, t_table = _make(device, (2, 1, 1, 1024), (1, 1, 64, 128), ttnn.TILE_LAYOUT)
    out = run(t_idx, t_table, out_tiled=True, threads=threads)
    _assert_exact(out, _expected(idx, table))


# --- R2: an index beyond the table --------------------------------------------------------------------------


@pytest.mark.parametrize("table_layout", ["TILE", "RM"])
def test_index_beyond_table(device, table_layout):
    """Indices up to V + 100: no fault, and every position with a valid index is exact."""
    V = 64
    idx, table, t_idx, t_table = _make(device, (1, 1, 1, 64), (1, 1, V, 128), LAYOUTS[table_layout], high=V + 100)
    assert (idx >= V).any()
    got = ttnn.to_torch(embedding_gen(t_idx, t_table, layout=ttnn.ROW_MAJOR_LAYOUT))
    valid = (idx.reshape(1, -1) < V)
    expected = _expected(idx, table)
    assert torch.equal(got[valid], expected[valid])


# --- one device session with other programs ----------------------------------------------------------------


def test_one_device_session(device):
    """embedding_gen after another program (a tt-metal to_layout, and itself with another shape), and a
    tt-metal program (to_layout to row-major) after it, all on one open device."""
    torch.manual_seed(1)
    before = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16)
    t_before = ttnn.to_layout(ttnn.from_torch(before, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(t_before), before)

    idx, table, t_idx, t_table = _make(device, (1, 32), (1, 1, 8192, 64), ttnn.TILE_LAYOUT, seed=2)
    out = embedding_gen(t_idx, t_table, layout=ttnn.TILE_LAYOUT)
    _assert_exact(out, _expected(idx, table))

    idx2, table2, t_idx2, t_table2 = _make(device, (1, 1, 1, 32), (1, 1, 64, 128), ttnn.ROW_MAJOR_LAYOUT, seed=3)
    out2 = embedding_gen(t_idx2, t_table2, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                         padding_idx=5)
    after = ttnn.to_layout(out2, ttnn.ROW_MAJOR_LAYOUT)
    assert torch.equal(ttnn.to_torch(ttnn.from_device(after)), _expected(idx2, table2))

    out3 = embedding_gen(t_idx, t_table, layout=ttnn.ROW_MAJOR_LAYOUT)
    _assert_exact(out3, _expected(idx, table))
    again = ttnn.to_layout(out3, ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(again), _expected(idx, table))
