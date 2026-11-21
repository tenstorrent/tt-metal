# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tests the design's Work Distribution requires (op_design.md, "Tests the plan needs"):

1. every thread of the plan ran: each worker slot of the partials scratch holds that worker's own row max (a correct
   result alone would not show it: one NEO could have done all the work), at the design's plan (4, 4, 2) and at a
   forced one-thread plan, on one cluster and on several;
2. the op shares a device session: it runs after another program, and a program of tt-metal's own
   (`ttnn.to_layout` to row-major, an untilize with untagged transfers) runs after it and reads its output.
"""

import math

import pytest
import torch
import ttnn

from eval import golden_runner
from eval.specs.paged_scaled_dot_product_attention_decode import spec as S
from ttnn.operations.paged_scaled_dot_product_attention_decode import (
    paged_scaled_dot_product_attention_decode,
    paged_scaled_dot_product_attention_decode_program_artifacts as artifacts,
)

OP = "paged_scaled_dot_product_attention_decode"
CELL = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "memory_config": "L1", "sharding": "height",
        "output_memory": "DRAM", "fp32_dest_acc_en": True}


@pytest.fixture
def knobs():
    saved = (artifacts._THREAD_PLAN_OVERRIDE, artifacts._MAX_CLUSTERS_PER_USER, artifacts._DEBUG_KEEP_SCRATCH)
    yield
    artifacts._THREAD_PLAN_OVERRIDE, artifacts._MAX_CLUSTERS_PER_USER, artifacts._DEBUG_KEEP_SCRATCH = saved
    artifacts._last_scratch = None


def _call(inputs, device, positions, seed=7):
    """Build a cell's inputs as the golden suite does, call the op, return (cell, output, expected)."""
    spec = golden_runner.bound(OP).spec
    cell, plan = golden_runner.build_cell(OP, inputs, CELL, device, {"positions": positions}, seed=seed)
    dev = golden_runner.device_tensors(plan, cell.tensors, device)
    call = golden_runner.bind_call(spec, dev, cell.kwargs)
    output = paged_scaled_dot_product_attention_decode(*call.args, **call.kwargs)
    expected = spec.reference(**{p: t.to(torch.float64) for p, t in cell.tensors.items()},
                              **{k: v for k, v in cell.kwargs.items() if k == "scale"})
    return cell, output, expected


def _worker_ranges(units, workers):
    base, rem = divmod(units, workers)
    return [(w * base + min(w, rem), base + (1 if w < rem else 0)) for w in range(workers)]


@pytest.mark.parametrize(
    "threads,max_clusters,inputs,position",
    [
        ((4, 4, 2), 1, S.LLAMA_SPLIT, 200),        # one cluster, 4 workers of 2, 2, 2, 1 position tiles
        ((4, 4, 2), 1, S.LLAMA_SPLIT, 40),         # one cluster, 2 workers busy, 2 empty (neutral partials)
        ((1, 1, 1), 1, S.LLAMA_SPLIT, 200),        # the one-thread plan: one worker takes everything
        ((4, 4, 2), None, S.LLAMA_CAPTURED, 4095), # 8 clusters x 4 NEOs = 32 workers on 1024 units (8 heads)
    ],
    ids=["442_one_cluster", "442_empty_workers", "111", "442_eight_clusters"],
)
def test_every_worker_wrote_its_own_partial(device, knobs, threads, max_clusters, inputs, position):
    artifacts._THREAD_PLAN_OVERRIDE = threads
    artifacts._MAX_CLUSTERS_PER_USER = max_clusters
    artifacts._DEBUG_KEEP_SCRATCH = True
    cell, output, expected = _call(inputs, device, [position])
    golden_runner.bound(OP).spec.hooks["check"](cell, output, expected)

    R, C, W = threads
    q = cell.tensors["input_tensor_q"].to(torch.float64)[0, 0]           # [NH, D]
    k = cell.tensors["input_tensor_k"].to(torch.float64)                 # [blocks, NKV, bs, D]
    table = cell.tensors["page_table_tensor"].long()[0]
    heads, head_dim = q.shape
    kv_heads, block_size = k.shape[1], k.shape[2]
    group = heads // kv_heads
    scale = inputs[5] if inputs[5] is not None else 1 / math.sqrt(head_dim)
    upt = position // 32 + 1
    units = kv_heads * upt
    # The output has q's shape: the plan the host made for this call.
    summary = artifacts.program_summary(
        output, k, cell.tensors["page_table_tensor"], threads=threads, grid=device.compute_with_storage_grid_size())
    clusters = artifacts.clusters_for(units, C, summary["clusters_per_user"], artifacts.MERGE_COST_UNITS)
    workers = clusters * C
    dt = head_dim // 32
    scratch = ttnn.to_torch(artifacts._last_scratch).float()[0, 0]
    seen_busy = 0
    for w, (start, count) in enumerate(_worker_ranges(units, workers)):
        m = scratch[w * 32:(w + 1) * 32, dt * 32]                         # the row max, column 0 of tile Dt
        assert not torch.any(m == artifacts._SCRATCH_SENTINEL), f"worker {w}: slot never written"
        if count == 0:
            assert torch.all(m < -1e29), f"worker {w}: an empty worker emits the neutral row max"
            continue
        seen_busy += 1
        want = torch.full((heads,), -float("inf"), dtype=torch.float64)
        for v in range(start, start + count):
            head, t = divmod(v, upt)
            for p in range(t * 32, min(t * 32 + 32, position + 1)):
                key = k[table[p // block_size], head, p % block_size]
                rows = slice(head * group, (head + 1) * group)
                want[rows] = torch.maximum(want[rows], (q[rows] * scale) @ key)
        real = want > -float("inf")
        got = m[:heads].double()
        assert torch.allclose(got[real], want[real], atol=0.1, rtol=0.05), (w, got[real], want[real])
    assert seen_busy == min(units, workers)


def test_shares_a_device_session(device):
    """Another program first, then the op twice (two plans' worth of cores), then tt-metal's untilize of its output."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    before = ttnn.to_layout(ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device), ttnn.ROW_MAJOR_LAYOUT)
    assert torch.allclose(ttnn.to_torch(before).float(), x.float())
    for inputs, position in ((S.QWEN_TWO_KV_HEADS, 100), (S.PAIRS_OF_HEADS, 70)):
        cell, output, expected = _call(inputs, device, [position])
        golden_runner.bound(OP).spec.hooks["check"](cell, output, expected)
        row_major = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        got = ttnn.to_torch(ttnn.from_device(row_major)).double()
        pcc = torch.corrcoef(torch.stack([got.flatten(), expected.flatten()]))[0, 1]
        assert pcc > 0.99, pcc
