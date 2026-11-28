# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time harness for the perf refinements in op_requirements.md: production cases only (BB-22).

Run under `scripts/run_safe_pytest.sh --profile <this file>` and summarize the ops CSV with
`probes/perf_csv_summary.py <csv> <labels in test order>` (12 eager calls per group, first 2 dropped).

All cases are claimed in SUPPORTED (Refinement 1) and run through the real entry point; `_knobs`
pins a non-default plan.
"""

from __future__ import annotations


import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved import g6_sharded_to_interleaved
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import Knobs


EAGER_CALLS = 12

CASES = {
    # FLUX.2-dev (1, 5120) bf16, 32 width shards of 1 x 5 tiles -> DRAM interleaved.
    "p2": (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 160),
        (((0, 0), (7, 3)),),
        ttnn.DRAM_MEMORY_CONFIG,
    ),
    # Guard: the p2 tensor written to L1 interleaved instead (L1-output plan on a small grid).
    "p2l1": (
        (1, 5120),
        ttnn.bfloat16,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        (32, 160),
        (((0, 0), (7, 3)),),
        ttnn.L1_MEMORY_CONFIG,
    ),
    # Panoptic-DeepLab (1, 128, 256, 32) bf8b, 103 height shards of 10 x 1 tiles -> L1 interleaved (PERF FOCUS).
    "p1": (
        (1, 128, 256, 32),
        ttnn.bfloat8_b,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        (320, 32),
        (((0, 0), (10, 8)), ((0, 9), (3, 9))),
        ttnn.L1_MEMORY_CONFIG,
    ),
}
KNOBS = {
    "auto": None,  # the entry point's own plan (select_knobs): what a model gets
    "default": Knobs(),
    "lanes2": Knobs(writer_lanes=2),
    # Refinement 2 levers: bank_stagger, noc_placement (lane NoC choice), and their combinations.
    "stagger": Knobs(bank_stagger=True),
    "noc0": Knobs(lane_nocs=(0, 1)),
    "lanes2_stagger": Knobs(writer_lanes=2, bank_stagger=True),
    "dual": Knobs(dual_noc=True),
    "dual_stagger": Knobs(dual_noc=True, bank_stagger=True),
}


def _input(device, case):
    shape, dtype, scheme, shard, grid, _ = case
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
    mc = ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, ttnn.ShardOrientation.ROW_MAJOR))
    torch.manual_seed(1)
    x = torch.randn(shape).to(torch.bfloat16)
    return x, ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)


@pytest.mark.parametrize("knobs", list(KNOBS))
@pytest.mark.parametrize("name", list(CASES))
def test_eager_calls(device, name, knobs):
    case = CASES[name]
    _, tt_input = _input(device, case)
    expected = ttnn.to_torch(tt_input)
    for _ in range(EAGER_CALLS):
        out = g6_sharded_to_interleaved(tt_input, case[-1], _knobs=KNOBS[knobs])
        ttnn.deallocate(out) if _ < EAGER_CALLS - 1 else None
    assert torch.equal(ttnn.to_torch(out).view(torch.int16), expected.view(torch.int16))
    for t in (out, tt_input):
        ttnn.deallocate(t)


TRACE_CALLS = 16
TRACE_REPLAYS = 50


@pytest.mark.use_module_device({"trace_region_size": 1 << 20})
@pytest.mark.parametrize("knobs", list(KNOBS))
@pytest.mark.parametrize("name", list(CASES))
def test_trace_replay_ns_per_op(device, name, knobs):
    """Host wall clock per op over replays of a TRACE_CALLS-call trace: what one call costs a traced model."""
    import time

    case = CASES[name]
    _, tt_input = _input(device, case)
    expected = ttnn.to_torch(tt_input).view(torch.int16)
    ttnn.deallocate(g6_sharded_to_interleaved(tt_input, case[-1], _knobs=KNOBS[knobs]))  # warm cache (BB-16)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = [g6_sharded_to_interleaved(tt_input, case[-1], _knobs=KNOBS[knobs]) for _ in range(TRACE_CALLS)]
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter_ns()
        for _ in range(TRACE_REPLAYS):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter_ns() - t0) / (TRACE_REPLAYS * TRACE_CALLS))
    print(f"\nREPLAY {name} {knobs} ns/op={[round(s) for s in samples]}")
    assert torch.equal(ttnn.to_torch(outputs[-1]).view(torch.int16), expected)
    ttnn.release_trace(device, trace_id)
    for t in outputs + [tt_input]:
        ttnn.deallocate(t)
