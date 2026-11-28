# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time harness for the perf refinements in op_requirements.md: the two production cases only (BB-22).

- `test_eager_calls`: 12 eager calls per (case, knobs). Run under `scripts/run_safe_pytest.sh --profile` and read
  DEVICE KERNEL DURATION from the ops CSV, dropping the first two calls of each group.
- `test_trace_replay_ns_per_op`: host wall clock over N replays of a K-call trace, i.e. what one call costs a
  traced model (dispatch plus kernel). Prints `REPLAY ... ns/op`; bit-exactness is the only assertion.

Both call `_dispatch`, which skips only the support gate, so the flagged p1 case is measurable before the
refinement that adds it to SUPPORTED.
"""

from __future__ import annotations

import time

import pytest
import torch
import ttnn

from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import _dispatch
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor import Knobs

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
KNOBS = {"lanes1": Knobs(), "lanes2": Knobs(num_write_lanes=2)}
EAGER_CALLS = 12
TRACE_CALLS = 16  # calls per captured trace: amortizes the execute_trace command itself
TRACE_REPLAYS = 50


def _input(device, case):
    shape, dtype, scheme, shard, grid, _ = case
    size = device.compute_with_storage_grid_size()
    for _, (x1, y1) in grid:
        if x1 >= size.x or y1 >= size.y:
            pytest.skip(f"shard grid end ({x1},{y1}) exceeds the device worker grid ({size.x},{size.y})")
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in grid])
    mc = ttnn.MemoryConfig(scheme, ttnn.BufferType.L1, ttnn.ShardSpec(crs, shard, ttnn.ShardOrientation.ROW_MAJOR))
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    tt_input = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    return tt_input, ttnn.to_torch(tt_input)


def _assert_exact(output, expected):
    got = ttnn.to_torch(output)
    assert got.dtype == expected.dtype and torch.equal(got.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("knobs", list(KNOBS))
@pytest.mark.parametrize("name", list(CASES))
def test_eager_calls(device, name, knobs):
    case = CASES[name]
    tt_input, expected = _input(device, case)
    for _ in range(EAGER_CALLS):
        output = _dispatch(tt_input, case[-1], KNOBS[knobs])
        _assert_exact(output, expected)
        ttnn.deallocate(output)
    ttnn.deallocate(tt_input)


@pytest.mark.use_module_device({"trace_region_size": 4 << 20})
@pytest.mark.parametrize("knobs", list(KNOBS))
@pytest.mark.parametrize("name", list(CASES))
def test_trace_replay_ns_per_op(device, name, knobs):
    case = CASES[name]
    tt_input, expected = _input(device, case)
    ttnn.deallocate(_dispatch(tt_input, case[-1], KNOBS[knobs]))  # program cache warm before capture (BB-16)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = [_dispatch(tt_input, case[-1], KNOBS[knobs]) for _ in range(TRACE_CALLS)]
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        for _ in range(TRACE_REPLAYS):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        samples.append(round((time.perf_counter_ns() - start) / (TRACE_REPLAYS * TRACE_CALLS)))
    ttnn.release_trace(device, trace_id)
    print(f"\nREPLAY {name} {knobs} ns/op={samples}")

    for output in outputs:
        _assert_exact(output, expected)
        ttnn.deallocate(output)
    ttnn.deallocate(tt_input)
