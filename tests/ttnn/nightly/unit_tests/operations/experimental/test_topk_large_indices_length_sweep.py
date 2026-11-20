# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Length sweep of topk_large_indices on the sparse-MLA production shape.

160 rows per chip, k = 2048, one physical capacity of 1,049,600 keys (the ring score op's capacity for 1M tokens)
and the valid length passed as trace-style metadata, so one program serves every length from 5K to 1M as in
chunked prefill. Each case runs on the top-k core rectangles of the sparse-MLA overlap layout (9 rows below the
gather row: 12 columns = 108 cores, 11 columns = 99 cores) with DRAM reads on NOC_0.

The bound is passed the way sparse MLA passes it: valid_length_tensor plus a constant offset, capped by a
valid_end_tensor. The numbers are top-k alone and untraced: no sub-device manager is loaded and the gather row is idle,
so DRAM and NoC are uncontended.

Opt-in (TOPK_LENGTH_SWEEP=1): a spread gate on a shared runner is too tight for a nightly job. Two-level runs claim
their phase-1 work dynamically, so their spread gate is wider than the single-level one. Set
TOPK_SWEEP_RESULTS=<path> to append one JSON line per case.
"""

import json
import os
import statistics

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from tt_py_test_utils_common.utility_functions import skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler
from tests.ttnn.nightly.unit_tests.operations.experimental.topk_two_level_reference import assert_topk_exact

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
    pytest.mark.skipif(os.environ.get("TOPK_LENGTH_SWEEP") != "1", reason="opt-in perf sweep: set TOPK_LENGTH_SWEEP=1"),
]

NUM_ROWS = 160
K = 2048
T_CAP = 1_049_600
REPEATS = 5
SWEEP_LENGTHS = [
    5_120,
    16_384,
    32_768,
    56_320,
    65_536,
    98_304,
    131_072,
    196_608,
    262_144,
    517_120,
    524_288,
    524_304,
    1_048_576,
]
# Top-k rectangles of the sparse-MLA overlap layout: row 0 belongs to the KV gather.
GRIDS = {"g108": 12, "g99": 11}
TOP_K_ROWS = 9
# The model passes the chunk-global offset separately from the metadata value; any nonzero constant exercises it.
OFFSET = 160
# Rows checked against the oracle: whole rows and both ends of the split rows on the 108-core grid.
CHECK_ROWS = (0, 107, 108, 159)


def _generate(kind: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(2048)
    if kind == "iid":
        return torch.randn(NUM_ROWS, T_CAP, generator=g).to(torch.bfloat16)
    if kind == "ties64":
        # 64 distinct bf16 levels across the whole row: heavy ties at every length.
        levels = torch.from_numpy(((0x4100 + np.arange(64, dtype=np.uint32)) << 16).view(np.float32).copy())
        picks = torch.randint(0, 64, (NUM_ROWS, T_CAP), generator=g)
        return levels[picks].to(torch.bfloat16)
    raise ValueError(kind)


@pytest.fixture(scope="module")
def host_inputs():
    """Host rows per generator, shared by the module; each test uploads and frees its own device copy."""
    cache = {}

    def get(kind: str) -> torch.Tensor:
        if kind not in cache:
            cache[kind] = _generate(kind)
        return cache[kind]

    yield get
    cache.clear()


def _metadata(device, value: int):
    tensor = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    return ttnn.from_torch(
        tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _grid(columns: int):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(columns - 1, TOP_K_ROWS))])


def _topk_duration_ns(records) -> float:
    matches = [r for r in records if any("topk_large_indices" in s for s in r["kernel_sources"])]
    assert len({r["runtime_id"] for r in matches}) == 1, f"expected one top-k program, got {len(matches)} records"
    return max(r["duration_ns"] for r in matches)


def _check_rows(torch_input: torch.Tensor, tt_out, valid_length: int) -> None:
    indices = ttnn.to_torch(tt_out, dtype=torch.uint32).reshape(NUM_ROWS, K).to(torch.int64)
    for row in CHECK_ROWS:
        assert_topk_exact(torch_input[row], indices[row], valid_length, K)


# legacy: no block_select. tl_single / tl_two: a two-level program forced to one mode through the cutover.
# tl_default: a two-level program with the default cutover.
MODES = {"legacy": None, "tl_single": 0xFFFFFFFF, "tl_two": 0, "tl_default": "default"}
SWEEP_MODES = os.environ.get("TOPK_SWEEP_MODES", "legacy").split(",")


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("mode", SWEEP_MODES)
@pytest.mark.parametrize("kind", ["iid", "ties64"])
@pytest.mark.parametrize("grid_id", list(GRIDS))
@pytest.mark.parametrize("valid_length", SWEEP_LENGTHS)
def test_topk_large_indices_length_sweep_perf(device, host_inputs, kind, grid_id, valid_length, mode):
    require_realtime_profiler("topk_large_indices length sweep")
    grid = device.compute_with_storage_grid_size()
    if grid.x < GRIDS[grid_id] or grid.y < TOP_K_ROWS + 1:
        pytest.skip(f"{grid_id} needs a {GRIDS[grid_id]}x{TOP_K_ROWS + 1} worker grid, got {grid}")

    torch_input = host_inputs(kind)
    tt_input = ttnn.from_torch(
        torch_input.reshape(1, 1, NUM_ROWS, T_CAP),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    metadata = _metadata(device, valid_length - OFFSET)
    valid_end = _metadata(device, valid_length)
    core_grid = _grid(GRIDS[grid_id])

    if MODES[mode] is None:
        two_level = {}
    elif MODES[mode] == "default":
        two_level = {"block_select": 32}
    else:
        two_level = {"block_select": 32, "two_level_cutover": MODES[mode]}

    def run():
        return ttnn.experimental.topk_large_indices(
            tt_input,
            k=K,
            valid_length_tensor=metadata,
            valid_length_offset=OFFSET,
            valid_end_tensor=valid_end,
            sub_core_grids=core_grid,
            dram_read_noc=ttnn.NOC.NOC_0,
            **two_level,
        )

    out = run()
    ttnn.synchronize_device(device)
    _check_rows(torch_input, out, valid_length)

    durations = []
    for _ in range(REPEATS):
        _, records = profile_realtime_program(device, run, collect_all=True)
        durations.append(_topk_duration_ns(records))
    median = statistics.median(durations)
    spread = (max(durations) - min(durations)) / median
    result = {
        "mode": mode,
        "kind": kind,
        "grid": grid_id,
        "valid_length": valid_length,
        "t_cap": T_CAP,
        "median_ns": median,
        "min_ns": min(durations),
        "spread": spread,
    }
    ttnn.deallocate(tt_input)
    logger.info(f"TOPK_SWEEP {json.dumps(result)}")
    path = os.environ.get("TOPK_SWEEP_RESULTS")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
    # tl_default runs two-level from the default cutover (two_level_default_cutover).
    runs_two_level = mode == "tl_two" or (mode == "tl_default" and valid_length >= 98_304)
    max_spread = 0.08 if runs_two_level else 0.02
    assert spread <= max_spread, f"run-to-run spread {spread:.3%} above {max_spread:.0%}: {durations}"
