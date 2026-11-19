# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Length sweep of topk_large_indices on the sparse-MLA production shape.

160 rows per chip, k = 2048, one physical capacity of 1,049,600 keys (the ring score op's capacity for 1M tokens)
and the valid length passed as trace-style metadata, so one program serves every length from 5K to 1M as in
chunked prefill. Each case runs on the top-k core rectangles of the sparse-MLA overlap layout (9 rows below the
gather row: 12 columns = 108 cores, 11 columns = 99 cores) with DRAM reads on NOC_0.

Set TOPK_SWEEP_RESULTS=<path> to append one JSON line per case.
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

pytestmark = [
    pytest.mark.use_module_device,
    pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only"),
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

_INPUTS: dict = {}


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


def _input(device, kind: str):
    if kind not in _INPUTS:
        torch_input = _generate(kind)
        tt_input = ttnn.from_torch(
            torch_input.reshape(1, 1, NUM_ROWS, T_CAP),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _INPUTS[kind] = (torch_input, tt_input)
    return _INPUTS[kind]


def _metadata(device, value: int):
    tensor = torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1)
    return ttnn.from_torch(
        tensor, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _grid(columns: int):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(columns - 1, TOP_K_ROWS))])


def _topk_duration_ns(records) -> float:
    durations = [r["duration_ns"] for r in records if any("topk_large_indices" in s for s in r["kernel_sources"])]
    assert durations, f"no topk_large_indices program among {len(records)} records"
    return max(durations)


def _check_row0(torch_input: torch.Tensor, tt_out, valid_length: int) -> None:
    indices = ttnn.to_torch(tt_out, dtype=torch.uint32).reshape(NUM_ROWS, K)[0].to(torch.int64)
    row = torch_input[0, :valid_length].float()
    expected = torch.topk(row, K, sorted=True).values
    got = row[indices]
    assert torch.equal(got, expected), "row 0 top-k values differ from torch.topk"


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("kind", ["iid", "ties64"])
@pytest.mark.parametrize("grid_id", list(GRIDS))
@pytest.mark.parametrize("valid_length", SWEEP_LENGTHS)
def test_topk_large_indices_length_sweep_perf(device, kind, grid_id, valid_length):
    require_realtime_profiler("topk_large_indices length sweep")
    grid = device.compute_with_storage_grid_size()
    if grid.x < GRIDS[grid_id] or grid.y < TOP_K_ROWS + 1:
        pytest.skip(f"{grid_id} needs a {GRIDS[grid_id]}x{TOP_K_ROWS + 1} worker grid, got {grid}")

    torch_input, tt_input = _input(device, kind)
    metadata = _metadata(device, valid_length)
    core_grid = _grid(GRIDS[grid_id])

    def run():
        return ttnn.experimental.topk_large_indices(
            tt_input, k=K, valid_length_tensor=metadata, sub_core_grids=core_grid, dram_read_noc=ttnn.NOC.NOC_0
        )

    out = run()
    ttnn.synchronize_device(device)
    _check_row0(torch_input, out, valid_length)

    durations = []
    for _ in range(REPEATS):
        _, records = profile_realtime_program(device, run, collect_all=True)
        durations.append(_topk_duration_ns(records))
    median = statistics.median(durations)
    spread = (max(durations) - min(durations)) / median
    result = {
        "kind": kind,
        "grid": grid_id,
        "valid_length": valid_length,
        "t_cap": T_CAP,
        "median_ns": median,
        "min_ns": min(durations),
        "spread": spread,
    }
    logger.info(f"TOPK_SWEEP {json.dumps(result)}")
    path = os.environ.get("TOPK_SWEEP_RESULTS")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
    assert spread <= 0.02, f"run-to-run spread {spread:.3%} above 2%: {durations}"
