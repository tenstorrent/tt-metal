# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-time profile of the batched ``matmul_decode`` (DeepSeek o_a) at ``M`` rows, against ``M`` single-row calls.

The shape is o_a's: 8 groups of ``[4096] -> [1024]`` on an 8x8 grid (``Bc = 1``), A the ROW_MAJOR HEIGHT_SHARDED
``[M, batch * K]`` replica and the packed ``[M, batch * N]`` result mcast back onto the B grid. For every weight
source and every ``M`` in ``ROWS`` the op first runs once to fill the program cache and check PCC, then
``ITERATIONS`` times after its ``<weight>-m<M>`` signpost. Run::

    python -m tracy -p -r -v -m pytest tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_batched_multi_row_perf.py

then summarize the ops report (median device kernel time per signpost, against ``M`` looped single-row calls)::

    python3 tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_decode_batched_multi_row_perf.py \
        generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv
"""

from __future__ import annotations

import csv
import statistics
import sys

import pytest
import torch

import ttnn
from tracy import signpost

from tests.ttnn.nightly.unit_tests.operations.matmul.test_matmul_decode import batched_replicated_inputs
from tests.ttnn.utils_for_testing import assert_with_pcc

BATCH, K, N, B_BLOCKS, N_BLOCKS = 8, 4096, 1024, 8, 8
ROWS = [1, 5]
ITERATIONS = 10


@pytest.mark.parametrize("dram_weight", [True, False], ids=["dram_weight", "l1_weight"])
def test_batched_multi_row_profile(device, dram_weight) -> None:
    grid = device.compute_with_storage_grid_size()
    if grid.x < N_BLOCKS or grid.y < B_BLOCKS:
        pytest.skip(f"needs an {N_BLOCKS}x{B_BLOCKS} core grid")
    torch.manual_seed(0)
    weight = "dram" if dram_weight else "l1"
    for m in ROWS:
        torch_a, torch_b, a, b, b_grid, kwargs = batched_replicated_inputs(
            device, m, BATCH, K, N, B_BLOCKS, N_BLOCKS, dram_weight
        )
        kwargs.update(output_core_grid=b_grid, output_mcast_two_hub=True)
        signpost(f"warmup-{weight}-m{m}")
        out = ttnn.experimental.matmul_decode(a, b, **kwargs)
        replica = ttnn.to_torch(out).float().reshape(-1, m, BATCH, N)[0]
        assert_with_pcc(torch.matmul(torch_a.float(), torch_b.float()), replica.permute(1, 0, 2), 0.99)
        ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        signpost(f"{weight}-m{m}")
        for _ in range(ITERATIONS):
            out = ttnn.experimental.matmul_decode(a, b, **kwargs)
            ttnn.synchronize_device(device)
            ttnn.deallocate(out)
        ttnn.deallocate(a)
        ttnn.deallocate(b)


def summarize(csv_path: str) -> None:
    """Median device kernel time of matmul_decode under every signpost, against ``M`` looped single-row calls."""
    durations: dict[str, list[float]] = {}
    section = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["OP TYPE"] == "signpost":
                section = row["OP CODE"]
            elif section is not None and not section.startswith("warmup") and "MatmulDecode" in row["OP CODE"]:
                durations.setdefault(section, []).append(float(row["DEVICE KERNEL DURATION [ns]"]) / 1000)
    for weight in sorted({s.split("-")[0] for s in durations}):
        single = statistics.median(durations[f"{weight}-m1"])
        print(f"\n{weight} weight: M = 1 call {single:.2f} us")
        print(f"{'M':>3} {'M rows us':>10} {'us / row':>9} {'looped us':>10} {'M / 1':>6} {'speedup':>8}")
        for m in ROWS:
            times = durations.get(f"{weight}-m{m}")
            if not times:
                continue
            t = statistics.median(times)
            print(f"{m:>3} {t:>10.2f} {t / m:>9.2f} {single * m:>10.2f} {t / single:>6.2f} {single * m / t:>7.2f}x")


if __name__ == "__main__":
    summarize(sys.argv[1])
