# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device perf harness for rms_norm. Correctness-checked, timing-measured.

Run under the Tracy profiler to get per-op device kernel time:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py

Each test is a single fresh-cache dispatch of one shape so the emitted
ops_perf_results CSV has one row per case (DEVICE KERNEL DURATION [ns] +
CORE COUNT). Correctness is still asserted — a perf number from a wrong kernel
is worthless.

Shapes mirror `eval/golden_tests/rms_norm/feature_spec.py`'s LLM-derived perf
cases: (1, 1, rows, hidden) with rows = 32 (decode, latency-bound, ht_total = 1
tile-row) and rows = 8192 (prefill, bandwidth-bound, ht_total = 256 tile-rows).

`RMS_NORM_BLOCK_BUDGET_KB` overrides the block-factor knob
(`L1_BLOCK_BUDGET_BYTES`) so the block size can be A/B-measured without editing
the op:

    RMS_NORM_BLOCK_BUDGET_KB=128 scripts/run_safe_pytest.sh --profile ...

`RMS_NORM_X_RES_DEPTH` overrides the resident-strip depth knob, and
`RMS_NORM_FORCE_STREAMING=1` disables the residency fast path entirely, so the
1-read-pass vs 2-read-pass lever can be measured as an ablation.

Measured — blackhole_p150b, 110-core grid, AICLK 1350 MHz, bf16 / HiFi4 /
fp32_dest_acc_en=True, DEVICE KERNEL DURATION [ns], fresh JIT cache per run.
"first commit" is the correctness-only baseline (commit 56cbc4f8).

    shape            cores   first commit      tuned    speedup
    (1,1,32,1024)        1         13_559     12_172      1.11x
    (1,1,32,2304)        1         25_404     21_382      1.19x
    (1,1,32,5120)        1         51_434     42_439      1.21x
    (1,1,32,7168)        1         63_313     57_474      1.10x
    (1,1,8192,1024)    110        103_238    102_114      1.01x
    (1,1,8192,2304)    110        214_633    215_181      1.00x
    (1,1,8192,5120)    110        482_655    483_074      1.00x
    (1,1,8192,7168)    110        835_407    831_973      1.00x

Two independent 512 KB runs agreed to within 1-2%, which sets the noise band;
everything reported as 1.00x is inside it.

Bottleneck classification (from the per-RISC columns of the same CSV): the
prefill rows are READER-bound — NCRISC occupies 90-99% of the kernel
(8192x5120: 442_221 of 482_655 ns) — so they sit at the interleaved-DRAM read
floor and the levers that pay off there are the ones that reduce bytes or
transactions, not the ones that overlap stages. The decode rows are
LATENCY-bound on one core (ht_total == 1 tile-row leaves the independent row
axis with a single unit of work), which is why the overlap levers win 1.10-1.21x
there and are explicitly disabled when the grid is full.

Block-factor knob A/B (L1_BLOCK_BUDGET_BYTES): 512 KB beats 1024 KB
everywhere — the wide prefill case degrades badly at 1024 KB
((1,1,8192,7168): 831_973 -> 1_058_535 ns, 0.79x), so the default is measured,
not assumed.

Refinement 2 built op_design.md's Lamp L1 — the cross-core W-split: each core
reduces its own W slice to a raw partial, the group root combines them and
multicasts 1/rms back. `RMS_NORM_W_SPLIT=0` forces the phase-0 row-only split,
so the two schemes A/B directly. Measured, same box/config, DEVICE KERNEL
DURATION [ns], fresh JIT cache per variant:

    shape             row-split  cores | W-split  cores | speedup
    (1,1,32,1024)        12_271      1 |   6_999     32 |  1.75x
    (1,1,32,2304)        21_429      1 |   7_663     36 |  2.80x
    (1,1,32,5120)        42_484      1 |   9_611     40 |  4.42x
    (1,1,32,7168)        57_608      1 |  11_279     56 |  5.11x
    (1,1,8192,1024)     103_052    110 | 101_922    110 |  1.01x
    (1,1,8192,2304)     217_543    110 | 215_814    110 |  1.01x
    (1,1,8192,5120)     493_253    110 | 487_581    110 |  1.01x
    (1,1,8192,7168)     830_471    110 | 834_799    110 |  0.99x

The decode column is the whole win, and it is grid occupancy: the split turns
1 busy core into 32-56. Prefill is untouched (the split is not engaged there —
its row axis already fills the grid), every prefill row inside the 1-2% noise
band. `(1,1,32,7168)` at 11_279 ns is already inside Refinement 3's clock-scaled
<= 14_894 ns ceiling at THIS config; R3 re-measures at the pinned perf config
(HiFi2 / fp32_dest_acc_en=False) and tunes the split's own knobs (cores per
group, combine shape, per-core chunk granularity).
"""

import os

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


PERF_SHAPES = [
    # decode (ht_total == 1 tile-row -> the row axis offers ONE core; wider W is
    # in-core chunked. This is the regime op_design.md Lamp L1 unlocks).
    (1, 1, 32, 1024),
    (1, 1, 32, 2304),
    (1, 1, 32, 5120),
    (1, 1, 32, 7168),
    # prefill (ht_total == 256 tile-rows -> fills the whole compute grid).
    (1, 1, 8192, 1024),
    (1, 1, 8192, 2304),
    (1, 1, 8192, 5120),
    (1, 1, 8192, 7168),
]


@pytest.fixture(autouse=True)
def _knob_overrides():
    """Apply the env-var knob overrides for one test, then restore."""
    saved = (pd.L1_BLOCK_BUDGET_BYTES, pd.L1_CB_BUDGET_BYTES, pd.X_RESIDENT_DEPTH)
    kb = os.environ.get("RMS_NORM_BLOCK_BUDGET_KB")
    if kb:
        pd.L1_BLOCK_BUDGET_BYTES = int(kb) * 1024
    depth = os.environ.get("RMS_NORM_X_RES_DEPTH")
    if depth:
        pd.X_RESIDENT_DEPTH = int(depth)
    if os.environ.get("RMS_NORM_FORCE_STREAMING") == "1":
        # A CB budget of 0 makes both residency predicates false, selecting the
        # bounded streaming fallback (2 reader passes) without touching the op.
        pd.L1_CB_BUDGET_BYTES = 0
    yield
    pd.L1_BLOCK_BUDGET_BYTES, pd.L1_CB_BUDGET_BYTES, pd.X_RESIDENT_DEPTH = saved


@pytest.mark.parametrize("shape", PERF_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_perf(device, shape):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=1e-6)

    xf = torch_x.to(torch.float32)
    expected = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)

    actual = ttnn.to_torch(tt_out).to(torch.float32)
    passed, message = check_with_pcc(expected, actual, 0.995)
    assert passed, message


def test_report_blocking(device):
    """Print the derived blocking for every perf shape (no device work).

    Cheap visibility into what the knobs actually resolved to, so a perf number
    can be attributed to a block size / residency regime / core count.
    """
    grid = device.compute_with_storage_grid_size()
    print(f"\ncompute grid: {grid.x} x {grid.y} = {grid.x * grid.y} cores")
    header = (
        f"{'shape':22s} {'Wt/core':>7} {'C':>4} {'NW':>4} {'H':>3} "
        f"{'ht_tot':>7} {'CW':>4} {'cores':>6} {'xdep':>5} {'gres':>5} {'rdbat':>6} {'CB KB':>7}"
    )
    print(header)
    for shape in PERF_SHAPES:
        tt_x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_g = ttnn.from_torch(
            torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ht_total, wt_global = pd._tile_geometry(tt_x)
        placement = pd._select_placement(device, grid, tt_x, ht_total, wt_global, False)
        blk = pd._derive_blocking(tt_x, tt_g, grid.x * grid.y, placement)
        print(
            f"{str(shape):22s} {blk.Wt:7d} {blk.wt_chunk:4d} {blk.nw:4d} {blk.ht_block:3d} "
            f"{blk.ht_total:7d} {placement.cw:4d} {placement.num_cores:6d} "
            f"{blk.x_res_depth:5d} {int(blk.gamma_resident):5d} "
            f"{pd._x_read_chunks(blk):6d} {blk.cb_total_bytes // 1024:7d}"
        )
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_g)
