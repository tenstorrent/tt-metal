# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Perf bench for moreh_mean reduce-along-H, used to compare two commits of tt-metal:

    base = merge-base(HEAD, main) = fda1e45f96f   (mask-tile path)
    head = 9751f4fd9f5                            (partial-scaler path)

Not a correctness test. Run it under the device profiler and read
DEVICE KERNEL DURATION [ns] (column 19) from
generated/profiler/reports/<ts>/ops_perf_results_*.csv:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/moreh_mean_perf/bench_moreh_mean_h.py

Set BENCH_CHECK=1 to also verify numerics (adds ttnn.to_torch readbacks, which
show up as extra rows in the profiler CSV -- so leave it off when measuring).

Shape design
------------
The H factory splits work over COLUMNS: units_to_divide = NC * Wt, and each core
reduces Ht tiles per column it owns. The code this branch deletes (copy + mask_tile
+ pack through a masked_input DFB, plus a second accumulating reduce call) ran once
PER COLUMN, so:

  * the relative win should be largest at small Ht (fixed per-column overhead is a
    bigger fraction of per-column work),
  * and should shrink as Ht grows and the reduce itself dominates.

Every shape holds columns fixed at NC*Wt = 7*64 = 448, which divides evenly on both
a 56-core (8x7) and a 64-core (8x8) grid, so the work split is balanced and the only
variable across the ragged shapes is Ht.

ragged_* shapes have origin_H % 32 == 17 -> do_mask_h = true (the masked path).
aligned_* shapes have origin_H % 32 == 0 -> do_mask_h = false. These are NOT a null
control: base still emitted two reduce calls plus an accumulator reload even when
unmasked, while head emits one, so head may win here too.

The real contamination check is w_control_*: reducing along W uses MorehMeanWFactory,
which this branch does not touch at all. Same op, same build, unmodified code path.
If w_control moves by more than the ~2-3% noise band between the two commits, the
comparison is contaminated (different build config, clock drift, etc.) and the H
numbers cannot be trusted.
"""

import os

import pytest
import torch

import ttnn

TILE = 32
NC = 7
WT = 64
W = WT * TILE  # 2048 -> Wt = 64; NC * Wt = 448 columns

CHECK = os.environ.get("BENCH_CHECK", "0") == "1"


def _ragged_h(ht):
    """origin_H with Ht tiles and a 17-row partial last tile."""
    return (ht - 1) * TILE + 17


# (id, origin_H, dim) -- dim 2 = reduce H (changed), dim 3 = reduce W (untouched)
CASES = [
    # masked path, Ht sweep: expect the largest relative win at the top
    ("ragged_ht1", _ragged_h(1), 2),
    ("ragged_ht4", _ragged_h(4), 2),
    ("ragged_ht16", _ragged_h(16), 2),
    ("ragged_ht32", _ragged_h(32), 2),
    # unmasked path: one reduce call (head) vs two + accum reload (base)
    ("aligned_ht4", 4 * TILE, 2),
    ("aligned_ht32", 32 * TILE, 2),
    # untouched code path -- contamination check, must not move
    ("w_control_ragged", _ragged_h(4), 3),
    ("w_control_aligned", 4 * TILE, 3),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_moreh_mean_bench(case, device):
    case_id, origin_H, dim = case
    torch.manual_seed(2024)

    shape = [1, NC, origin_H, W]
    torch_input = torch.rand(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=float("nan"),
    )

    # One measured invocation per case. DEVICE KERNEL DURATION has no warm-up
    # transient, so no trial loop -- see the perf-measure measurement discipline.
    ttnn_output = ttnn.operations.moreh.mean(ttnn_input, dim=[dim], keepdim=True)

    if CHECK:
        expected = torch.mean(torch_input, dim=dim, keepdim=True)
        actual = ttnn.to_torch(ttnn_output)
        torch.testing.assert_close(
            actual.to(torch.float32),
            expected,
            rtol=0.1,
            atol=0.1,
            msg=lambda m: f"{case_id} (origin_H={origin_H}, dim={dim}) mismatch:\n{m}",
        )
