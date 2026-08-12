# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the combine bake-off. Perf is NEVER asserted here.

    scripts/run_safe_pytest.sh --run-all \
      ttnn/ttnn/operations/rms_norm/perf_experiments/allgather_combine/test_allgather_combine.py

Every variant must land the same rstd on every member of its reduction group. The
`distinct` partials make each core's contribution 12-25% of a small box's sum, so a
dropped, duplicated or mis-slotted contribution fails loudly; `exact` uses equal
powers of two, whose partial sums are bf16-exact in ANY reduction order, so the only
residual error is the bf16 rounding of the rsqrt output.
"""

import pytest

from .combine_bench import GEOMS, MODE, run
from .harness import TOL

# (geometry, partial kind). Small boxes carry the strict gate; the big boxes check that
# the same protocols survive a 110-core rectangle and 8 concurrent groups.
CASES = [
    ("small_2x2", "distinct"),
    ("small_3x3", "distinct"),
    ("small_4x2_r4_b2", "distinct"),  # rows_t > 1, multi-block: slot reuse + the ag_free ack
    ("col_1x8", "distinct"),  # 1-wide box: the op's flat tree
    ("wshard_8x1", "exact"),
    ("wshard_7x4", "exact"),
    ("focus_11x10", "exact"),
    ("bshard_8x1_r16", "exact"),  # 8 concurrent groups, 16 tile-rows, 2 blocks
]


@pytest.mark.parametrize("geom_name, kind", CASES)
@pytest.mark.parametrize("variant", ["baseline", "baseline_nohs", "allgather", "sum_mcast", "flat_allgather"])
def test_combine_variant(device, geom_name, kind, variant):
    if MODE[variant] == 2 and GEOMS[geom_name].num_blocks > 1:
        pytest.skip("flat_allgather is single-block only (see combine_dataflow.cpp)")
    _ns, max_rel, _got, _exp = run(device, geom_name, variant, kind)
    assert max_rel <= TOL[kind], f"{geom_name}/{variant}: max relative error {max_rel:.5f} > {TOL[kind]}"
