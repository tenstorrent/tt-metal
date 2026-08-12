# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the I9 pipelined-combine bake-off. Perf is NEVER asserted.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/pipelined_combine/test_pipelined_combine.py

`distinct` partials make each core's contribution 12-25% of a small box's sum, so a
dropped, duplicated or mis-slotted contribution fails loudly. `exact` uses equal powers
of two, whose partial sums are bf16-EXACT in ANY reduction order — which is what makes
the pipelined (arrival-ordered-release) variant comparable to the baseline ordering.
"""

import pytest

from .pipe_bench import GEOMS, MODE, SKEWS, run
from .harness import TOL

CASES = [
    ("small_3x3", "distinct"),
    ("small_4x2", "distinct"),
    ("small_4x2_r4_b2", "distinct"),  # rows_t > 1, multi-block
    ("col_1x8", "distinct"),  # 1-wide box: the flat tree
    ("wshard_8x1", "exact"),
    ("wshard_7x4", "exact"),
    ("focus_11x10", "exact"),
    ("focus_11x10_b4", "exact"),
    ("bshard_8x1_r16", "exact"),  # 8 concurrent groups, 16 tile-rows, 2 blocks
]


@pytest.mark.parametrize("geom_name, kind", CASES)
@pytest.mark.parametrize("variant", ["baseline", "flag", "incr", "incr_sem"])
@pytest.mark.parametrize("skew", ["none", "big"])
def test_combine_variant(device, geom_name, kind, variant, skew):
    if MODE[variant] in (6, 7) and GEOMS[geom_name].rows_t > 1:
        pytest.skip("incremental variants require rows_t == 1 (gather CB layout)")
    _ns, max_rel = run(device, geom_name, variant, kind, SKEWS[skew])
    assert max_rel <= TOL[kind], f"{geom_name}/{variant}/{skew}: max rel err {max_rel:.5f} > {TOL[kind]}"
