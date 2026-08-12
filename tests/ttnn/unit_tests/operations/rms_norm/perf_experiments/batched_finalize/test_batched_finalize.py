# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the `batched_finalize` isolated bake-off (Perf-2 idea I12).

Correctness is the ONLY pass/fail: every variant's COLUMN 0 (the only column the op reads —
the apply consumes rstd as `OperandKind::Col`) must match today's chain bit-for-bit, or, for
the fused variants that keep `x+eps` in an fp32 LREG, match the fp64 reference. Perf is
measured and printed by `bench.main`, never asserted.

    BATCHED_N=1,32 scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/batched_finalize/test_batched_finalize.py

The numbers in `bench.py`'s header were taken through `tt-probe.sh` (the device flock is
shared with sibling agents):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 1500 scripts/tt-probe.sh rms_norm <<'EOF'
    import sys; sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
    from batched_finalize import bench; bench.main(tile_counts=(1,4,8,16,32,64), blocks=(2,4,8))
    EOF
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batched_finalize import bench  # noqa: E402


def _tile_counts():
    return tuple(int(x) for x in os.environ.get("BATCHED_N", "1,4,32").split(","))


def test_batched_finalize():
    tile_counts = _tile_counts()
    results = bench.main(tile_counts=tile_counts, blocks=(2, 4))
    assert bench.verdicts(results, tile_counts), "see the CORRECTNESS VERDICTS block above"
