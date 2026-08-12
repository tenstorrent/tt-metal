# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the `cskip_finalize` isolated bake-off.

Correctness is the ONLY pass/fail: every variant's COLUMN 0 (the only column the op ever
reads — the apply consumes rstd as `OperandKind::Col`) must match today's chain. Perf is
measured and printed by `bench.main`, never asserted.

NOTE on this box: `run_safe_pytest.sh` polls the device flock with `flock -w 20` and loses
it to the sibling agents' `tt-probe.sh` (which blocks on `flock 9` and so queues in the
kernel). The measurements in `bench.py`'s header were taken through `tt-probe.sh`:

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 560 \
    scripts/tt-probe.sh rms_norm <<'EOF'
    from ttnn.operations.rms_norm.perf_experiments.cskip_finalize import bench
    bench.main(tile_counts=(1, 2, 4, 16, 32))
    EOF

    CSKIP_N=1,16 scripts/run_safe_pytest.sh --run-all \
      ttnn/ttnn/operations/rms_norm/perf_experiments/cskip_finalize/test_cskip_finalize.py
"""

from __future__ import annotations

import os

import pytest

from ttnn.operations.rms_norm.perf_experiments.cskip_finalize import bench


def _tile_counts():
    return tuple(int(x) for x in os.environ.get("CSKIP_N", "1,4,16,32").split(","))


@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_cskip_finalize(fp32_dest_acc_en):
    tile_counts = _tile_counts()
    results = bench.main(
        tile_counts=tile_counts,
        fp32=fp32_dest_acc_en,
        tag_prefix="fp32_" if fp32_dest_acc_en else "",
    )
    assert bench.verdicts(results, tile_counts), "see the CORRECTNESS VERDICTS block above"
