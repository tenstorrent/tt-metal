# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner for the `mcast_ack_elision` isolated bake-off (rms_norm perf idea I8).

Correctness is the ONLY pass/fail: with the ack elided, every receiver must still see
the exact per-block broadcast value in a single-slot, poisoned landing buffer. Perf is
printed by `bench.main`, never asserted.

Measured through tt-probe (the profiler env is what makes the ns column real):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 560 scripts/tt-probe.sh rms_norm <<'EOF'
    import sys, ttnn
    sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/mcast_ack_elision")
    import bench
    dev = ttnn.open_device(device_id=0)
    bench.main(dev, geo_names=("decode110_b1","decode110_b4","bshard64_b2"))
    ttnn.close_device(dev)
    EOF
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import bench  # noqa: E402

GEOMETRIES = ("decode110_b1", "decode110_b2", "decode110_b4", "bshard64_b2", "wshard8_b1")


@pytest.fixture(scope="module")
def dev():
    d = ttnn.open_device(device_id=0)
    try:
        yield d
    finally:
        ttnn.close_device(d)


@pytest.mark.parametrize("geo_name", GEOMETRIES)
def test_ack_elision_is_correct(dev, geo_name):
    results = bench.main(dev, geo_names=(geo_name,))
    for (_, variant), (_, stats) in results.items():
        assert stats["poison_reads"] == 0, f"{variant}: consumed the landing slot before the broadcast landed"
        assert stats["wrong_values"] == 0, f"{variant}: stale/wrong broadcast value {stats['first_wrong']}"
