# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Auto-target must be the achievable floor (measured - Σgap), NOT Σideal.

A per-op ideal can OVERESTIMATE (an L1-resident op modeled at DRAM bandwidth when l1_bw is
unknown, or a dispatch floor summed as if ops ran serially), so Σideal can exceed the measured
total. Targeting Σideal would make a `measured < Σideal` exit falsely declare DONE while real
per-op gaps remain. The exit must instead chase `measured - Σgap`."""

from agent import roofline

ENV = {
    "arch": "blackhole",
    "dram_bw_gbps": 512.0,
    "l1_bw_gbps": None,  # unknown -> L1 ops fall back to the (slower) DRAM floor -> overestimate
    "worker_cores": 130,
    "peak_tflops_per_core": {"lofi": 5.4, "hifi2": 2.7, "hifi4": 1.35},
}


def _profile():
    # op A: real gap (measured 0.1 >> its tiny floor). op B: huge bytes modeled at DRAM bw ->
    # its ideal (~0.195ms) OVERSHOOTS its measured 0.001ms -> gap clamped to 0 but inflates Σideal.
    return {
        "device_ms": 0.101,
        "buckets": [
            {
                "id": "matmul",
                "device_ms": 0.1,
                "top_ops": [
                    {
                        "op_code": "matmul",
                        "shape": "32x1024 @ 1024x1024",
                        "fidelity": "lofi",
                        "device_ms": 0.1,
                        "count": 1,
                    }
                ],
            },
            {
                "id": "datamove",
                "device_ms": 0.001,
                "top_ops": [
                    {"op_code": "Tilize", "bytes": 100e6, "memory": "dram_interleaved", "device_ms": 0.001, "count": 1}
                ],
            },
        ],
    }


def test_sigma_ideal_overestimates_so_target_uses_gap():
    r = roofline.compute_rooflines(_profile(), ENV)
    baseline = 0.101
    # the overestimate: Σideal exceeds the measured total of the modeled ops
    assert r["total_ideal_ms"] > r["modeled_device_ms"], (r["total_ideal_ms"], r["modeled_device_ms"])
    # yet there is REAL headroom (op A's gap), so the run is NOT done
    assert r["total_gap_ms"] > 0
    # the correct target is the achievable floor, strictly below measured (so the loop continues)
    achievable = round(max(0.0, baseline - r["total_gap_ms"]), 4)
    assert 0.0 <= achievable < baseline
    # ...and it is NOT the (inflated) Σideal that the old code used
    assert achievable < r["total_ideal_ms"]
