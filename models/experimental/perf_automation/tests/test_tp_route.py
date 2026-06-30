# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP route-decider + regime propagation (wiring). decide_parallelism turns (model size, capacity,
mesh) into a route: single-chip / tensor-parallel / infeasible — automatically, no flag. The regime
then propagates to the loop's perf_mcp subprocess via the TT_PERF_TP_REGIME env var (default off)."""
import os
import subprocess
import sys

from agent.tp import decide_parallelism

CAP = 100


def test_route_single_chip_when_model_fits():
    r = decide_parallelism(50, CAP, 4, 16, 2048)
    assert r["route"] == "single-chip" and r["tp_regime"] is False


def test_route_tensor_parallel_when_too_big_auto():
    r = decide_parallelism(200, CAP, 4, 16, 2048)
    assert r["route"] == "tensor-parallel" and r["tp_regime"] is True and r["tp"] == 4 and r["dp"] == 1


def test_route_infeasible_when_no_legal_tp_fits():
    r = decide_parallelism(640, CAP, 4, 16, 2048)
    assert r["route"] == "infeasible" and r["tp_regime"] is False


def test_fits_model_default_does_not_enable_tp_latency():
    r = decide_parallelism(50, CAP, 4, 16, 2048, metric="device_ms", allow_tp_latency=False)
    assert r["route"] == "single-chip" and r["tp_regime"] is False


def test_fits_model_tp_latency_opt_in_under_latency_metric():
    r = decide_parallelism(50, CAP, 4, 16, 2048, metric="device_ms", allow_tp_latency=True)
    assert r["route"] == "single-chip+tp-latency" and r["tp_regime"] is True


def test_tp_latency_gated_off_under_throughput_metric():
    r = decide_parallelism(50, CAP, 4, 16, 2048, metric="fps", allow_tp_latency=True)
    assert r["route"] == "single-chip" and r["tp_regime"] is False


def test_tp_latency_gated_off_on_single_chip_box():
    r = decide_parallelism(50, CAP, 1, 16, 2048, metric="device_ms", allow_tp_latency=True)
    assert r["route"] == "single-chip" and r["tp_regime"] is False


def _regime_in_fresh_import(value):
    out = subprocess.run(
        [sys.executable, "-c", "from cc_optimize import perf_mcp; print(perf_mcp._TP_REGIME)"],
        capture_output=True,
        text=True,
        env={**os.environ, "TT_PERF_TP_REGIME": value},
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    return out.stdout.strip()


def test_regime_env_propagates_to_subprocess_default_off():
    assert _regime_in_fresh_import("0") == "False"


def test_regime_env_propagates_to_subprocess_on():
    assert _regime_in_fresh_import("1") == "True"
