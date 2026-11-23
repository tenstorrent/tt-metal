# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""fabric_mcast_relay bench entry (code: ttnn/ttnn/operations/all_gather_rms_norm/perf_experiments/fabric_mcast_relay).

  FMR_VARIANTS     comma list of relay, relay_u, mcast (default all three)
  FMR_SIZES        comma list of R_tiles:W_local (default 20:384,20:32)
  FMR_PERF         1 (default): eval/ccl_perf protocol after the correctness check; 0: correctness only
  FMR_EXPERIMENTS  ccl_perf experiments (default 5)
  FMR_OUT          result dir (default .eval/fmr)
"""

from __future__ import annotations

import os
from pathlib import Path

from eval import ccl_perf

PERF = os.environ.get("FMR_PERF", "1") == "1"
if PERF:
    for _k, _v in ccl_perf.PROFILER_ENV.items():
        os.environ.setdefault(_k, _v)

import pytest  # noqa: E402
import ttnn  # noqa: E402

from ttnn.operations.all_gather_rms_norm.perf_experiments.fabric_mcast_relay import bench  # noqa: E402

VARIANTS = os.environ.get("FMR_VARIANTS", "relay,relay_u,mcast").split(",")
SIZES = [tuple(int(v) for v in s.split(":")) for s in os.environ.get("FMR_SIZES", "20:384,20:32").split(",")]
EXPER = int(os.environ.get("FMR_EXPERIMENTS", "5"))
OUT = Path(os.environ.get("FMR_OUT", ".eval/fmr"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("size", SIZES, ids=[f"R{r}_W{w}" for r, w in SIZES])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fmr(variant, size, _fabric_mesh_cache):
    mesh = _fabric_mesh_cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE if PERF else 0)
    R, Wl = size
    call, out, full, plan = bench.make_call(mesh, variant, R, Wl)
    for coord, ch in sorted(plan["chips"].items()):
        print(
            f"FMR plan {coord} p={ch['p']} ports="
            + " ".join(f"{k[0]}{k[1]}:({c.x},{c.y})" for k, c in sorted(ch["ports"].items()))
        )
    call()
    ttnn.synchronize_device(mesh)
    bad = bench.check_output(mesh, out, full)
    print(f"FMR check {variant} R{R} W{Wl}: {'OK' if not bad else bad}")
    assert not bad, f"landing mismatch (chip, wrong elements): {bad}"
    for _ in range(3):  # repeat calls: counters re-armed, still correct
        call()
    ttnn.synchronize_device(mesh)
    assert not bench.check_output(mesh, out, full)
    if PERF:
        doc = ccl_perf.run(
            mesh, call, experiments=EXPER, out_path=OUT / f"{variant}_R{R}_W{Wl}.json", label=f"{variant} R{R} W{Wl}"
        )
        s = doc["summary"]
        print(
            f"FMR perf {variant} R{R} W{Wl}: fastest {s['fastest']['mean']:.3f} slowest {s['slowest']['mean']:.3f} us"
        )
    assert not bench.check_output(mesh, out, full)
