# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench driver for rms_norm — measurement only, never a pass/fail gate.

Dispatches the cumulative bench set and the per-lever ON/OFF counterfactuals from
`ttnn.operations.rms_norm._bench_rms_norm`, and writes the dispatch MANIFEST that
`_bench_rms_norm.report_from_csv()` folds the Tracy per-op CSV back onto.  Perf
is evidence; the only assertion here is that the dispatches happened.

    scripts/run_safe_pytest.sh --profile \\
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_bench.py -s

then

    python3 -c "from ttnn.operations.rms_norm import _bench_rms_norm as b; \\
                print(b.report_from_csv('<the printed CSV path>'))"

Env:
    RMS_BENCH_SHAPES   comma list of bench-shape names (default: all)
    RMS_BENCH_MODE     "baseline" | "levers" | "ablation" | "both" (default: both)
    RMS_BENCH_LEVER_SHAPES  comma list of shapes to run the lever arms on
"""

import os

import pytest

from ttnn.operations.rms_norm import _bench_rms_norm as bench


def _shapes(var, default):
    sel = os.environ.get(var)
    return [s for s in sel.split(",") if s] if sel else default


@pytest.mark.timeout(3600)
def test_rms_norm_bench(device):
    mode = os.environ.get("RMS_BENCH_MODE", "both")
    manifest = []

    if mode in ("baseline", "both"):
        manifest += bench.run_baseline(device, _shapes("RMS_BENCH_SHAPES", list(bench.BENCH_SHAPES)))

    if mode in ("ablation", "both"):
        for name in _shapes("RMS_BENCH_ABL_SHAPES", ["grid_filling", "grid_starved"]):
            manifest += bench.run_ablation(device, name)

    if mode in ("cores", "both"):
        for name in _shapes("RMS_BENCH_CORE_SHAPES", ["grid_filling"]):
            manifest += bench.run_core_sweep(device, name)

    if mode in ("gamma", "both"):
        for name in _shapes("RMS_BENCH_GAMMA_SHAPES", ["grid_filling"]):
            manifest += bench.run_gamma_replication(device, name)

    if mode in ("fidelity", "both"):
        for name in _shapes("RMS_BENCH_FID_SHAPES", ["grid_filling", "grid_starved"]):
            manifest += bench.run_fidelity(device, name)

    if mode in ("levers", "both"):
        for name in _shapes("RMS_BENCH_LEVER_SHAPES", ["grid_filling", "smallest"]):
            manifest += bench.run_levers(device, name)

    path = bench.write_manifest(manifest)
    print(f"\nRMS_BENCH: manifest -> {path} ({len(manifest)} arms)")
    for arm in manifest:
        print(f"  {arm['label']:<34} shape={arm['shape']:<14} levers={arm['levers']}")
    assert manifest, "bench dispatched nothing"
