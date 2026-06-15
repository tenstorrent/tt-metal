# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase 2.5 of the Nsight-counters plan: scope control for compute counters.

Compute counters are SPMD, so a few cores represent an op's compute. But the
sample must be drawn from each op's own grid (a fixed core mask misses ops
whose grid excludes those cores) and be deterministic (two runs pick the same
cores -> repeatable numbers). These tests pin both properties and check the
sampled per-op utilization reproduces the full-grid number on a real capture.
"""

import glob
import json
import os

import pandas as pd
import pytest

from tracy.perf_counter_scope import sample_cores_per_op


def _df(rows):
    return pd.DataFrame(
        rows, columns=["run_host_id", "trace_id_count", "core_x", "core_y", "counter type", "value", "ref cnt"]
    )


def _make_op(op_id, cores, value_fn):
    return [[op_id, 0, cx, cy, "FPU_COUNTER", value_fn(cx, cy), 1000] for (cx, cy) in cores]


def test_samples_k_cores_per_op():
    cores = [(x, 0) for x in range(10)]
    df = _df(_make_op(1, cores, lambda x, y: 100))
    sampled = sample_cores_per_op(df, k=3)
    picked = sampled.groupby(["run_host_id", "trace_id_count"]).size()
    assert (picked == 3).all()


def test_sample_is_per_op_grid_not_fixed_mask():
    # Op 1 lives on cores x=0..4; op 2 on a disjoint grid x=5..9.
    df = _df(
        _make_op(1, [(x, 0) for x in range(5)], lambda x, y: 100)
        + _make_op(2, [(x, 0) for x in range(5, 10)], lambda x, y: 200)
    )
    sampled = sample_cores_per_op(df, k=2)
    # Every op must still be represented, and only by cores from its own grid.
    for op_id, grid in ((1, set(range(5))), (2, set(range(5, 10)))):
        op_rows = sampled[sampled["run_host_id"] == op_id]
        assert not op_rows.empty, f"op {op_id} dropped by sampling"
        assert set(op_rows["core_x"]) <= grid


def test_sampling_is_deterministic():
    cores = [(x, y) for x in range(6) for y in range(6)]
    df = _df(_make_op(1, cores, lambda x, y: x + y))
    a = sample_cores_per_op(df, k=5)
    b = sample_cores_per_op(df, k=5)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_k_at_least_grid_returns_all():
    cores = [(x, 0) for x in range(4)]
    df = _df(_make_op(1, cores, lambda x, y: 100))
    sampled = sample_cores_per_op(df, k=10)
    assert len(sampled) == len(df)


def test_sampled_median_reproduces_full_on_real_capture():
    """On a real capture, the K-core sample reproduces the full-grid FPU
    utilization median within tolerance (the repeatability promise)."""
    logs = sorted(glob.glob("generated/profiler/*/.logs/profile_log_device.csv"), key=os.path.getmtime, reverse=True)
    capture = next((p for p in logs if "blackhole" in open(p).readline().lower()), None)
    if capture is None:
        pytest.skip("no blackhole capture to validate against")

    rows = []
    with open(capture) as f:
        f.readline()
        f.readline()
        import csv

        for r in csv.reader(f):
            if len(r) < 15 or r[4].strip() != "9090" or "FPU_COUNTER" not in r[14]:
                continue
            meta = r[14]
            val = int(meta.split('"value":')[1].split("}")[0].strip().strip('"'))
            ref = int(meta.split('"ref cnt":')[1].split(";")[0].strip().strip('"'))
            rows.append([int(r[7]), 0, int(r[1]), int(r[2]), "FPU_COUNTER", val, ref])
    if not rows:
        pytest.skip("capture has no FPU markers")

    df = _df(rows)
    # Pick the op with the widest grid so sampling has something to do.
    grid_sizes = df.groupby("run_host_id").size()
    op_id = grid_sizes.idxmax()
    op = df[df["run_host_id"] == op_id]
    if op["core_x"].nunique() * op["core_y"].nunique() < 8:
        pytest.skip("widest op grid too small to sample")

    full_med = (op["value"] / op["ref cnt"] * 100).median()
    sampled = sample_cores_per_op(df, k=3)
    sm = sampled[sampled["run_host_id"] == op_id]
    sampled_med = (sm["value"] / sm["ref cnt"] * 100).median()
    assert abs(full_med - sampled_med) <= 5.0, f"sampled median {sampled_med:.2f}% off full {full_med:.2f}% by >5 pts"
