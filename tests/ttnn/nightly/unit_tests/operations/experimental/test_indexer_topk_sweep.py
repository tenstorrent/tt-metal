# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure indexer `topk_large_indices` RAW device-kernel time (from tracy) for the two candidate
indexer sharding configs, across KV-cache lengths, at a fixed 5k global chunk (q).

The driver spawns the impl under tracy, reads the device ops log, and PRINTS a table (median
DEVICE KERNEL DURATION per case) you can paste back — no manual CSV digging.

topk is a pure per-chip op (no CCL) that parallelizes over rows across the worker cores, so the
CORE COUNT must match production (Galaxy chip = 120). RUN THIS ON THE 8x4 BOX.

Two configs (same 5120-token global chunk; differ only in rows-per-chip fed to topk):
  * sp_tp : heads TP-sharded          -> q_global/SP        = 5120/8  = 640 rows/chip (current)
  * sp_sp : seq resharded like sparse -> q_global/(SP*TP)   = 5120/32 = 160 rows/chip
Scored length T = kv_len + chunk. k = index_topk = 2048.

RUN (the driver runs tracy itself; you just run the driver):
  source python_env/bin/activate ; export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
  pytest tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_topk_sweep.py::test_indexer_topk_sweep -s
Then paste the "INDEXER TOPK SWEEP" table.
"""
import os
import re
import statistics
from unittest import mock

import pytest
import torch
import ttnn

CHUNK_GLOBAL = 5120  # q length (global), fixed
SP, TP = 8, 4
CONFIG_ROWS = {
    "sptp": CHUNK_GLOBAL // SP,         # 640 rows/chip — heads TP-sharded (current)
    "spsp": CHUNK_GLOBAL // (SP * TP),  # 160 rows/chip — seq resharded like sparse
}
KV_LENS = [0, 5120, 25600, 51200, 76800, 102400, 122880]  # kv-cache ISL (tokens)
K = 2048     # index_topk
WARMUP = 2   # first calls per case (program compile / first alloc) — dropped by the driver
ITERS = 10   # tracy captures each; we take the median


# ---------------------------------------------------------------------------
# Inner: the ops to profile (run under tracy by the driver). Gated so it only
# runs inside the driver's tracy subprocess, never in a top-level collection.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("DS_TOPK_IMPL") != "1", reason="run via the driver (test_indexer_topk_sweep)")
@pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="topk_large_indices is Blackhole-only")
def test_indexer_topk_sweep_impl(device):
    from tracy import signpost

    signpost("start")
    for config, rows in CONFIG_ROWS.items():
        for kv_len in KV_LENS:
            T = kv_len + CHUNK_GLOBAL
            k = min(K, T)
            torch.manual_seed(0)
            inp = ttnn.from_torch(
                torch.randn(1, 1, rows, T, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            # Signpost first, THEN warm + timed — all same-shape calls stay inside THIS case's
            # region (a warmup before the signpost would leak into the previous case's bucket).
            # The driver drops the first WARMUP samples per bucket (program compile / first alloc).
            signpost(f"case_{config}_kv{kv_len}_rows{rows}_T{T}")
            for _ in range(WARMUP + ITERS):
                ttnn.deallocate(ttnn.experimental.topk_large_indices(inp, k=k))
            ttnn.synchronize_device(device)
            ttnn.deallocate(inp)
    signpost("stop")


# ---------------------------------------------------------------------------
# Driver: run the impl under tracy, read the ops log, print the table.
# ---------------------------------------------------------------------------
def test_indexer_topk_sweep():
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    subdir = "indexer_topk_sweep"
    command = (
        "pytest tests/ttnn/nightly/unit_tests/operations/experimental/"
        "test_indexer_topk_sweep.py::test_indexer_topk_sweep_impl -s"
    )
    with mock.patch.dict(os.environ, {"DS_TOPK_IMPL": "1"}):
        run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"], op_support_count=5000)

    df = pd.read_csv(get_latest_ops_log_filename(subdir))
    df.columns = [c.strip() for c in df.columns]
    DUR = "DEVICE KERNEL DURATION [ns]"
    df[DUR] = pd.to_numeric(df[DUR], errors="coerce")

    # Segment by the per-case signpost (rows in file/chronological order). Do NOT device-filter the
    # whole frame first — signpost rows are host markers with DEVICE ID = NaN and would be dropped,
    # breaking segmentation. Filter to device 0 only when collecting the (per-chip) topk durations.
    cur = None
    buckets = {}
    meta = {}
    cores = set()
    for _, r in df.iterrows():
        oc = str(r["OP CODE"])
        m = re.match(r"case_(\w+)_kv(\d+)_rows(\d+)_T(\d+)", oc)
        if m:
            cur = oc
            meta[cur] = dict(config=m.group(1), kv=int(m.group(2)), rows=int(m.group(3)), T=int(m.group(4)))
            buckets.setdefault(cur, [])
            continue
        if oc == "TopkLargeIndicesDeviceOperation" and cur and pd.notna(r[DUR]):
            if pd.to_numeric(r.get("DEVICE ID"), errors="coerce") != 0:
                continue
            buckets[cur].append(r[DUR])
            c = pd.to_numeric(r.get("CORE COUNT"), errors="coerce")
            if pd.notna(c):
                cores.add(int(c))
    buckets = {k: v[WARMUP:] for k, v in buckets.items()}  # drop per-case warmup samples

    hdr = f"{'config':>7} {'rows':>6} {'kv_len':>8} {'T':>8} {'k':>6} {'in shape':>20} {'kernel us (med)':>16} {'n':>4}"
    lines = ["", "=" * 90, f"INDEXER TOPK SWEEP  chunk=5120 global, k={K}  CORE COUNT={sorted(cores)}", "=" * 90, hdr, "-" * len(hdr)]
    for case, durs in buckets.items():
        mm = meta[case]
        k = min(K, mm["T"])
        med = statistics.median(durs) / 1e3 if durs else float("nan")
        shape = f"[1,1,{mm['rows']},{mm['T']}]"
        lines.append(f"{mm['config']:>7} {mm['rows']:>6} {mm['kv']:>8} {mm['T']:>8} {k:>6} {shape:>20} {med:>16.1f} {len(durs):>4}")
    lines.append("=" * 90)
    print("\n".join(lines))
    assert buckets, "no topk ops captured — did the impl run under tracy?"
