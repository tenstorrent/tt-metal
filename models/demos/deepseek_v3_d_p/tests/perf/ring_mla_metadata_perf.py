#!/usr/bin/env python
"""Phase E: ring_mla (RingJointSDPA) device-kernel-time, metadata vs scalar, per topology, across all
32 galaxy devices. Runs test_mla_chunked_prefill[kimi, func, 8x4] under tracy for {scalar,metadata} x
{line,ring}, parses the ops CSV for RingJointSDPA rows between MLA_START/MLA_END, and reports per-device
kernel duration: worst device id + min-max range, plus the metadata-vs-scalar delta. Logs to
models/demos/deepseek_v3_d_p/ring_mla_perf.log.
"""
import os
import sys
import traceback

import pandas as pd
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename
from models.perf.device_perf_utils import run_device_perf

SCENARIO = "production-50k+5k"  # [5120]*11 -> 11 ring_mla calls, growing KV (chunk-aligned, the trace case)
LOGFILE = "models/demos/deepseek_v3_d_p/ring_mla_perf.log"
OPCODE = "RingJointSDPA"


def parse_per_device(csv_path):
    """Return (per_device_mean: dict[dev]->us, worst_dev, worst_us, min_us, max_us, n_devices, n_calls)
    for RingJointSDPA rows between MLA_START/MLA_END signposts."""
    df = pd.read_csv(csv_path)
    sp = df["OP TYPE"] == "signpost"
    is_start = sp & (df["OP CODE"] == "MLA_START")
    is_stop = sp & (df["OP CODE"] == "MLA_END")
    if is_start.any() and is_stop.any():
        depth = (is_start.astype(int) - is_stop.astype(int)).cumsum()
        df = df[(depth > 0) & ~sp]
    df = df[df["OP TYPE"] == "tt_dnn_device"]
    df = df[df["OP CODE"].str.contains(OPCODE, na=False, regex=False)]
    if df.empty:
        raise RuntimeError(f"no {OPCODE} rows in {csv_path}")
    dur = "DEVICE KERNEL DURATION [ns]"
    # per-device mean over its ring_mla calls, in microseconds
    per_dev = (df.groupby("DEVICE ID")[dur].mean() / 1000.0).to_dict()
    n_calls = int(df.groupby("DEVICE ID").size().max())
    worst_dev = max(per_dev, key=per_dev.get)
    vals = list(per_dev.values())
    return per_dev, worst_dev, per_dev[worst_dev], min(vals), max(vals), len(per_dev), n_calls


def run_one(meta_id, topo):
    selector = f"blackhole-{meta_id}-kimi-{SCENARIO}-func-8x4-{topo}"
    test = f"models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_chunked_prefill"
    command = f'pytest "{test}[{selector}]"'
    subdir = f"ring_mla_perf_{meta_id}_{topo}"
    logger.info(f"[phaseE] running {selector}")
    run_device_perf(command, subdir=subdir, num_iterations=1, cols=["DEVICE KERNEL"], batch_size=1)
    csv = get_latest_ops_log_filename(subdir)
    return parse_per_device(csv)


def main():
    results = {}
    for topo in ["line", "ring"]:
        for meta_id in ["scalar", "metadata"]:
            try:
                results[(topo, meta_id)] = run_one(meta_id, topo)
            except Exception as e:
                results[(topo, meta_id)] = ("ERROR", str(e))
                logger.error(f"[phaseE] {topo}/{meta_id} FAILED: {e}\n{traceback.format_exc()}")

    lines = []
    lines.append("# Phase E: ring_mla (RingJointSDPA) device kernel time — metadata vs scalar")
    lines.append(f"# scenario={SCENARIO} (11 chunk-aligned calls, growing KV), kimi, 8x4 (32 devices), func")
    lines.append("# format: <test_name> dev_kernel_us_no_metadata dev_kernel_us_with_metadata  (worst-device mean us)")
    lines.append("")
    for topo in ["line", "ring"]:
        s = results.get((topo, "scalar"))
        m = results.get((topo, "metadata"))
        name = f"test_mla_chunked_prefill[kimi-{SCENARIO}-func-8x4-{topo}]"
        if not s or s[0] == "ERROR" or not m or m[0] == "ERROR":
            serr = s[1] if s and s[0] == "ERROR" else "ok"
            merr = m[1] if m and m[0] == "ERROR" else "ok"
            lines.append(f"{name}  SKIPPED/ERROR (scalar={serr[:60]} metadata={merr[:60]})")
            continue
        _, sdev, sworst, smin, smax, sn, scalls = s
        _, mdev, mworst, mmin, mmax, mn, mcalls = m
        delta_pct = 100.0 * (mworst - sworst) / sworst if sworst else float("nan")
        flag = "  <<< >5% REGRESSION" if delta_pct > 5.0 else ("  <<< >5% FASTER" if delta_pct < -5.0 else "")
        lines.append(f"{name}  {sworst:.2f}  {mworst:.2f}   (delta {delta_pct:+.1f}%){flag}")
        lines.append(
            f"    scalar  : worst dev {sdev} {sworst:.2f}us, range {smin:.2f}-{smax:.2f}us "
            f"over {sn} devices, {scalls} calls/dev"
        )
        lines.append(
            f"    metadata: worst dev {mdev} {mworst:.2f}us, range {mmin:.2f}-{mmax:.2f}us "
            f"over {mn} devices, {mcalls} calls/dev"
        )
        lines.append("")

    out = "\n".join(lines)
    print(out)
    with open(LOGFILE, "w") as f:
        f.write(out + "\n")
    logger.success(f"[phaseE] wrote {LOGFILE}")


if __name__ == "__main__":
    main()
