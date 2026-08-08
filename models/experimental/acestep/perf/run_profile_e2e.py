# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Driver: run the e2e Tracy device profiler and print per-STAGE + per-OP device-kernel breakdowns.

    export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
    python models/experimental/acestep/perf/run_profile_e2e.py

Launches test_profile_e2e under the Tracy device profiler (real device_kernel_duration per op),
then reads the ops CSV directly. Splits the measured region (between the 'start' and 'stop'
signposts, after warmup) into the encode / denoise / vae stages using the 'enc_mark' / 'dit_mark'
/ 'vae_mark' signpost rows, and prints each stage's op-code breakdown. This shows the TRUE device
cost of each stage and each op type across the whole prompt->audio pipeline.
"""

import glob
import os

import pandas as pd

from tracy.process_model_log import run_device_profiler

SUBDIR = "acestep_e2e"
COMMAND = "pytest models/experimental/acestep/perf/test_profile_e2e.py::test_profile_e2e -q"


def _latest_csv():
    files = glob.glob(f"generated/profiler/{SUBDIR}/reports/*/ops_perf_results_*.csv")
    return max(files, key=os.path.getmtime) if files else None


def _breakdown(df, dur):
    g = df[["OP CODE", dur]].copy()
    g[dur] = pd.to_numeric(g[dur], errors="coerce")
    g = g.dropna()
    agg = g.groupby("OP CODE")[dur].agg(total_ns="sum", count="count")
    agg["us"] = agg["total_ns"] / 1000.0
    return agg.sort_values("total_ns", ascending=False)


def main():
    run_device_profiler(
        COMMAND,
        SUBDIR,
        check_test_return_code=False,
        device_analysis_types=["device_kernel_duration"],
        op_support_count=50000,
    )
    csv = _latest_csv()
    df = pd.read_csv(csv)
    dur = next(c for c in df.columns if "DEVICE KERNEL DURATION" in c.upper())

    # Rows are in execution order. Split by signpost markers (OP TYPE == 'signpost').
    is_sp = df["OP TYPE"] == "signpost"
    marks = {str(df.loc[i, "OP CODE"]): i for i in df.index[is_sp]}
    order = ["start", "enc_mark", "dit_mark", "vae_mark", "stop"]
    present = [m for m in order if m in marks]
    stages = [("encode", "enc_mark", "dit_mark"), ("denoise", "dit_mark", "vae_mark"), ("vae", "vae_mark", "stop")]

    print("\n===== ACE-Step e2e device-kernel time by STAGE =====")
    if "start" in marks and "stop" in marks:
        total = _breakdown(df.loc[marks["start"] + 1 : marks["stop"] - 1], dur)["us"].sum()
        for name, a, b in stages:
            if a in marks and b in marks:
                seg = df.loc[marks[a] + 1 : marks[b] - 1]
                bd = _breakdown(seg, dur)
                print(f"\n--- {name}: {bd['us'].sum():.0f} us ({bd['us'].sum() / total * 100:.1f}% of measured) ---")
                print(bd.head(8).to_string())
        print(f"\nTOTAL measured device-kernel time: {total:.0f} us")
    else:
        print("markers not found; full-region breakdown:")
        print(_breakdown(df, dur).head(25).to_string())


if __name__ == "__main__":
    main()
