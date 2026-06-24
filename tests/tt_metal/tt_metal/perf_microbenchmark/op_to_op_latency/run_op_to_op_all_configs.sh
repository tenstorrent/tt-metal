#!/usr/bin/env bash
# Run the op-to-op latency benchmark NUM_RUNS times across multiple reader
# configurations, then print a side-by-side comparison summary.
#
# Configs:
#   1) baseline_push1_cb2     -- original: push-1, CB depth 2, 2 tiles/core
#   2) push2_cb4_incremental  -- push-2, CB depth 4 (2x push), incremental
#   3) push2_cb4_batch        -- push-2, CB depth 4, --reader-batch-push
#   4) push2_cb6_bufftuned    -- push-2, CB depth 6 (buffer-tune pick)
#
# Override NUM_RUNS by passing as arg 1 (default 5).
set -euo pipefail

NUM_RUNS="${1:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
export TT_METAL_HOME
export TT_METAL_DEVICE_PROFILER=1

BUILD_DIR="${BUILD_DIR:-${TT_METAL_HOME}/build_Release}"
MULTI_SH="${SCRIPT_DIR}/run_op_to_op_multi.sh"
RUNS_PARENT="${TT_METAL_HOME}/generated/profiler/op_to_op_runs"
COMMON_ARGS="--use-trace --num-programs 2 --compute-nops 1000 --use-device-profiler --use-realtime-profiler"

cmake --build "${BUILD_DIR}" --target test_op_to_op_latency -j

run_config() {
  local label="$1"
  local extra="$2"
  local tiles="$3"
  local cb="$4"
  local push="$5"
  echo "########################################"
  echo "### Config: ${label}"
  echo "### Args : ${extra}"
  echo "########################################"
  CONFIG_LABEL="${label}" \
  EXTRA_ARGS="${COMMON_ARGS} ${extra}" \
  TILES_PER_CORE="${tiles}" \
  INPUT_CB_DEPTH="${cb}" \
  READER_PUSH="${push}" \
  BUILD=0 \
    bash "${MULTI_SH}" "${NUM_RUNS}"
}

run_config "baseline_push1_cb2"    "--reader-push-tiles 1 --input-cb-depth-tiles 2 --num-pages-per-core 2" 2 2 1
run_config "push2_cb4_incremental" "--reader-push-tiles 2 --input-cb-depth-tiles 4 --num-pages-per-core 4" 4 4 2
run_config "push2_cb4_batch"       "--reader-push-tiles 2 --input-cb-depth-tiles 4 --num-pages-per-core 4 --reader-batch-push" 4 4 2
run_config "push2_cb6_bufftuned"   "--reader-push-tiles 2 --input-cb-depth-tiles 6 --num-pages-per-core 4" 4 6 2

# Side-by-side comparison summary: device op-to-op + dispatch done->go,
# aggregated across NUM_RUNS for each config.
python3 - "${RUNS_PARENT}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

runs_parent = Path(sys.argv[1])
configs = ["baseline_push1_cb2", "push2_cb4_incremental", "push2_cb4_batch", "push2_cb6_bufftuned"]

device_rows, dispatch_rows = [], []
for cfg in configs:
    dev_csv = runs_parent / cfg / "multi_run_device_op_to_op_gap.csv"
    disp_csv = runs_parent / cfg / "multi_run_dispatch_done_to_go.csv"
    if dev_csv.is_file():
        df = pd.read_csv(dev_csv)
        for col in ("mean_gap_us", "median_gap_us", "min_gap_us", "max_gap_us"):
            if col not in df.columns:
                df[col] = float("nan")
        device_rows.append({
            "config": cfg,
            "runs": len(df),
            "median_us_mean":  df["median_gap_us"].mean(),
            "median_us_std":   df["median_gap_us"].std(),
            "median_us_min":   df["median_gap_us"].min(),
            "median_us_max":   df["median_gap_us"].max(),
            "mean_us_mean":    df["mean_gap_us"].mean(),
            "mean_us_std":     df["mean_gap_us"].std(),
            "overall_min_us":  df["min_gap_us"].min() if df["min_gap_us"].notna().any() else float("nan"),
            "overall_max_us":  df["max_gap_us"].max() if df["max_gap_us"].notna().any() else float("nan"),
        })
    if disp_csv.is_file():
        df = pd.read_csv(disp_csv)
        if "dispatch_gap_us" in df.columns:
            s = df["dispatch_gap_us"]
            dispatch_rows.append({
                "config": cfg,
                "transitions": len(df),
                "dispatch_us_mean":   s.mean(),
                "dispatch_us_std":    s.std(),
                "dispatch_us_min":    s.min(),
                "dispatch_us_max":    s.max(),
                "dispatch_us_median": s.median(),
            })

if device_rows:
    device_df = pd.DataFrame(device_rows)
    out = runs_parent / "config_comparison_device_gap.csv"
    device_df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(device_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if dispatch_rows:
    disp_df = pd.DataFrame(dispatch_rows)
    out = runs_parent / "config_comparison_dispatch_gap.csv"
    disp_df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(disp_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
PY

echo
echo "All configs done. Per-config runs under: ${RUNS_PARENT}/<config>/run_*/"
echo "Per-config aggregates : ${RUNS_PARENT}/<config>/multi_run_*.csv"
echo "Side-by-side summary  : ${RUNS_PARENT}/config_comparison_*.csv"
