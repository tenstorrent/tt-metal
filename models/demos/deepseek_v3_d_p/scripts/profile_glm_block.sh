#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Per-op device trace of ONE GLM-5.3 decoder block (DSA sparse-SDPA MLA + indexer + 256-expert MoE, layer 6)
# on a Blackhole Galaxy (8x4), seq 5120. Runs test_prefill_block.py::test_glm_prefill_block[moe] under Tracy
# in GLM_BLOCK_PROFILE mode (warm-up forward, then one signposted forward), then checks for dropped ops.
#
#   ./models/demos/deepseek_v3_d_p/scripts/profile_glm_block.sh
#   LAYER=dense ./models/demos/deepseek_v3_d_p/scripts/profile_glm_block.sh   # dense layer 0 instead
#
# Env: TT_METAL_HOME (default: repo root), LAYER=moe|dense, MAX_OPS (default 1800: warn if the window
# holds more ops per device than the profiler buffer reliably keeps), the usual GLM53_HF_MODEL /
# TT_GLM53_PREFILL_TTNN_CACHE (real weights when the cache is complete, else random: op timing is the same).
set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
LAYER="${LAYER:-moe}"
MAX_OPS="${MAX_OPS:-1800}"

export MESH_DEVICE=TG
export GLM_BLOCK_PROFILE=1
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT:-2000}"
OUT="${OUT:-$TT_METAL_HOME/glm_block_profile/$(date +%Y%m%d_%H%M%S)_${LAYER}}"
mkdir -p "$OUT"

python3 -m tracy -r -p -v -o "$OUT" -m pytest \
  "models/demos/deepseek_v3_d_p/tests/test_prefill_block.py::test_glm_prefill_block" \
  -k "seq5120 and torus-xy-8x4 and glm53 and $LAYER" -svv --tb=short --timeout=0 2>&1 | tee "$OUT/run.log"

CSV="$(ls -t "$OUT"/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)"
[ -n "$CSV" ] || CSV="$(find "$OUT" -name 'ops_perf_results_*.csv' | head -1)"
[ -n "$CSV" ] || { echo "no ops_perf_results CSV under $OUT"; exit 1; }
echo "ops csv: $CSV"

python3 -I - "$CSV" "$MAX_OPS" <<'EOF'
import sys
import pandas as pd

csv, max_ops = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(csv)
sp = df[df["OP TYPE"] == "signpost"]
t0 = sp[sp["OP CODE"] == "PROFILE_MEASURE_START"]["HOST START TS"].min()
t1 = sp[sp["OP CODE"] == "PROFILE_MEASURE_END"]["HOST START TS"].min()
if pd.isna(t0) or pd.isna(t1):
    sys.exit("PROFILE_MEASURE_START/END signposts not found in the CSV")
ops = df[(df["OP TYPE"] != "signpost") & (df["HOST START TS"] > t0) & (df["HOST START TS"] < t1)]
dev_col = "DEVICE ID" if "DEVICE ID" in ops.columns else None
per_dev = ops.groupby(dev_col).size() if dev_col else pd.Series({0: len(ops)})
print(f"ops in measured window per device: min {per_dev.min()} max {per_dev.max()} (devices: {len(per_dev)})")
dur = "DEVICE KERNEL DURATION [ns]"
if dur in ops.columns:
    lost = int(ops[dur].isna().sum() + (ops[dur] == 0).sum())
    print(f"ops with no device duration (dropped): {lost}")
    if lost:
        sys.exit("FAIL: ops lost device data; raise TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT or drain the profiler more often")
if per_dev.max() > max_ops:
    print(f"WARNING: {per_dev.max()} ops per device exceeds {max_ops}")
EOF
echo "done: $OUT"
