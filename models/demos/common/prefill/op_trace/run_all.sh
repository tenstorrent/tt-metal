#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Run both single-layer op traces on a Blackhole Galaxy and collect every ops CSV into one directory
# (plus a .tar.gz of it) that can be downloaded:
#   glm     GLM-5.3 DSA + MoE block (layer 6), seq 5120      -> models/demos/deepseek_v3_d_p/scripts/profile_glm_block.sh
#   gemma   Gemma4-31B sliding-window GQA layer, 2k + 4k chunks -> models/demos/gemma4_d_p/scripts/profile_sliding_layer.sh
#
#   ./models/demos/common/prefill/op_trace/run_all.sh                 # both
#   WHICH="gemma" SIZES="4096" ./models/demos/common/prefill/op_trace/run_all.sh
#
# Env: WHICH ("glm gemma" default), BUNDLE (output dir, default <repo>/op_trace_bundle/<timestamp>), plus the knobs
# of the two scripts (SIZES, CHUNK, LAYER, GLM53_HF_MODEL, TT_GLM53_PREFILL_TTNN_CACHE, HF_MODEL, ...).
# One failing run does not stop the other; the exit code is non-zero if any run failed. See README.md.
set -uo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
cd "$TT_METAL_HOME"
export TT_METAL_HOME PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
WHICH="${WHICH:-glm gemma}"
BUNDLE="${BUNDLE:-$TT_METAL_HOME/op_trace_bundle/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BUNDLE"
SUMMARY="$BUNDLE/SUMMARY.txt"
{ echo "host: $(hostname)  date: $(date -Is)"; echo "git: $(git rev-parse --short HEAD) $(git branch --show-current)"; } > "$SUMMARY"

rc_all=0
run_one() { # name, script, [env assignments...]
  local name="$1" script="$2"; shift 2
  local work="$BUNDLE/$name"
  mkdir -p "$work"
  echo "===== $name: $script ====="
  env "$@" bash "$script" > "$work/script.log" 2>&1
  local rc=$?
  tail -n 25 "$work/script.log"
  # every ops CSV (+ the tt-perf-report style summaries next to it) flat into the bundle, original dir in the name
  while IFS= read -r f; do
    cp "$f" "$BUNDLE/${name}__$(basename "$(dirname "$f")")__$(basename "$f")"
  done < <(find "$work" -name 'ops_perf_results_*.csv' 2>/dev/null)
  local n; n=$(find "$work" -name 'ops_perf_results_*.csv' 2>/dev/null | wc -l | tr -d ' ')
  echo "$name: exit=$rc csvs=$n" | tee -a "$SUMMARY"
  grep -h -E 'ops in (the )?(measured|sliding)|ops with no device duration|ops in window per device|dropped' "$work/script.log" >> "$SUMMARY" || true
  [ "$rc" -eq 0 ] && [ "$n" -gt 0 ] || rc_all=1
}

for w in $WHICH; do
  case "$w" in
    glm)   run_one glm   models/demos/deepseek_v3_d_p/scripts/profile_glm_block.sh OUT="$BUNDLE/glm/trace" ;;
    gemma) run_one gemma models/demos/gemma4_d_p/scripts/profile_sliding_layer.sh OUT_ROOT="$BUNDLE/gemma/trace" ;;
    *) echo "unknown WHICH entry: $w (glm|gemma)"; rc_all=1 ;;
  esac
done

tar -C "$(dirname "$BUNDLE")" -czf "$BUNDLE.tar.gz" "$(basename "$BUNDLE")"
echo "bundle: $BUNDLE"; echo "tarball: $BUNDLE.tar.gz"; echo "summary:"; cat "$SUMMARY"
exit "$rc_all"
