#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Gemma4-31B prefill matmul chunk-size study (issue #59851), end to end.
#
#   models/demos/gemma4_d_p/scripts/run_matmul_chunk_sweep.sh <out_dir> <step> [args]
#
# Run from the tt-metal root with the python env active (HF_HOME / HF_MODEL / TT_CACHE_PATH set for "real" and
# "layers"). Steps; each writes under <out_dir> and can be rerun on its own:
#   sweep [n=3]       n profiler runs on one chip: every projection x C=256..32768, attention fidelity prod/LoFi/HiFi2
#                     (~8 min per run), then summary.csv, raw.csv, ns_per_token.png and the compact figure
#   real              real TP-rank-0 weights at the model chunks, to compare against random weights (one chip)
#   fallback          LoFi + fp32 accumulation on ttnn's default config at C >= 10240, 2 runs (one chip, experiment)
#   layers [sizes]    per-layer profile on the 8x4 galaxy (all 32 chips), default sizes 4096 8192 16384 2048,
#                     then layer_shares.csv and the whole-prefill estimate
#   report            rebuild every summary from what is in <out_dir>
#   all               sweep, real, fallback, layers
set -u
out=${1:?usage: run_matmul_chunk_sweep.sh <out_dir> <sweep|real|fallback|layers|report|all> [args]}
step=${2:?step}
shift 2
here=models/demos/gemma4_d_p/scripts
mkdir -p "$out"

sweep_runs() { ls -d "$out"/run[0-9]* 2>/dev/null; }

report() {
  local runs
  runs=$(sweep_runs)
  if [ -n "$runs" ]; then
    python $here/matmul_chunk_sweep_report.py $runs --out "$out" > "$out/summary.txt"
    python $here/matmul_chunk_sweep_plot.py "$out/summary.csv" "$out/ns_per_token_compact.png"
  fi
  if [ -d "$out/real" ]; then
    python $here/matmul_chunk_sweep_report.py "$out/real" --out "$out/real" > "$out/real/summary.txt"
  fi
  if [ -d "$out/fallback1" ]; then
    mkdir -p "$out/fallback"
    python $here/matmul_chunk_sweep_report.py "$out"/fallback[0-9] --out "$out/fallback" > "$out/fallback/summary.txt"
  fi
  if [ -d "$out/layer_perf" ]; then
    python $here/layer_perf_shares.py "$out"/layer_perf/sz* --out "$out/layer_shares.csv" > "$out/layer_shares.txt"
  fi
  echo "report written under $out"
}

profile_sweep() {  # <dir> [sweep args]
  local dir=$1
  shift
  rm -rf "$dir"
  mkdir -p "$dir"
  python -m tracy -r -p -o "$dir" $here/matmul_chunk_sweep.py --meta "$dir/meta.jsonl" "$@" > "$dir/run.log" 2>&1
  echo "$(basename "$dir") exit $? $(date)"
}

do_sweep() {
  for i in $(seq 1 "${1:-3}"); do
    profile_sweep "$out/run$i" --attn-fidelity prod lofi hifi2 --reps 10
  done
}

do_real() {
  [ -f "$out/tp0_weights.pt" ] || python $here/matmul_chunk_sweep_weights.py "$out/tp0_weights.pt"
  profile_sweep "$out/real" --weights "$out/tp0_weights.pt" --tag-prefix real- --chunks 1024 2048 4096 8192 16384 --reps 10
}

do_fallback() {
  for i in 1 2; do
    profile_sweep "$out/fallback$i" --fallback-lofi --attn-fidelity lofi --tag-prefix fb- \
      --chunks 10240 12288 16384 24576 32768 --reps 10
  done
}

do_layers() {
  local sizes=${*:-4096 8192 16384 2048}
  local t=models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n
  for size in $sizes; do
    local dir=$out/layer_perf/sz$size
    rm -rf "$dir"
    mkdir -p "$dir"
    PREFILL_SUMMARIES=$out/layer_perf/summaries python -m tracy -p -r -v -o "$dir" --check-exit-code \
      --op-support-count 20000 -m "pytest ${t}[blackhole-chunkci-both-sz${size}-ctx_256k-8x4] -vv --tb=short --timeout=7200" \
      > "$dir/run.log" 2>&1
    echo "layer_perf sz$size exit $? $(date)"
  done
}

case $step in
  sweep) do_sweep "$@" && report ;;
  real) do_real && report ;;
  fallback) do_fallback && report ;;
  layers) do_layers "$@" && report ;;
  report) report ;;
  all) do_sweep 3 && do_real && do_fallback && do_layers && report ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
