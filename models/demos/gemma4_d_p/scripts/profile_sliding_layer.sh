#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Per-op device trace of ONE Gemma4-31B sliding-window GQA decoder layer (the first sliding_attention layer)
# on a Blackhole Galaxy (8x4, CP8 x TP4). Runs the existing isolated-layer benchmark
# text_demo_prefill.py::test_prefill_layer_perf_chunk_n for layer_type=local under Tracy: the layer is
# compiled once, captured as a trace, and one replay is bracketed by the
# gemma4-layer-local-sz<SIZE>-chunk<CHUNK>-start/-stop signposts.
#
#   ./models/demos/gemma4_d_p/scripts/profile_sliding_layer.sh
#   SIZES="4096" CHUNK=0 ./models/demos/gemma4_d_p/scripts/profile_sliding_layer.sh
#
# Runs one Tracy session per chunk size (a shared session exceeds Tracy's 32K source-location limit).
# Env: SIZES (space-separated chunk sizes in tokens, default "2048 4096"), CHUNK (chunk index, default 1: the first chunk that has a
# previous chunk's sliding halo; 0 has none), OUT_ROOT (parent dir of the per-size dirs), plus the HF_* / TT_CACHE_PATH of the CI stage.
set -euo pipefail

TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "$TT_METAL_HOME"
export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
SIZES="${SIZES:-2048 4096}"
CHUNK="${CHUNK:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/mnt/models/huggingface}"
export HF_MODEL="${HF_MODEL:-google/gemma-4-31B-it}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it}"

for SIZE in $SIZES; do
  OUT="${OUT_ROOT:-$TT_METAL_HOME/gemma4_layer_profile}/${STAMP}_local_sz${SIZE}_chunk${CHUNK}"
  mkdir -p "$OUT"
  # Node id, not -k: tracy re-joins the -m argument into one shell command (same as the CI stage).
  T="models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n"
  # --op-support-count: the CI stage's device-profiler buffer size, so no op of the replay loses its data.
  python3 -m tracy -p -r -v -o "$OUT" --check-exit-code --op-support-count 20000 \
    -m "pytest ${T}[blackhole-chunk${CHUNK}-local-sz${SIZE}-ctx_256k-8x4] -vv --tb=short --timeout=7200" \
    2>&1 | tee "$OUT/run.log"

  # Per-cell op table + op-to-op gaps for the signposted window (the CI report).
  python3 models/demos/gemma4_d_p/scripts/layer_perf_report.py --profiler-dir "$OUT"

  CSV="$(find "$OUT" -name 'ops_perf_results_*.csv' | head -1)"
  [ -n "$CSV" ] || { echo "no ops_perf_results CSV under $OUT"; exit 1; }
  python3 -I models/demos/gemma4_d_p/scripts/check_layer_window.py "$CSV" "gemma4-layer-local-sz${SIZE}-chunk${CHUNK}"
  echo "done: $OUT"
done
