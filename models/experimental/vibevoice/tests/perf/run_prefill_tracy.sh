#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Dump a tt-perf-report for ONE warm VibeVoice LM prefill chunk (Tracy start/stop).
# Does not capture the full multi-chunk prefill_embeds loop.
#
# Usage (from tt-metal root):
#   source python_env/bin/activate
#   export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=blackhole
#   bash models/experimental/vibevoice/tests/perf/run_prefill_tracy.sh
#   bash models/experimental/vibevoice/tests/perf/run_prefill_tracy.sh 256   # chunk len
#   bash models/experimental/vibevoice/tests/perf/run_prefill_tracy.sh 256 3 # chunk + exp index
#   VV_PREFILL_PERF_START_POS=256 bash .../run_prefill_tracy.sh 256 1  # 2nd chunk
#
# Output: models/experimental/vibevoice/lm/prefill_exp${EXP}.txt

set -euo pipefail

SEQ_LEN="${1:-256}"
EXP="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT"

OUT_DIR="models/experimental/vibevoice/lm"
OUT_TXT="${OUT_DIR}/prefill_exp${EXP}.txt"
TEST="models/experimental/vibevoice/tests/perf/test_prefill_perf_dump.py"
PROFILER_ROOT="${PROFILER_ROOT:-generated/profiler/reports}"

mkdir -p "$OUT_DIR"

echo "================================================================================"
echo "VibeVoice LM prefill Tracy dump (single chunk)"
echo "  chunk_len=${SEQ_LEN}"
echo "  start_pos=${VV_PREFILL_PERF_START_POS:-0}"
echo "  output=${OUT_TXT}"
echo "================================================================================"

export VV_PREFILL_PERF_SEQ_LEN="${SEQ_LEN}"

python -m tracy -v -r -p --op-support-count 100000 \
  -m "pytest ${TEST} -k test_lm_prefill_tracy_signposts -s"

CSV="$(ls -td "${PROFILER_ROOT}"/*/ops_perf_results_*.csv 2>/dev/null | head -1 || true)"
if [[ -z "${CSV}" ]]; then
  echo "ERROR: no ops_perf_results_*.csv under ${PROFILER_ROOT}" >&2
  exit 1
fi

echo "Using CSV: ${CSV}"
tt-perf-report "${CSV}" --start-signpost start --end-signpost stop > "${OUT_TXT}"
echo "Wrote ${OUT_TXT}"
wc -l "${OUT_TXT}"
