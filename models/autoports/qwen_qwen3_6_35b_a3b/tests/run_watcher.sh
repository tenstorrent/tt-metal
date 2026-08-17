#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Watcher-clean evidence run for the Qwen3.6-35B-A3B functional decoder.
#
#   ./models/autoports/qwen_qwen3_6_35b_a3b/tests/run_watcher.sh
#
# Selection covers both decoder layer kinds through prefill, chunk continuation, eager decode,
# ragged per-slot positions, the paged cache write path and traced decode capture+replay.
# Watcher is deliberately NOT combined with the device profiler (overlapping debug resources) —
# `tests/run_perf.sh` is the separate profiling run.
set -euo pipefail

cd "${TT_METAL_HOME:?TT_METAL_HOME must be set}"
ARTIFACT_DIR=models/autoports/qwen_qwen3_6_35b_a3b/doc/functional_decoder
WATCH_DIR=$ARTIFACT_DIR/watcher
rm -rf "$WATCH_DIR"
mkdir -p "$WATCH_DIR"

SELECTOR=${1:-"traced_decode_pcc or prefill_chunk_continuation or decode_ragged_current_positions or paged_kv_cache_contents or decode_from_seeded_random_linear_state or decode_skips_inactive_slots"}

set +e
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_APPEND=0 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH="$WATCH_DIR" \
  timeout 10800 python -m pytest \
    models/autoports/qwen_qwen3_6_35b_a3b/tests/test_functional_decoder.py \
    -q --no-header -p no:cacheprovider -k "$SELECTOR" \
    > "$WATCH_DIR/pytest.log" 2>&1
RC=$?
set -e
echo "pytest exit: $RC"
tail -3 "$WATCH_DIR/pytest.log"

LOG=$(find "$WATCH_DIR" -name 'watcher.log' | head -1)
if [ -z "$LOG" ]; then
  echo "NO WATCHER LOG PRODUCED under $WATCH_DIR" >&2
  exit 2
fi
echo "watcher log: $LOG ($(wc -l < "$LOG") lines)"

# A clean watcher log still contains attach / dump / kernel-id / detach lines. What must be
# absent are the fatal/sanitizer classes watcher exists to detect.
PATTERN='Watcher detected|watcher_assert|Fatal|FATAL|sanitize|SANITIZE|out of bounds|out-of-bounds|Stack overflow|stack overflow|L1 address|invalid NOC|Invalid NOC|noc_async|CB out|hang|HANG|Debug Assert|ASSERT'
if grep -nE "$PATTERN" "$LOG" > "$WATCH_DIR/watcher_hits.txt"; then
  echo "WATCHER NOT CLEAN — hits recorded in $WATCH_DIR/watcher_hits.txt:" >&2
  head -40 "$WATCH_DIR/watcher_hits.txt" >&2
  exit 1
fi
: > "$WATCH_DIR/watcher_hits.txt"
echo "WATCHER CLEAN (no fatal/sanitize/overflow/NOC/CB findings)"

# The raw log is ~1 MB, over this repo's 500 KB committed-file limit; archive it so the
# evidence stays in tree (inspect with zless / zgrep).
gzip -f "$LOG"
