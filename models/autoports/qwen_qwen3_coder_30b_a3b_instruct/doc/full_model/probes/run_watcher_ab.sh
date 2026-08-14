#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Drives the stage-05 watcher A/B. Each leg needs its own process: the watcher
# aborts the process on the first tripped assert, so legs cannot share one.
#
#   bash doc/full_model/probes/run_watcher_ab.sh > doc/full_model/watcher_ab.log 2>&1
#
# Writes one raw log per leg into doc/full_model/watcher_ab/ and prints a
# summary table. Every "TRIPPED" row is a device ASSERT that compiles out when
# the watcher is off -- i.e. an invariant that is unchecked, not satisfied, on
# the delivered path.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOC="$(dirname "$HERE")"
OUT="$DOC/watcher_ab"
mkdir -p "$OUT"

export TT_METAL_WATCHER=10
export TT_METAL_WATCHER_DISABLE_ETH=1

run () {          # run <probe> <leg> <extra args...>
  local probe="$1"; shift
  local leg="$1"; shift
  local log="$OUT/${probe%.py}_${leg}.log"
  # The watcher aborts the process; silence bash's own job-status line.
  ( timeout 900 python "$HERE/$probe" --leg "$leg" "$@" > "$log" 2>&1 ) 2>/dev/null
  local rc=$?
  local verdict="clean"
  grep -q "tripped an assert" "$log" && verdict="TRIPPED"
  [ "$rc" -ne 0 ] && [ "$verdict" = "clean" ] && verdict="failed(rc=$rc)"
  printf '%-24s %-42s %s\n' "${probe%.py}" "$leg" "$verdict"
}

echo "=== Sampling1D, synthetic [1,1,32,37984]/die logits, no model ==="
for leg in argmax_nobarrier argmax_barrier split_k32 \
           argmax_stage_gather argmax_stage_slice argmax_stage_untilize argmax_stage_argmax \
           raw_gather argmax_shipped; do
  run sampler_watcher_ab.py "$leg" --reps 10
done

echo
echo "=== raw ttnn.experimental.all_gather_async, no Sampling1D ==="
for leg in $(python "$HERE/ccl_watcher_ab.py" --leg list 2>/dev/null); do
  run ccl_watcher_ab.py "$leg" --reps 10
done
