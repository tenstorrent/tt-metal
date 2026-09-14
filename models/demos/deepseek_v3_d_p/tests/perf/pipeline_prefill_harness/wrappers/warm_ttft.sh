#!/usr/bin/env bash
# Phase 3 only: the WARM latency re-run, in PAIR-MAJOR order.
#
# Ordered 25,600 -> 102,400 -> 261,120 -> 5,120 and 1rank/pp4 adjacent, so that if the clock runs
# out we still have COMPLETE PAIRS (a speedup needs both halves) and the most-quoted ISL first.
# 5,120 is last because the harness is documented to mis-measure a single-chunk request (§3.1).
# FORCE=1 throughout: these cells already have a cold log and must be overwritten.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../pipeline_prefill_harness/wrappers
HARNESS="$(cd "$S/.." && pwd)"
cd "$(cd "$HARNESS/../../../../../.." && pwd)" || exit 1    # repo root
source "$HARNESS/env.sh"
H=$(hostname); RES=$PWD/mistral4_perf_$H
HARD_STOP=${HARD_STOP:?}
left(){ echo $(( HARD_STOP - $(date +%s) )); }
for isl in 25600 102400 261120 5120; do
  for cfg in 1rank pp4; do
    # 1rank cells run ~3.5 min, pp4 ~1.5; refuse to start one that cannot finish.
    need=300; [ "$cfg" = pp4 ] && need=180
    if [ "$(left)" -lt "$need" ]; then echo "=== STOP warm pass before ${cfg}_${isl} ($(( $(left)/60 ))m left) ==="; break 2; fi
    echo "=== warm ttft ${cfg}_${isl} ($(date -Is), $(( $(left)/60 ))m left) ==="
    OUT="$RES" CONFIGS="$cfg" ISLS="$isl" MODES=ttft FORCE=1 "$HARNESS/run_matrix.sh"
  done
done
echo "=== warm pass END ($(date -Is)) ==="
