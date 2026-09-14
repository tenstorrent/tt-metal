#!/usr/bin/env bash
# Post-reset re-measurement. Each cell runs TWICE: pass 1 re-warms after the glx_reset,
# pass 2 is the number to report (trap 3). Pair-major so a partial run still yields a speedup.
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../pipeline_prefill_harness/wrappers
HARNESS="$(cd "$S/.." && pwd)"
cd "$(cd "$HARNESS/../../../../../.." && pwd)" || exit 1    # repo root
source "$HARNESS/env.sh"
H=$(hostname); RES=$PWD/mistral4_perf_$H
HARD_STOP=${HARD_STOP:?}
left(){ echo $(( HARD_STOP - $(date +%s) )); }
run(){ echo "=== $1 $2 pass$3 ($(date -Is), $(( $(left)/60 ))m left) ==="
       OUT="$RES" CONFIGS="$1" ISLS="$2" MODES=ttft FORCE=1 "$HARNESS/run_matrix.sh"; }
for pass in 1 2; do
  for cfg in pp4 1rank; do
    need=200; [ "$cfg" = 1rank ] && need=330
    [ "$(left)" -lt "$need" ] && { echo "=== STOP before ${cfg} pass$pass ($(( $(left)/60 ))m left) ==="; break 2; }
    # pass 2 of a cell overwrites pass 1; keep pass 1's log for comparison
    [ "$pass" = 2 ] && cp -r "$RES/${cfg}_25600_ttft" "$RES/${cfg}_25600_ttft.pass1" 2>/dev/null
    run $cfg 25600 $pass
  done
done
echo "=== retest END ($(date -Is)) ==="
