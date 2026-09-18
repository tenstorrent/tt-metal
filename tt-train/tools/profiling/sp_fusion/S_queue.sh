#!/usr/bin/env bash
# Agent S's device runs, most informative first; idempotent (a run whose log already holds 6 steps is skipped), so it
# can be killed and restarted. Results: logs/S_queue.txt (RESULT + PHASES lines). Usage: S_queue.sh
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
B="$SPFUSE/bench_sp_overlap.sh"; OUT="$SPFUSE/logs/S_queue.txt"
done_already() { [ "$(grep -c optimizer_step_done "$SPFUSE/logs/$1.log" 2>/dev/null)" -ge 6 ] 2>/dev/null; }
ovl() {  # ovl <impl> <bwd_impl> <overlap> <ccl> <batch> <memeff> <topo>
  local tag="$1"; [ "$2" != same ] && tag="$1-bwd$2"; local ccl="$4"; [ "$3" = off ] && ccl=none
  tag="${tag}_$3_${ccl//=/}_$7_b$5"; [ "$6" = 1 ] && tag="${tag}_memeff"
  if done_already "bench_sp_ovl_$tag"; then echo "skip $tag (done)" | tee -a "$OUT"; return; fi
  echo "== $(date +%T) $tag" | tee -a "$OUT"
  IMPL=$1 BWD_IMPL=$2 OVERLAP=$3 CCL=$4 BATCH=$5 MEMEFF=$6 "$B" $7 6 2>&1 | grep -E "^RESULT|^PHASES|HUNG|timeout|Traceback|Error" | tee -a "$OUT"
}
# 1. batch 1, line (ring measured before the wipe: split 0.609 / backward 0.592 / fused-bwd 0.599; nocomm split line 0.537)
ovl composed same     split    rows=1 1 0 line
ovl composed same     backward rows=1 1 0 line
ovl fused    composed backward rows=1 1 0 line
# 2. batch 5 with activation recompute (the batch-5 configuration to report; nocomm/composed/fused from the coordinator)
for t in ring line; do
  ovl composed same     split    rows=1 5 1 $t
  ovl composed same     backward rows=1 5 1 $t
  ovl fused    composed backward rows=1 5 1 $t
done
# 3. the schedule's own cost: NoComm backward through the two-queue machinery vs plain (ring, batch 1)
ovl composed nocomm split    rows=1 1 0 ring
ovl composed nocomm backward rows=1 1 0 ring
# 4. CCL region shape: one column, two rows (split vs backward)
for spec in columns=1 rows=2; do for t in ring line; do
  ovl composed same split    $spec 1 0 $t
  ovl composed same backward $spec 1 0 $t
done; done
# 5. batch 2
for t in ring line; do
  ovl composed same     split    rows=1 2 0 $t
  ovl composed same     backward rows=1 2 0 $t
  ovl fused    composed backward rows=1 2 0 $t
done
# 6. the ring batch-1 pair again, for the record (logs of the pre-wipe runs are gone)
ovl composed same split    rows=1 1 0 ring
ovl composed same backward rows=1 1 0 ring
ovl nocomm   same split    rows=1 1 0 ring
echo "S_QUEUE_DONE $(date +%T)" | tee -a "$OUT"
