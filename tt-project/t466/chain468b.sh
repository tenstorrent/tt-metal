#!/bin/bash
# #468 chain b (g15blx02): 6 s jobs on blx01, one broker job at a time through blx01-submit.sh (health + lint):
# cold 6 s base (<= 3 tries, the 6 s shapes are new compiles), cold 6 s lofi (<= 3), then the 6 s A/B on seeds 0-4
# (one pair per job, <= 2 tries each; a seed that fails twice is skipped and logged).
# Writes the first failing step's code to $M ("0" when all done; skipped seeds are listed in the log).
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd); SUB=/home/smarton/fasth3/tt-metal/tt-project/t288/blx01-submit.sh
O=${CHAIN_OUT:-$D}; M=$O/chain468b.done; LOG=$O/chain468b.log; rm -f $M
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
trap 'echo $? > "$M"' EXIT
log() { echo "[chain468b] $(date -u '+%F %T') $*" | tee -a $LOG; }
run() {
  local spec=$1 out id t
  for t in $(seq 12); do
    if out=$($SUB $spec 2>&1); then break; fi
    log "submit $spec refused (try $t): $(echo "$out" | tail -2 | tr '\n' ' ')"; out=; sleep 300
  done
  id=$(echo "$out" | sed -n 's/^job=//p'); [ -n "$id" ] || return 11
  log "$spec -> blx01 job $id"
  until $SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -qiE '^Status: +(done|completed|failed|killed|timeout|cancel|error|exit|ended)'; do
    if $SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -q '^Status:' && ! $SSH "tt-device-mcp status -j $id" | grep -qiE '^Status: +(running|queued|pending)'; then break; fi
    sleep 30
  done
  log "job $id: $($SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
}
pass() { $SSH "test -e /var/tmp/fasth3/t466/out_$1/PASS"; }
for t in 1 2 3; do pass c6s0_base && break; run $D/spec_c6s0.txt || true; done
pass c6s0_base || exit 12
for t in 1 2 3; do pass c6s0_lofi && break; run $D/spec_c6s0l.txt || true; done
pass c6s0_lofi || exit 14
for s in 0 1 2 3 4; do
  for t in 1 2; do pass ab6s${s}_base && pass ab6s${s}_lofi && break; run $D/spec_ab6s$s.txt || true; done
  pass ab6s${s}_base && pass ab6s${s}_lofi || log "seed $s skipped: failed twice"
done
log "all done"
