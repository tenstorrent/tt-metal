#!/bin/bash
# #468 chain (g15blx02): wait for cp466 on blx01, then the cold base job (c10s0, up to 5 tries while the JIT cache
# builds; each try gets further), the cold lofi job (up to 3 tries), then the 10 s seed-0 A/B (up to 2 tries). One broker job at a time, each through blx01-submit.sh (health + lint).
# Writes the first failing step's code to $M ("0" when all done).
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd); SUB=/home/smarton/fasth3/tt-metal/tt-project/t288/blx01-submit.sh
O=${CHAIN_OUT:-$D}; M=$O/chain468.done; LOG=$O/chain468.log; rm -f $M
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"
trap 'echo $? > "$M"' EXIT
log() { echo "[chain468] $(date -u '+%F %T') $*" | tee -a $LOG; }
# 1. copy (poll at most 90 min)
for i in $(seq 180); do
  rc=$($SSH cat /var/tmp/fasth3/t466/detach/cp466.rc 2>/dev/null || true)
  [ -n "$rc" ] && break; sleep 30
done
log "cp466 rc=${rc:-none}"; [ "$rc" = 0 ] || exit 10
# submit a spec, waiting out busy/unhealthy (bounded), then wait for the job to end; prints the job id
run() {
  local spec=$1 out id t
  for t in $(seq 12); do
    if out=$($SUB $spec 2>&1); then break; fi
    log "submit $spec refused (try $t): $(echo "$out" | tail -2 | tr '\n' ' ')"; out=; sleep 300
  done
  id=$(echo "$out" | sed -n 's/^job=//p'); [ -n "$id" ] || return 11
  log "$spec -> blx01 job $id"
  until $SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -qiE '^Status: +(done|completed|failed|killed|timeout|cancel|error|exit)'; do
    if $SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -q '^Status:' && ! $SSH "tt-device-mcp status -j $id" | grep -qiE '^Status: +(running|queued|pending)'; then break; fi
    sleep 30
  done
  log "job $id: $($SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
}
pass() { $SSH "test -e /var/tmp/fasth3/t466/out_$1/PASS"; }
for t in 1 2 3 4 5; do pass c10s0_base && break; run $D/spec_c10s0.txt; done
pass c10s0_base || exit 12
for t in 1 2 3; do pass c10s0_lofi && break; run $D/spec_c10s0l.txt; done
pass c10s0_lofi || exit 14
for t in 1 2; do run $D/spec_ab10s0.txt; pass ab10s0_base && pass ab10s0_lofi && break; done
pass ab10s0_base && pass ab10s0_lofi || exit 13
log "all done"
