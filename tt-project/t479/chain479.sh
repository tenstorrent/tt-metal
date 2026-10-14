#!/bin/bash
# #479 chain (g15blx02): wait for the blx01 build (rb479.rc), collect-only check (no device), then job d1
# (sla0.85 arm, dumps vsa_sdpa calls 0/49/150/199) through blx01-submit.sh, up to 3 tries (the first run builds the
# cold h3hf caches and may time out; caches persist), then the CPU replay of each dump on blx01 (nice, no device).
# Marker: chain479.done (exit code), log chain479.log, replay output $T/replay479.txt on blx01.
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd); SUB=/home/smarton/fasth3/tt-metal/tt-project/t288/blx01-submit.sh
O=${CHAIN_OUT:-$D}; M=$O/chain479.done; LOG=$O/chain479.log; rm -f $M
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 g15blx01"; T=/var/tmp/fasth3/t479
trap 'echo $? > "$M"' EXIT
log() { echo "[chain479] $(date -u '+%F %T') $*" | tee -a $LOG; }
for i in $(seq 240); do
  rc=$($SSH cat $T/detach/rb479.rc 2>/dev/null || true); [ -n "$rc" ] && break; sleep 30
done
log "build rc=${rc:-none}"; [ "$rc" = 0 ] || exit 10
col=$($SSH "cd $T/b && source /var/tmp/fasth3/t460/python_env/bin/activate && export HOME=/var/tmp/fasth3/home TMPDIR=/var/tmp/fasth3/tmp TT_METAL_HOME=$T/b PYTHONPATH=$T/b:$T/b/ttnn:$T/b/tools:/var/tmp/fasth3/t409/pydeps && timeout 300 python -m pytest --collect-only -q -p no:cacheprovider models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '10s and 4x8 and not 15s and not 4x32 and not WH' 2>&1 | tail -4" || true)
log "collect: $(echo "$col" | tr '\n' '|')"
echo "$col" | grep -q '1/[0-9]* tests\? collected\|^1 test collected' || exit 15
run() {
  local spec=$1 out id t
  # ADOPT=<job id>: wait on a job a killed chain already submitted instead of submitting again
  if [ -n "${ADOPT:-}" ]; then out="job=$ADOPT"; ADOPT=; else
  for t in $(seq 40); do
    if out=$($SUB $spec 2>&1); then break; fi
    log "submit $spec refused (try $t): $(echo "$out" | tail -2 | tr '\n' ' ')"; out=; sleep 300
  done; fi
  id=$(echo "$out" | sed -n 's/^job=//p'); [ -n "$id" ] || return 11
  log "$spec -> blx01 job $id"
  while :; do
    s=$($SSH "tt-device-mcp status -j $id" 2>/dev/null | grep '^Status:' || true)
    [ -n "$s" ] && ! echo "$s" | grep -qiE 'running|queued|pending' && break
    sleep 30
  done
  log "job $id: $($SSH "tt-device-mcp status -j $id" 2>/dev/null | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
}
done_d1() { $SSH "grep -q 'RuntimeError: T479_DUMP_DONE' $T/out_d1-sla0.85/run.log"; }
for t in 1 2 3; do
  run $D/spec_d1.txt || true
  done_d1 && break
  log "d1 try $t: $($SSH "tail -3 $T/out_d1-sla0.85/run.log" 2>/dev/null | tr '\n' ' ')"
done
done_d1 && log "d1 dumped all calls" || log "d1 did not finish; replaying what was dumped"
$SSH "cd $T && source /var/tmp/fasth3/t460/python_env/bin/activate && for f in \$(ls out_d1-sla0.85/dump/call*.pt | sort -V); do echo \"== \$f\"; nice -n 19 timeout 900 python -I replay479.py \$f 75449 0.85; done" > $O/replay479.txt 2>&1 || log "replay rc=$?"
$SSH "cat > $T/replay479.txt" < $O/replay479.txt
log "replay done: $(grep -c '^==' $O/replay479.txt) dumps"
done_d1
