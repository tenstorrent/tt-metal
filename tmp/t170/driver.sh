#!/bin/bash
# t170 driver on blx01 (g15blx01): the t164 eval pack, then phase 2, all under /var/tmp/fasth3/t170.
# Phase 1: one broker job per line of configs.txt, in order, each after the health gate and with no other
# smarton job running or queued. A failed job with broker errors around it is a drop and is rerun; two drops
# in a row skip that config. A baseline that fails or is skipped stops the driver.
# Then post.py (CPU: PCC/PSNR vs this run's baseline, same gen index and prompt) and the phase-2 pick: the 1-2
# configs with the lowest mean gen1/gen2 warm e2e that beat the baseline by 20 ms and pass a loose "not
# broken" gate (PCC >= 0.95, PSNR >= 25 dB on gen1 and gen2). Phase 2: baseline5 + <cfg>5 (default prompt,
# seeds 0-4, the ref_dv145 protocol) as separate jobs, then post.py on them. VBench runs on g15blx02.
# Marker: $V/DRIVER.done = "<code> <reason>"; pid in $V/driver.pid. Results: $T/res/<label>/.
# Usage: setsid nohup bash driver.sh > $V/driver.out 2>&1 &
F=/var/tmp/fasth3; T=$F/t170; O=$T/tree; W=$F/t48; R=$T/res
CONFIGS=${CONFIGS:-$T/configs.txt}; TAG=${TAG:-pack}
V=$T/driver_$TAG; mkdir -p $V $R; echo $$ > $V/driver.pid
SL=/var/log/tt-device-broker/server.log; D=$V/driver.log; BOOT0=$(uptime -s)
ENVY=$F/t159/env.yaml
# Job 399 (g15, default warmup, warm cache) held the device 162 s; blx01 loads ~30 s slower (job 621), and the
# pack adds gen#2: ~200 s, +50% = 300. Knob limits come from the measured baseline wall (+50%, +120 s for
# their unmeasured JIT compiles, at most 600); a knob job that times out is retried once at 600.
TO_BASE=${TO_BASE:-300}; TO_KNOB=${TO_KNOB:-450}; CAP=600
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T170_DRIVER_DONE $1 $2"; echo "$1 $2" > $V/DRIVER.done; exit 0; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | grep -qi 'upgrade' && { log "health: broker upgrade running"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE '🔧|health-gate|fabric-check|recover|bridge-reset|power-cycle' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  # The broker logs the end of its own reset-and-verify at ERROR level; that line means the gate passed.
  echo "$last" | grep -qE 'reset complete \+ health verified' && return 0
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_ready() {
  for i in $(seq 480); do
    health || { sleep 60; continue; }
    active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
    [ -n "$active" ] && { log "busy: $(echo $active | cut -c1-160)"; sleep 60; continue; }
    return 0
  done
  done_ 8 "broker not healthy or free for 8 h"
}
disk_ok() {
  free=$(df --output=avail -BG /var/tmp | tail -1 | tr -dc 0-9); log "/var/tmp free ${free} GiB, t170 $(du -sh $T | cut -f1)"
  [ "$free" -gt 50 ]
}
watch_job() {
  while :; do
    ST=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$ST" | grep -qiE "^Status: *(running|queued|pending)" || break
    sleep 30
  done
  log "job $JOB: $(echo "$ST" | grep -iE '^(Status|Exit|Runtime)' | tr -s ' ' | tr '\n' ' ')"
  sleep 30
  awk -v s="$T0" 'substr($0,1,19) >= s' $SL | grep -v DEBUG > $V/broker_slice_$JOB.log 2>/dev/null
  journalctl -k --since "$T0 UTC" --no-pager 2>/dev/null | grep -iE 'pcie|aer|tenstorrent|fatal|mce|link' > $V/journal_slice_$JOB.log
  EVID=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$EVID" ] && echo "$EVID" > $V/drop_evidence_$JOB.log
}
# run_one <label> <flags...>: returns 0 ok, 1 failed, 2 skipped after two drops in a row.
run_one() {
  label=$1; shift; drops=0; retried=0
  [ "$label" = "${PRIOR_DROP:-}" ] && drops=1
  # ADOPT=<label>:<job>: an earlier driver (killed by a host reboot) left that label's job queued or running
  # in the broker; watch it instead of submitting a duplicate.
  adopt=""; [ "${ADOPT%%:*}" = "$label" ] && adopt=${ADOPT#*:}
  to=$TO_KNOB; case $label in baseline|baseline5) to=$TO_BASE ;; esac
  while :; do
    A0=""
    if [ -n "$adopt" ]; then
      JOB=$adopt; adopt=""; A0=$(date +%s); log "$label adopting broker job $JOB"
      T0=$(now)
      while tt-device-mcp status -j $JOB 2>&1 | grep -qiE '^Status: *(queued|pending)'; do T0=$(now); sleep 30; done
    else
      wait_ready
      disk_ok || done_ 14 "/var/tmp under 50 GiB free before $label"
      T0=$(now)
      out=$(tt-device-mcp run-bg "env PYTEST_S=$((to - 30)) bash $T/run_cfg.sh $label $*" -w $O -e $ENVY -t $to 2>&1); src=$?
      log "$label submit rc=$src -t $to: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
      [ $src = 0 ] || done_ 7 "$label submit failed"
      JOB=$(echo "$out" | grep -oE '[0-9]+' | tail -1); echo "$label $JOB $T0 -t $to" >> $V/jobs.txt
    fi
    watch_job
    ok=0
    echo "$ST" | grep -qiE "^Status: *completed" && echo "$ST" | grep -qiE "^Exit: *0" \
      && grep -q "T164_EXIT\[$label\]=0" $R/$label/run.log 2>/dev/null && ok=1
    if [ $ok = 1 ]; then
      [ -n "$EVID" ] && log "$label job $JOB ok, but broker errors after it (drop logged, result kept)"
      log "$label OK job $JOB"; return 0
    fi
    if [ -n "$A0" ] && [ "$(stat -c %Y $R/$label/run.log 2>/dev/null || echo 0)" -lt "$A0" ]; then
      log "$label adopted job $JOB ended without running; submitting a new one"; continue
    fi
    if [ -n "$EVID" ]; then
      drops=$((drops + 1)); log "$label DROP #$drops job $JOB (evidence drop_evidence_$JOB.log)"
      [ $drops -ge 2 ] && { log "$label SKIPPED: two drops in a row"; return 2; }
      continue
    fi
    if [ $retried = 0 ] && [ $to -lt $CAP ] && { grep -qE '\+ Timeout \+' $R/$label/run.log 2>/dev/null || echo "$ST" | grep -qiE 'timeout|timed out'; }; then
      retried=1; to=$CAP; log "$label job $JOB timed out without broker errors; one retry at -t $to"; continue
    fi
    log "$label FAILED job $JOB without broker errors"; return 1
  done
}
run_list() {
  while read -r label flags <&3; do
    [ -z "$label" ] || [ "${label:0:1}" = "#" ] && continue
    if grep -q "T164_EXIT\[$label\]=0" $R/$label/run.log 2>/dev/null; then log "$label already done, skipped"; continue; fi
    run_one $label $flags; rc=$?
    echo "$label rc=$rc job $JOB" >> $V/results.txt
    if [ "${label#baseline}" != "$label" ]; then
      [ $rc != 0 ] && done_ 6 "$label did not complete (rc $rc); the pack needs it"
      w=$(grep -oE 'process wall [0-9]+' $R/$label/run.log | grep -oE '[0-9]+$')
      [ -n "$w" ] && TO_KNOB=$(( w * 3 / 2 + 120 )) && [ $TO_KNOB -gt $CAP ] && TO_KNOB=$CAP
      log "$label wall ${w:-?} s -> knob limit $TO_KNOB s"
    fi
  done 3< $1
}
post() {
  (cd $O && source $W/python_env/bin/activate && export HOME=$F/home TMPDIR=$F/tmp HF_HUB_OFFLINE=1 \
    PYTHONPATH=$O:$W/ttnn:$W/tools LTX_EVAL_THREADS=8 && nice -n 19 timeout 3000 python $T/post.py $R $1) >> $V/post.log 2>&1
  log "post.py $1 rc=$?"
}
log "start tag=$TAG configs=$CONFIGS boot=$BOOT0 pid=$$"
run_list $CONFIGS
post $CONFIGS
COMMON="LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0"
P=$T/phase2; mkdir -p $P
$W/python_env/bin/python - $R/summary_configs.json $CONFIGS "$COMMON" > $P/configs5.txt <<'EOF'
import json, sys
res, lines, common = json.load(open(sys.argv[1])), open(sys.argv[2]).read().splitlines(), sys.argv[3]
flags = {l.split()[0]: " ".join(l.split()[1:]) for l in lines if l.strip() and not l.startswith("#")}
def mean_e2e(v):
    t = [v["gens"][g]["e2e"] for g in ("1", "2") if "e2e" in v.get("gens", {}).get(g, {})]
    return sum(t) / len(t) if len(t) == 2 else None
def sane(v):
    q = [v["gens"].get(g, {}).get("quality") for g in ("1", "2")]
    return all(x and x["pcc"] >= 0.95 and x["psnr"] >= 25 for x in q)
b = res.get("baseline", {})
base = mean_e2e(b) if b.get("exit") == "0" else None
if base is None:
    sys.exit(0)
picks = sorted((mean_e2e(v), c) for c, v in res.items()
               if c != "baseline" and v.get("exit") == "0" and mean_e2e(v) and mean_e2e(v) < base - 0.02 and sane(v))
print(f"baseline5 {common}")
for _, c in picks[:2]:
    print(f"{c}5 {flags[c]} {common}")
EOF
log "configs5: $(tr '\n' ';' < $P/configs5.txt)"
[ "$(wc -l < $P/configs5.txt)" -ge 2 ] || done_ 20 "pack done; no config beat the baseline and passed the gate (summary_configs.md)"
run_list $P/configs5.txt
post $P/configs5.txt
disk_ok
done_ 0 "pack and phase 2 device jobs done"
