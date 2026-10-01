#!/bin/bash
# t32 driver part 2: S1 self (+ split-K) then S2 self on blx03, one job at a time. Ends with DRIVE32B_DONE.
R="ssh -o BatchMode=yes g14blx03"
wait_job() { while :; do s=$($R "tt-device-mcp status -j $1" 2>&1); grep -qiE "running|queued" <<<"$s" || break; sleep 60; done; echo "$s" | head -5; }
ok() { $R "tt-device-mcp logs -n 100000 $1" 2>/dev/null | grep -q "RUN_EXIT=0"; }
submit() {
  while :; do
    out=$(ttp lock g14blx03-device -- $R "cd ~/fasth3/t32 && active=\$(tt-device-mcp status 1 2>&1 | sed -n '/^RUNNING/,/^RECENT/p' | grep -w smarton); [ -n \"\$active\" ] && exit 75; tt-device-mcp run-bg \"bash tmp/blx03_sweep.sh $*\" -e tmp/env.yaml -w \$PWD -t 600" 2>&1); rc=$?
    [ $rc -eq 75 ] && { sleep 120; continue; }; break
  done
  grep -oE "Job [0-9]+" <<<"$out" | grep -oE "[0-9]+"
}
for spec in "s1_self_splitk|stage_1 LTX_SWEEP_SELF=96,608_96,416 LTX_SWEEP_CROSS_K= LTX_SWEEP_OPS=5" \
            "s2_self|stage_2 LTX_SWEEP_SELF=384,256_192,448_128,608 LTX_SWEEP_CROSS_K= LTX_SWEEP_SPLITK=0 LTX_SWEEP_OPS=5"; do
  name=${spec%%|*}; id=$(submit ${spec#*|}); echo "JOB[$name]=$id"
  [ -z "$id" ] && { echo "STOP submit failed"; break; }
  wait_job $id; ok $id || { echo "STOP job $id not ok"; break; }
done
echo DRIVE32B_DONE
