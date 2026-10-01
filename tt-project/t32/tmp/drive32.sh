#!/bin/bash
# t32 driver on g15blx02: runs the blx03 sweep jobs one at a time, after job $1 (already queued).
# Stops on the first job that does not complete with exit 0. Ends with DRIVE32_DONE.
L=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t32/tmp/drive32.log
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
id=$1; echo "JOB[s2_splitk]=$id"; wait_job $id
ok $id || { echo "STOP job $id not ok"; echo DRIVE32_DONE; exit 1; }
for spec in "s1_splitk|stage_1 LTX_SWEEP_SELF= LTX_SWEEP_CROSS_K= LTX_SWEEP_OPS=5" \
            "s1_self|stage_1 LTX_SWEEP_SELF=96,608_96,416 LTX_SWEEP_CROSS_K= LTX_SWEEP_SPLITK=0 LTX_SWEEP_OPS=5" \
            "s2_self|stage_2 LTX_SWEEP_SELF=384,256_192,448_128,608 LTX_SWEEP_CROSS_K= LTX_SWEEP_SPLITK=0 LTX_SWEEP_OPS=5"; do
  name=${spec%%|*}; id=$(submit ${spec#*|}); echo "JOB[$name]=$id"
  [ -z "$id" ] && { echo "STOP submit failed"; break; }
  wait_job $id; ok $id || { echo "STOP job $id not ok"; break; }
done
echo DRIVE32_DONE
