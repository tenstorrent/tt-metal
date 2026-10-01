#!/bin/bash
# t27 on blx03: prewarm + process run (3-stage wrapper), wait, then thread run on the warm cache. Ends with DRIVE27_DONE.
cd /home/smarton/fasth3/t27
L=tmp/drive27.log
jid() { grep -oiE 'job[^0-9]{0,12}[0-9]+' | grep -oE '[0-9]+' | tail -1; }
out=$(bash tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh -e tmp/blx03_env.yaml -w $PWD -t 590 -- "bash tmp/blx03_ab.sh process" 2>&1)
echo "$out" | tail -20
P=$(echo "$out" | tail -5 | jid); echo "PROCESS_JOB=$P"
[ -n "$P" ] && tt-device-mcp wait $P > tmp/job_process.log 2>&1
out=$(tt-device-mcp run-bg "bash tmp/blx03_ab.sh thread" -w $PWD -t 590 -e tmp/blx03_env.yaml 2>&1); echo "$out"
T=$(echo "$out" | jid); echo "THREAD_JOB=$T"
[ -n "$T" ] && tt-device-mcp wait $T > tmp/job_thread.log 2>&1
grep -hE "RUN_EXIT|E2E_WALL_S gen#[23]" tmp/job_process.log tmp/job_thread.log
echo "DRIVE27_DONE"
