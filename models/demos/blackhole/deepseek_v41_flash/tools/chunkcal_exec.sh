#!/bin/bash
# usage (through pfrun.sh): chunkcal_exec.sh <tag> "<KEY=val ...>" <sessions> <rowtok list>
# 40-layer in-process chunk-size sweep: one build, scenario i runs with DSV41_PREFILL_ROW_TOKENS = list[i]; no warmup run (total_replay_loop excludes the capture),
# failing scenarios (OOM) are logged and the larger budgets of the same ISL skipped. All other DSV41_* variables are unset.
tag=$1; flags=$2; sess=$3; rt=$4
for v in $(compgen -e | grep -E '^(DSV41_|MOE_COMPUTE_)'); do unset "$v"; done
export MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 DSV41_LAYERS=0-39 DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 DSV41_SESSION=$sess DSV41_ROWTOK_LIST=$rt
export DSV41_SESSION_NOWARM=1 DSV41_SESSION_CONTINUE=1
for kv in $flags; do export "$kv"; done
cd "$(git rev-parse --show-toplevel)"
echo "CC_ENV $(hostname -s) $(date +%FT%T) worktree=$(git rev-parse --short=11 HEAD) $(git status --short -uno | wc -l) modified files"
env | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)' | sort | sed 's/^/CC_ENV /'
log=/mnt/tt-data/ssinghal/dsv4-logs/pf_chunkcal_$tag.log
timeout 50000 pytest -x -s -q -o junit_suite_name=chunkcal_$tag models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k session &
pid=$!
( sleep 60; exec /mnt/tt-data/ssinghal/hangwatch.sh $pid $log 45 ) &
wait $pid; rc=$?
echo "CC_EXIT rc=$rc"
exit $rc
