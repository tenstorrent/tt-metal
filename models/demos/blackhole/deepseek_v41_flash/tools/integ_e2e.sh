#!/bin/bash
# usage (through pfrun.sh): integ_e2e.sh <tag> "<KEY=val ...>" <sessions> [DSV41_PFA_AB spec, e.g. "-|OPT"]
# 40-layer demo session(s) with the grid's environment (all other DSV41_* variables unset). Paired A/B inside one process:
#   DSV41_PFA_AB="-|OPT" (baseline attention knobs | umbrella DSV41_PREFILL_OPT=1) and DSV41_PF_ASYNC_LIST="0,1" cycle per scenario.
tag=$1; flags=$2; sess=$3; ab=$4
for v in $(compgen -e | grep -E '^(DSV41_|MOE_COMPUTE_)'); do unset "$v"; done
export MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 DSV41_LAYERS=0-39 DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 DSV41_SESSION=$sess
for kv in $flags; do export "$kv"; done
[ -n "$ab" ] && export DSV41_PFA_AB="$ab"
cd "$(git rev-parse --show-toplevel)"
echo "E2E_ENV $(hostname -s) $(date +%FT%T) worktree=$(git rev-parse --short=11 HEAD) $(git status --short -uno | wc -l) modified files"
env | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)' | sort | sed 's/^/E2E_ENV /'
exec pytest -x -s -q -o junit_suite_name=integ_$tag models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k session
