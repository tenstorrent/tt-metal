#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, final head merged with main: main's program (every opt-in define removed, no block section) against
# the head at the shapes Blackhole models run (test_eb_r11.py), one pass. usage: r12fin.sh <-k expression>
cd /work
export EB_RUN_LIMIT=2400 EB_REPS=4 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
echo "##### $(date -u +%T) fin: off=tests/eb_r3_ci/r12/off_all.txt main_env=EB_R3_NO_BLOCK=1 k=$1"
EB_K_EXPR="$1" bash tests/eb_r3_ci/ab_r10.sh tests/eb_r3_ci/r12/off_all.txt EB_R3_NO_BLOCK=1 -p eb_k_plugin tests/eb_r3_ci/test_eb_r11.py
echo "##### end $(date -u +%T)"
