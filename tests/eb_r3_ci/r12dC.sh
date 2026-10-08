#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, final head merged with main: full cross products per tile against per face into bf16 and fp32 (d2x).
cd /work
T=tests/eb_r3_ci
EB_RUN_LIMIT=2400 bash $T/dump_run.sh $T/r12_d2x_bf16.txt
EB_RUN_LIMIT=4000 bash $T/dump_run.sh $T/r12_d2x_fp32.txt
