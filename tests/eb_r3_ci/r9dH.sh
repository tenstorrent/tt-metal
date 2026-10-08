#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the per-tile against per-face multiply, full cross products into bf16 and fp32 at four fidelities (specs d2x).
cd /work
T=tests/eb_r3_ci
EB_RUN_LIMIT=2400 bash $T/dump_run.sh $T/spec_d2x_bf16.txt
EB_RUN_LIMIT=4000 bash $T/dump_run.sh $T/spec_d2x_fp32.txt
