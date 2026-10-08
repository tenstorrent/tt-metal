#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the block sections, the broadcast sections and those with an activation, against main's program (ci5 specs e1, e2, f1).
cd /work
T=tests/eb_r3_ci
EB_RUN_LIMIT=3000 bash $T/dump_run.sh $T/spec_e1.txt
EB_RUN_LIMIT=2400 bash $T/dump_run.sh $T/spec_e2.txt
EB_RUN_LIMIT=2400 bash $T/dump_run.sh $T/spec_f1.txt
