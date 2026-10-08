#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the dest-reuse forms with the one-face rule, and the operand pass and native routing cases of the fourth and fifth passes.
cd /work
T=tests/eb_r3_ci
bash $T/dr9_dump.sh
EB_RUN_LIMIT=6000 bash $T/dump_run.sh $T/r9b2_dump.spec
