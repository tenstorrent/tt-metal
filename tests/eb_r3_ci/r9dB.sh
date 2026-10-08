#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): each post activation through an add, and the per-tile against per-face multiply at four fidelities with the dest-reuse forms (ci5 specs e3, e4).
cd /work
T=tests/eb_r3_ci
EB_RUN_LIMIT=2400 bash $T/dump_run.sh $T/spec_e3.txt
EB_RUN_LIMIT=4500 bash $T/dump_run.sh $T/spec_e4.txt
