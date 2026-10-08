#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, final head merged with main: the per-tile LLK program against the per-face one through binary_ng's
# multiply (CI toggle EB_R3_PER_TILE) at four fidelities, the broadcast and scalar kernels, the dest-reuse forms (spec e4).
cd /work
T=tests/eb_r3_ci
EB_RUN_LIMIT=4500 bash $T/dump_run.sh $T/r12_e4.txt
