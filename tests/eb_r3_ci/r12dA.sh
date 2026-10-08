#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, rules of 16:40 / 17:40, final head merged with main (sfpi 7.86.0): binary_ng's block sections against
# main's program (spec e1), the dest-reuse forms per face against per tile (dr9), softmax's broadcast define removed against kept.
cd /work
T=tests/eb_r3_ci
EB_RUN_LIMIT=3000 bash $T/dump_run.sh $T/spec_e1.txt
bash $T/dr9_dump.sh
EB_RUN_LIMIT=1500 EB_KEDIT_DIR=/tmp/ebk_sm bash $T/dump_kedit.sh $T/tog_softmax_r9.txt $T/test_eb_dump_callers.py -k test_softmax
