#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the whole-tile programs: the SDPA tail column multiply and reduce_to_root's partial-face columns, per tile against per face.
cd /work
T=tests/eb_r3_ci
bash $T/dump8.sh 'sdpa or control'
bash $T/dump8.sh 'control or none or (col and 8x32) or (col and 4x32)'
bash $T/dump8.sh '(col and 2x32) or (col and 1x32)'
