#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the HiFi3 rule against HiFi4 (spec h2 bfp4 ba).
cd /work
T=tests/eb_r3_ci
bash $T/hifi3_dump.sh $T/spec_h2_bfp4_ba.txt
