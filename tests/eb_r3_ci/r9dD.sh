#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (sfpi 7.86.0): the HiFi3 rule against HiFi4 (specs h1, h2 bf16 ab, h2 bfp8 ab).
cd /work
T=tests/eb_r3_ci
bash $T/hifi3_dump.sh $T/spec_h1.txt
bash $T/hifi3_dump.sh $T/spec_h2_bf16_ab.txt
bash $T/hifi3_dump.sh $T/spec_h2_bfp8_ab.txt
