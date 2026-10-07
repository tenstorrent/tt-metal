#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: bge_m3's balanced layernorm, main's kernel against the final one (standard, broadcast and row-broadcast
# dest-reuse multiplies per tile): device time A/B and bits.
cd /work
bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_bge_final.txt tests/eb_r3_ci/test_eb_r3_ops.py -k bge
bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_bge_final.txt tests/eb_r3_ci/test_eb_r3_ops.py -k bge
