#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: the DiT fused Welford layernorm at TP 1 on one chip, device time A/B and bits (optin_x4.txt).
cd /work
bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_x4.txt tests/eb_r3_ci/test_eb_dit.py -k test_dit_ln_tp1
bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_x4.txt tests/eb_r3_ci/test_eb_dit.py -k test_dit_ln_tp1
