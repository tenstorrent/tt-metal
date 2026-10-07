#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: the DiT fused distributed norms on one chip, device time A/B and bits (optin_x4.txt).
cd /work
O=tests/eb_r3_ci/optin_x4.txt
echo "##### selection: test_eb_dit.py"; bash tests/eb_r3_ci/ab_set.sh $O tests/eb_r3_ci/test_eb_dit.py
echo "##### selection: test_dit_rmsnorm_local single_device"; bash tests/eb_r3_ci/ab_set.sh $O models/tt_dit/tests/unit/test_dit_rmsnorm_local.py -k single_device
echo "##### bits"; bash tests/eb_r3_ci/bits_ab.sh $O tests/eb_r3_ci/test_eb_dit.py models/tt_dit/tests/unit/test_dit_rmsnorm_local.py -k single_device
