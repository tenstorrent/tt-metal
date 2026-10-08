#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass (#58726): main's routing against native routing at the class boundaries; three passes.
cd /work
for i in 1 2 3; do
  echo "##### pass $i nat6: current vs native"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NO_NATIVE=1" "EB_R3_NATIVE_ALL=1" tests/eb_r3_ci/test_eb_r5.py -k "test_nat6 or test_nat7"
done
