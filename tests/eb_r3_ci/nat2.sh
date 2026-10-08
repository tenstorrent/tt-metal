#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (#58726): native routing for a block or width sharded a with a column or scalar b, with
# activations and a Float32 output: current routing against native, and against native with DEST sections for any layout
# with an activation (EB_R3_SECTIONS_ANY). Three passes.
cd /work
T=tests/eb_r3_ci/test_eb_r3_mp.py
for i in 1 2 3; do
  echo "##### pass $i nat: current routing vs native"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_NATIVE_BCAST=1" $T -k test_nat2
  echo "##### pass $i natsec: current routing vs native with sections"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_NATIVE_BCAST=1 EB_R3_SECTIONS_ANY=1" $T -k test_nat2
done
