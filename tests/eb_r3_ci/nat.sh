#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (#58726): a block or width sharded a with a column, scalar or row b in DRAM, and a height
# sharded a with a row b: binary_ng's current routing (off the native sharded path) against native routing. Three passes.
cd /work
T=tests/eb_r3_ci/test_eb_r3_mp.py
for i in 1 2 3; do
  echo "##### pass $i native: current routing vs native"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_NATIVE_BCAST=1 EB_R3_NATIVE_ROW=1" $T -k test_nat
done
