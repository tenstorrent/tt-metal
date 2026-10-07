#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (#58726): native routing of block and width sharded broadcasts without activations, on
# grids of 2 to 64 cores and shards of 64 to 256 tiles; three passes.
cd /work
T=tests/eb_r3_ci/test_eb_r3_mp.py
for i in 1 2 3; do
  echo "##### pass $i nat3: current routing vs native"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_NATIVE_BCAST=1" $T -k test_nat3
done
