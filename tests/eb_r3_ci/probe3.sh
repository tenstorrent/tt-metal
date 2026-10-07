#!/usr/bin/env bash
# Round 3 eltwise binary, third pass (#58725, #58726): the compute probe on the cases of test_eb_probe3.py, three passes.
cd /work
for i in 1 2 3; do
  echo "##### probe pass $i"; bash tests/eb_r3_ci/ab_envs.sh "EB_DUMMY=1" "EB_R3_PROBE_INIT=1" tests/eb_r3_ci/test_eb_probe3.py
done
