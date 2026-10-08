#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass (#58725): sharded ops with an operand activation; the head against one more binary init
# per DEST section (probe), against the operand pass over 4 sections before one init (multi4), and main's multiply hand-off
# against multi4. Three passes.
cd /work
T=tests/eb_r3_ci/test_eb_r3_mp.py
for i in 1 2 3; do
  echo "##### pass $i probe: head vs one more init per section"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_PROBE_CHUNK_INIT=1" $T -k test_mp
  echo "##### pass $i multi4: head vs the operand pass over 4 sections"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_MULTI_PASS=4" $T -k test_mp
  echo "##### pass $i main4: main's hand-off vs multi4"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_PER_FACE=1" "EB_R3_MULTI_PASS=4" $T -k test_mp
done
echo "##### multi2: head vs the operand pass over 2 sections"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_MULTI_PASS=2" $T -k test_mp
