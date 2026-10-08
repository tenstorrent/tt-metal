#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (#58726): the broadcast sections' operand CB against L1 left free by a filler,
# one process from roomy to tight, main's program against the head (outcomes and bits), then the head's check per case.
cd /work
M="EB_R3_NO_BLOCK=1 EB_R3_NO_BCAST_CHUNK=1 EB_R3_MAIN_REINIT=1 EB_R3_PER_FACE=1 EB_R3_NO_HIFI3=1 EB_R3_NO_PRE_SECTIONS=1 EB_R3_NO_NATIVE=1"
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
echo "##### l1fill col bits: main vs head"; bash tests/eb_r3_ci/bits_envs.sh "$M" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_l1fill and add_arelu_col"
echo "##### l1fill col head plan"
EB_R3_LOG_RULE=1 timeout 1800 python3 -m pytest -p no:cacheprovider -v -s -rfE tests/eb_r3_ci/test_eb_r5.py -k "test_l1fill and add_arelu_col" 2>&1 \
  | grep -E "test_l1fill\[|EB_R3_RULE l1|PASSED|FAILED|TT_FATAL|TT_THROW|clash" | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-260
echo "##### l1fill col: head with the activation sections off (main's tile per section) against the head"
bash tests/eb_r3_ci/bits_envs.sh "EB_R3_NO_BCAST_ACT=1" "EB_DUMMY=1" tests/eb_r3_ci/test_eb_r5.py -k "test_l1fill and add_arelu_col"
