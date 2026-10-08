#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass (#58725): the Python-scalar single section with the operand pass against the head without
# it, each test alone in its own processes; the 32-tile rows the control; three passes.
cd /work
IDS=$(python3 -c "import itertools; print(' '.join('-'.join(c) for c in itertools.product(('rsub_s','add_arelu_s'),('ws32_t4','hs8_t8','hs8_t32'),('bf16','bfp8'))))")
for i in 1 2 3; do
  for t in $IDS; do
    echo "##### pass $i iso6: head vs single-section pass [$t]"; bash tests/eb_r3_ci/ab_envs.sh "EB_R3_NONE=1" "EB_R3_PRE_ONE_SCALAR=1" tests/eb_r3_ci/test_eb_iso6.py -k "$t"
  done
done
