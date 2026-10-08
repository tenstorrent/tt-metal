#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head: the DiT fused-norm kernels' defines ($1: h or b) each removed against kept,
# the other kept, three passes; the tests by -k (their node ids did not match in r9oh / r9ob).
cd /work
export EB_RUN_LIMIT=2400 EB_K_EXPR="test_dit_tp1 or test_dit_ln_tp1 or (test_local_rmsnorm_rope and single_device)"
grep dit_ tests/eb_r3_ci/r9opt/off_$1.txt > /tmp/off_dit.txt; cat /tmp/off_dit.txt
for i in 1 2 3; do
  echo "##### pass $i off${1}dit: define removed vs head"
  bash tests/eb_r3_ci/ab_off.sh /tmp/off_dit.txt -p eb_k_plugin tests/eb_r3_ci/test_eb_dit.py models/tt_dit/tests/unit/test_dit_rmsnorm_local.py
  grep -h "collected\|EB_SELECT" /tmp/eboff/log_1_main.txt | head -3
done
