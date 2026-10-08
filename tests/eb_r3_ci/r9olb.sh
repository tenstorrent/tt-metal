#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head (LoudBox 2x4): attn_res_gather_softmax's two defines, each removed against
# kept with the other kept, three passes.
cd /work
for i in 1 2 3; do
  for b in h b; do
    echo "##### pass $i off${b}lb: define removed vs head"
    grep attn_res tests/eb_r3_ci/r9opt/off_$b.txt > /tmp/off_lb_$b.txt
    EB_NODES_FILE=/work/tests/eb_r3_ci/r9opt/ids_${b}_loudbox.txt bash tests/eb_r3_ci/ab_off.sh /tmp/off_lb_$b.txt -p eb_select_plugin tests/ttnn/unit_tests/operations/experimental/test_attn_res_gather_softmax.py
  done
done
