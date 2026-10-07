#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): three more A/B passes of the left-out distributed layernorm, KDA chunk scan and
# rgb_to_yuv kernels, to pool with the first pass.
cd /work
for i in 1 2 3; do
  echo "##### pass $i n2"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_n2.txt --nodes-file tests/eb_r3_ci/nodes_n2.txt
  echo "##### pass $i ex"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_ex.txt --nodes-file tests/eb_r3_ci/nodes_rep_ex.txt
done
