#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review): three A/B passes of the HF rotary's standard-multiply switch on top of its broadcast
# one, and of moreh's small layer norm backward input-gradient kernel (BH skips lifted).
cd /work
L=tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py
for i in 1 2 3; do
  echo "##### pass $i hf rotary T over BCAST"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_hf.txt --nodes-file tests/eb_r3_ci/nodes_rep_hf.txt
  echo "##### pass $i moreh layer norm backward"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_mlnb.txt -p eb_unskip_plugin $L -k "backward and not large"
done
