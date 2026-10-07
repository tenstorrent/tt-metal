#!/usr/bin/env bash
# Round 3 eltwise binary: llama rotary decode (sharded kernel) with sharded and interleaved cos/sin and trans_mat, twice.
cd /work
for i in 1 2; do
  echo "##### pass $i"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rlld.txt --nodes-file tests/eb_r3_ci/nodes_rll_cs.txt -p eb_unskip_plugin
done
