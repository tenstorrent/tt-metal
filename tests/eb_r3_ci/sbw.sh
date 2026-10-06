#!/usr/bin/env bash
# Round 3 eltwise binary: moreh softmax backward (W) with both opt-ins, the standard-form one alone and the broadcast one alone.
cd /work
for o in optin_sbw optin_sbw_t optin_sbw_b; do
  echo "##### $o"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/$o.txt --nodes-file tests/eb_r3_ci/nodes_sbw.txt
done
