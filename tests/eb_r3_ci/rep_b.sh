#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 review): three more A/B passes of indexer_score and the fused recurrent gated delta rule.
cd /work
I=tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py
for i in 1 2 3; do
  echo "##### pass $i indexer accuracy"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_idx.txt $I -k accuracy
  echo "##### pass $i indexer shapes"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_idx.txt $I -k shapes
  echo "##### pass $i frgdn qwen36"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_frgdn.txt models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py -k fused
  echo "##### pass $i distributed welford layernorm"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rep_wlf.txt tests/eb_r3_ci/test_eb_x3.py -k welford
  echo "##### pass $i frgdn device fixture"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_frgdn.txt tests/eb_r3_ci/test_eb_frgdn.py
done
