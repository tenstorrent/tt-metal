#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723, #58724 review): softmax's dest-reuse add on the padded last tile, three more A/B passes of the
# fused scale-mask cases on a width that is not a multiple of 32; indexer_score bits with its direct init passing the hand-off.
cd /work
SM=tests/ttnn/unit_tests/operations/fused/test_softmax.py
I=tests/ttnn/nightly/unit_tests/operations/experimental/indexer_score/test_indexer_score.py
for i in 1 2 3; do
  echo "##### pass $i softmax padded dest reuse"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_smdr.txt $SM -k "scale_mask_softmax_non_divisible_width"
done
echo "##### bits indexer"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_rep_idx.txt -p eb_seed_plugin $I
