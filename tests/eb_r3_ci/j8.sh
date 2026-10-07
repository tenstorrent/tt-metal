#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 review): softmax's padded-tile dest-reuse add, three more passes of the attention softmax on
# widths that are not a multiple of 32 (fused scale mask) and, as the control whose program is the same on both sides, the
# plain softmax on those widths.
cd /work
SM=tests/ttnn/unit_tests/operations/fused/test_softmax.py
for i in 1 2 3; do
  echo "##### pass $i"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_smdr.txt $SM -k "attention_softmax_non_tile_aligned_width or test_softmax_non_divisible_width"
done
