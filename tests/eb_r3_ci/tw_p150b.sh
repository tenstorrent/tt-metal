#!/usr/bin/env bash
# Round 3 eltwise binary (#58723, #58724 review, twin round): three more device time passes of the twins whose first three
# passes were mixed or near the spread (reduce_to_root, reduce_to_one, GatedReduce, post_sdpa), P150, JIT root /work.
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
for p in 4 5 6; do
  echo "##### twins pass $p"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/twins/optin_twins.txt tests/eb_r3_ci/twins/test_eb_twins.py -k "test_twin_r2r or test_twin_r21 or test_twin_gr or test_twin_psdpa"
done
