#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 third review), twin round 2 on a P150: reduce_to_root's twin with the broadcast opt-in alone
# (optin_wt_rb.txt) and the standard opt-in alone (optin_wt_rh.txt), and post_sdpa's tail at HiFi4, HiFi2 and LoFi; main's
# program against each, bits once, three device time passes.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
TW=tests/eb_r3_ci/twins/test_eb_twins.py
for v in rb rh p; do
  K="test_twin_r2r"; [[ $v == p ]] && K="test_twin_psdpa"
  echo "##### bits $v"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/twins/optin_wt_$v.txt -p eb_seed_plugin $TW -k "$K"
  for p in 1 2 3; do
    echo "##### $v pass $p"; EB_RUN_LIMIT=2400 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/twins/optin_wt_$v.txt $TW -k "$K"
  done
done
