#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head: the strided reduce-scatter's twin with its standard multiplies per tile
# (ELTWISE_BINARY_PER_TILE_HANDOFF committed) against it with the broadcast hand-off too, broadcast gate; bits, then three
# passes of main optin optin main.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
TW=tests/eb_r3_ci/twins/test_eb_twins.py
OPT=tests/eb_r3_ci/r9opt/optin_srs_b.txt
echo "##### bits"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $TW -k "test_twin_srs and broadcast"
for p in 1 2 3; do
  echo "##### twins pass $p"; EB_RUN_LIMIT=1800 bash tests/eb_r3_ci/ab_set.sh $OPT --nodes-file tests/eb_r3_ci/r9opt/ids_b_twin_srs.txt
done
