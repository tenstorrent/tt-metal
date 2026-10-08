#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 third review): single-core twins of reduce_to_root's compute kernel, post_sdpa's SDPA tail
# (rounds 1 and 2) and SDPA decode's output rescale on half tiles, main's per-face program against the whole-tile program
# (tests/eb_r3_ci/twins/optin_wt.txt), on a Blackhole P150 with the JIT reading every header from /work. Bits per twin family,
# the ELF sets of the two kernel caches, then the device time A/B (main optin optin main under the profiler) in three passes
# over the cases that passed on both sides.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
TW=tests/eb_r3_ci/twins/test_eb_twins.py
OPT=tests/eb_r3_ci/twins/optin_wt.txt
rm -f /tmp/tw_nodes.txt
for grp in ${TW_GROUPS:-r2r psdpa sdpad}; do
  echo "##### bits twin $grp"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh $OPT -p eb_seed_plugin $TW -k "test_twin_$grp"
  python3 - <<'PY' >> /tmp/tw_nodes.txt
import json
a = json.load(open("/tmp/ebbits/hash_main.json")); b = json.load(open("/tmp/ebbits/hash_optin.json"))
for t, o in a["outcome"].items():
    if o == "passed" and b["outcome"].get(t) == "passed":
        print(t)
PY
  echo "##### elf twin $grp"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "^tw_$grp$" 2>&1
done
echo "##### nodes passing on both sides: $(wc -l < /tmp/tw_nodes.txt)"
for p in 1 2 3; do
  echo "##### twins pass $p"; EB_RUN_LIMIT=2400 bash tests/eb_r3_ci/ab_set.sh $OPT --nodes-file /tmp/tw_nodes.txt
done
