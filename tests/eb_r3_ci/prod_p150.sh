#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58723 third review), P150, JIT root /work. SDPA decode on half tiles (the Llama2-70B single-iteration
# case at HiFi2 and paged llama3.1 at HiFi4): the whole-tile column broadcast against the PR head's per-face one
# (optin_dec_head.txt adds EB_CI_NO_1X2, so "main" is the new program and "optin" the PR head's). The sdpa_tail micro op
# (post_sdpa's SDPA tail on one core, HiFi4): main's program against SDPA_BCAST_COL_REUSE_PER_TILE_HANDOFF. Bits of each, then
# three device time passes.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
DEC=tests/eb_r3_ci/nodes_dec.txt
ST=models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_tail.py
if [[ -z "${PROD_ONLY:-}" || "${PROD_ONLY}" == dec ]]; then
  echo "##### bits decode"; EB_NODES_FILE=$(readlink -f $DEC) EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_dec_head.txt -p eb_seed_plugin -p eb_select_plugin $(cut -d: -f1 $DEC | sort -u)
  echo "##### elf decode"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "sdpa_flash_decode" 2>&1 | head -20
fi
if [[ -z "${PROD_ONLY:-}" || "${PROD_ONLY}" == stail ]]; then
  echo "##### bits sdpa_tail"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_stail.txt -p eb_seed_plugin $ST
  echo "##### elf sdpa_tail"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebbits/cache_main /tmp/ebbits/cache_optin "sdpa_tail" 2>&1 | head -20
fi
for p in 1 2 3; do
  if [[ -z "${PROD_ONLY:-}" || "${PROD_ONLY}" == dec ]]; then
    echo "##### decode pass $p"; EB_RUN_LIMIT=2400 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_dec_head.txt --nodes-file $DEC
  fi
  if [[ -z "${PROD_ONLY:-}" || "${PROD_ONLY}" == stail ]]; then
    echo "##### sdpa_tail pass $p"; EB_RUN_LIMIT=1500 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_stail.txt $ST
  fi
done
