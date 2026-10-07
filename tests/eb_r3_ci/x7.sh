#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: fused recurrent GDN on the device fixture (device time A/B, seeded bits), and the llama rotary decode
# selection again (its run 4 aborted in batch x3).
cd /work
echo "##### frgdn"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_frgdn.txt tests/eb_r3_ci/test_eb_frgdn.py
echo "##### frgdn bits"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_frgdn.txt -p eb_seed_plugin tests/eb_r3_ci/test_eb_frgdn.py
echo "##### rotary llama decode"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_rlld.txt -p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py -k decode
