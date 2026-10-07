#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: topk_router_gpt and bge_m3's balanced layernorm with the broadcast opt-in (device time A/B, bits),
# and the balanced layernorm with its row-broadcast dest-reuse ops at LoFi (device time A/B: the bound on that multiply's hold).
cd /work
T=tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py
echo "##### topk"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_x6.txt $T -k deterministic
echo "##### bge"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_x6.txt tests/eb_r3_ci/test_eb_r3_ops.py -k bge
echo "##### bge lofi"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_bge_lofi.txt tests/eb_r3_ci/test_eb_r3_ops.py -k bge
echo "##### bits"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_x6.txt $T tests/eb_r3_ci/test_eb_r3_ops.py -k "deterministic or bge"
