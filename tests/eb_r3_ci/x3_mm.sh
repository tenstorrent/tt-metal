#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: the x3 selections whose bits differed, main's kernels against main's kernels.
cd /work
O=tests/eb_r3_ci/optin_none.txt
bash tests/eb_r3_ci/bits_ab.sh $O models/demos/blackhole/qwen36/tests/test_fused_recurrent_gdn.py tests/eb_r3_ci/test_eb_gdn.py
bash tests/eb_r3_ci/bits_ab.sh $O -p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py -k decode
