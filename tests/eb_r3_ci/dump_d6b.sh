#!/usr/bin/env bash
# item 6 on the ci5 head: caller kernels, PR against main (each kernel's opt-in define removed), one kedit run per group so a
# hang stays in its group; the 2D pre-allgather last
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
T=tests/eb_r3_ci
export EB_RUN_LIMIT=1200
EB_KEDIT_DIR=/tmp/ebk_ln bash $T/dump_kedit.sh $T/tog_callers.txt $T/test_eb_dump_callers.py -k "test_ln_pre and not 2d"
EB_KEDIT_DIR=/tmp/ebk_misc bash $T/dump_kedit.sh $T/tog_callers.txt $T/test_eb_dump_callers.py -k "test_frgdn or test_rope_hf or test_indexer or test_softmax"
EB_KEDIT_DIR=/tmp/ebk_sm bash $T/dump_kedit.sh $T/tog_softmax_both.txt $T/test_eb_dump_callers.py -k test_softmax
EB_KEDIT_DIR=/tmp/ebk_2d bash $T/dump_kedit.sh $T/tog_callers.txt $T/test_eb_dump_callers.py -k "test_ln_pre_2d"
