#!/usr/bin/env bash
# item 6 on the ci5 head: indexer_score_dsa (compute_indexer_score.cpp) at a valid causal start
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
T=tests/eb_r3_ci
EB_RUN_LIMIT=1200 EB_KEDIT_DIR=/tmp/ebk_ix bash $T/dump_kedit.sh $T/tog_callers.txt $T/test_eb_dump_callers.py -k "test_indexer"
