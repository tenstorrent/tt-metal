#!/usr/bin/env bash
# items 3 and 6 on the ci5 head: ttnn.bcast MUL and rotate_half (bcast kernels), then the caller kernels
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
T=tests/eb_r3_ci
EB_RUN_LIMIT=1200 EB_KEDIT_DIR=/tmp/ebk_bc bash $T/dump_kedit.sh $T/tog_bcast.txt $T/test_eb_dump_kedit.py -k "test_bcast or test_rotate_half"
bash $T/dump_d6b.sh
