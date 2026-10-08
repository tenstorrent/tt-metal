#!/usr/bin/env bash
# item 3: ttnn.bcast MUL and rotate_half, PR kernels against main's (the bcast hand-off define removed)
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
bash tests/eb_r3_ci/dump_kedit.sh tests/eb_r3_ci/tog_bcast.txt tests/eb_r3_ci/test_eb_dump_kedit.py -k "test_bcast or test_rotate_half"
