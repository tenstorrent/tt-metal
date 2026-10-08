#!/usr/bin/env bash
# item 3 (third pass): the block-sharded H broadcast (bcast_h_sharded_optimised.cpp) with shards of 16 tile rows (8-tile DEST
# blocks, beyond the 4 tiles of fp32 DEST under the CI fp32 toggle) and of 4 tile rows (4-tile DEST blocks)
[[ -n "${HWLOCK_HELD:-}" || -n "${GITHUB_ACTIONS:-}" || -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ]] || { echo "not under hwlock" >&2; exit 2; }
bash tests/eb_r3_ci/dump_kedit.sh tests/eb_r3_ci/tog_bcast.txt tests/eb_r3_ci/test_eb_dump_kedit.py -k "test_bcast and H-bs"
