# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the lane grouping of DSV41_XROW_BATCH (T24): U key blocks per save call through the U lanes of the receiving row."""
from models.demos.blackhole.deepseek_v41_flash.tt.apc_paged_model import xrow_key_groups


def test_every_block_exactly_once_with_distinct_lanes():
    for U in (2, 4):
        for n in (0, 1, 2, 3, 7, 126, 1345):
            groups = xrow_key_groups(n, U)
            flat = [k for g in groups for _, k in g]
            assert flat == list(range(n))
            assert all(len({j for j, _ in g}) == len(g) <= U for g in groups)
            assert len(groups) == -(-n // U)  # 1/U of the save calls
