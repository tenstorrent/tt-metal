# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only guard on the fused MM+RS shape table (no device required)."""

import ttnn


def test_fused_mmrs_table_excludes_corrupting_shape():
    """(22144, 1024, 5120) passes every unit context (35-replay trace, submesh,
    two-instance adjacency — see tests/nightly/tg/ccl/) yet corrupts the 35-step
    trunk (visual smear, frame std 46 vs 71 gold) — kept out of the fused table
    until a unit-level repro exists. down_proj (22144, 3200, 5120) is also out:
    op-level green at window 2, but its L1-resident MM window clashes with
    downstream matmul CBs in the traced trunk. Nothing on the 10x10 clamp may be
    fused until that lands (see the cosmos3 README, Fused MM+RS status)."""
    from models.tt_dit.utils.matmul import fused_mmrs_configs

    table = fused_mmrs_configs.get(ttnn.CoreCoord(10, 10), {})
    assert (22144, 3200, 5120) not in table
    assert (22144, 1024, 5120) not in table
