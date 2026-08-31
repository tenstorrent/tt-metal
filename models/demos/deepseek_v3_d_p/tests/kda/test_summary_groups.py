# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for KDA summary-group size resolution."""

from models.demos.deepseek_v3_d_p.tt.kda import ops


def test_grouped_scan_uses_largest_valid_configured_divisor() -> None:
    assert ops._effective_summary_group_chunks(160, 20) == 20
    assert ops._effective_summary_group_chunks(64, 20) == 16
    assert ops._effective_summary_group_chunks(88, 21) == 11
    assert ops._effective_summary_group_chunks(161, 8) == 7
