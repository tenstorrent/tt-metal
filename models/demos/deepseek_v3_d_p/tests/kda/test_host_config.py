# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for KDA recurrence and program configuration."""

from models.demos.deepseek_v3_d_p.tt.kda import ops
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import _effective_qkv_channel_chunk_size


def test_grouped_scan_uses_largest_valid_configured_divisor() -> None:
    assert ops._effective_summary_group_chunks(160, 20) == 20
    assert ops._effective_summary_group_chunks(64, 20) == 16
    assert ops._effective_summary_group_chunks(88, 21) == 11
    assert ops._effective_summary_group_chunks(161, 8) == 7


def test_effective_qkv_channel_chunk_size_respects_ceiling_and_divisibility() -> None:
    assert _effective_qkv_channel_chunk_size(1280, 512) == 320
    assert _effective_qkv_channel_chunk_size(1280, 768) == 640
    assert _effective_qkv_channel_chunk_size(3072, 512) == 512
    assert _effective_qkv_channel_chunk_size(3072, 768) == 768


def test_qkv_channel_chunk_size_requires_positive_tile_multiple(expect_error) -> None:
    for invalid in (0, -32, 31, 33):
        with expect_error(ValueError, "qkv_channel_chunk_size must be a positive multiple"):
            KDAProgramConfig(qkv_channel_chunk_size=invalid)


def test_qkv_channel_chunk_size_accepts_tile_multiple() -> None:
    assert KDAProgramConfig(qkv_channel_chunk_size=512).qkv_channel_chunk_size == 512
