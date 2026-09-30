# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The fabric packet must carry whole CCL pages, which is not the same as the arch ceiling.

`ccl_common.cpp` warns when the configured packet differs from
`min(hw_max / page, max_scatter_write_chunks) * page`, and the served run took ttnn's 4352 B
default against 2048 B pages, three pages short of what the link allows. These cases pin the
Python derivation to the same arithmetic.
"""

import pytest

import ttnn
from models.demos.qwen38_27b_t3k.tt.generator import fabric_payload_bytes

# page bytes, wormhole packet, blackhole packet
_IDEAL = [
    (1088, 4352, 4352),  # bfloat8_b tile: four chunks on both, already ttnn's default
    (2048, 6144, 8192),  # bfloat16 tile: three pages on wormhole, four on blackhole
    (4096, 4096, 12288),
    (7616, 7616, 15232),
]


@pytest.mark.parametrize("page,wormhole,blackhole", _IDEAL)
def test_the_packet_carries_the_most_whole_pages_the_link_allows(monkeypatch, page, wormhole, blackhole):
    for arch, expected in (("wormhole_b0", wormhole), ("blackhole", blackhole)):
        monkeypatch.setattr(ttnn, "get_arch_name", lambda arch=arch: arch)
        assert fabric_payload_bytes(page) == expected


@pytest.mark.parametrize("page", [8192, 16384, 262144])
def test_a_page_wider_than_a_packet_falls_back_to_the_whole_packet(monkeypatch, page):
    # Matches the pages_per_packet == 0 branch in ccl_common.cpp rather than returning zero.
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    assert fabric_payload_bytes(page) == 7616


def test_an_unknown_arch_still_returns_whole_pages(monkeypatch):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "grayskull")
    assert fabric_payload_bytes(2048) == 8192


def test_the_default_page_is_the_prefill_collective_tile(monkeypatch):
    # bfloat16, the dtype ccl_dtype_prefill holds; the decode collective is already ideal.
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    assert fabric_payload_bytes() == 6144
