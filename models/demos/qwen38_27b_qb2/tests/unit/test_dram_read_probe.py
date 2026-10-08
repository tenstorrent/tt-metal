# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Coverage and packet-tail checks independent of Metal or an allocated device."""

import pytest

from models.demos.qwen38_27b_qb2.tt.dram_read_probe import (
    PAGE_BYTES,
    WORDS,
    assignments,
    expected_markers,
    pattern_word,
    validate_ring,
    variants,
)


@pytest.mark.parametrize("pages", [8, 128, 2056, 262144])
@pytest.mark.parametrize(
    "mode,placement", [("interleaved_grid", "grid"), ("bank_tiles", "row"), ("bank_bulk", "pinned")]
)
def test_every_page_owned_exactly_once_including_uneven_worker_tails(pages, mode, placement):
    work = assignments(pages, mode, placement)
    assert len(set((x, y) for x, y, *_ in work)) == len(work)
    covered = [p for _, _, first, stride, count in work for p in range(first, first + stride * count, stride)]
    assert sorted(covered) == list(range(pages))
    if mode != "interleaved_grid":
        assert all(first == worker and stride == 8 for worker, (_, _, first, stride, _) in enumerate(work))


@pytest.mark.parametrize("packet_pages", [1, 4, 8, 15])
def test_markers_include_last_short_packet_and_order(packet_pages):
    work = assignments(2056, "bank_bulk", "pinned")[6]
    expected = expected_markers(work, packet_pages, 100)
    raw = [[pattern_word(page, word, 100) for word in range(WORDS)] for page in range(6, 2056, 8)]
    blocks = [sum(raw[i : i + packet_pages], []) for i in range(0, len(raw), packet_pages)]
    assert expected[:2] == [257, len(blocks)]
    assert expected[2] == sum(block[0] for block in blocks) % 2**32
    assert expected[3] == sum(block[-1] for block in blocks) % 2**32
    assert expected[4] == sum((i + 1) * (block[0] ^ block[-1]) for i, block in enumerate(blocks)) % 2**32
    reordered = sum((i + 1) * (block[0] ^ block[-1]) for i, block in enumerate(reversed(blocks))) % 2**32
    assert reordered != expected[4]
    assert expected_markers(work, packet_pages, 101)[2:5] != expected[2:5]


def test_all_variants_have_legal_disjoint_ring_storage():
    cases = variants()
    assert len(cases) == len({tuple(sorted(row.items())) for row in cases}) == 11
    for row in cases:
        validate_ring(row["packet_pages"], row["depth"])
        assert row["packet_pages"] * PAGE_BYTES <= 16384
        assert row["depth"] * row["packet_pages"] * PAGE_BYTES <= 131072


@pytest.mark.parametrize("packet_pages,depth", [(0, 1), (16, 1), (4, 0), (4, 3), (4, 16), (True, 4)])
def test_reject_out_of_range_packets_and_tags(packet_pages, depth, expect_error):
    message = (
        "Packets must contain" if type(packet_pages) is not int or not 1 <= packet_pages <= 15 else "Ring depth must be"
    )
    with expect_error(ValueError, message):
        validate_ring(packet_pages, depth)


@pytest.mark.parametrize(
    "pages,mode,placement,message",
    [
        (7, "bank_bulk", "pinned", "Page count must be"),
        (9, "bank_bulk", "row", "Page count must be"),
        (128, "bank_bulk", "grid", "Bank readers need"),
        (128, "interleaved_grid", "pinned", "Interleaved-grid control"),
    ],
)
def test_reject_unrepresentable_bank_coverage(pages, mode, placement, message, expect_error):
    with expect_error(ValueError, message):
        assignments(pages, mode, placement)
