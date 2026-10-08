# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Placement and legal packet geometry; hardware test proves the actual protocol."""

import pytest

from models.demos.qwen38_27b_qb2.tt.dram_delivery_probe import consumer_cores, variants
from models.demos.qwen38_27b_qb2.tt.dram_read_probe import PAGE_BYTES, PINNED_BANK_CORES, validate_ring


@pytest.mark.parametrize("placement", ["near", "center", "opposite"])
def test_sender_receiver_roles_never_share_a_core(placement):
    consumers = consumer_cores(placement)
    assert len(set(consumers)) == 8
    assert not set(consumers) & set(PINNED_BANK_CORES)
    assert all(0 <= x < 12 and 0 <= y < 10 for x, y in consumers)


def test_bounded_transfer_and_unique_sweep():
    rows = variants()
    assert len(rows) == len({tuple(sorted(row.items())) for row in rows}) == 6
    for row in rows:
        validate_ring(row["packet_pages"], row["depth"])
        assert row["packet_pages"] * PAGE_BYTES <= 16384
        assert row["depth"] * row["packet_pages"] * PAGE_BYTES <= 65536


def test_reject_bad_placement(expect_error):
    with expect_error(ValueError, "Consumer placement must be"):
        consumer_cores("unknown")
