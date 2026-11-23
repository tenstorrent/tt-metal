# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from models.demos.common.prefill.runners.host_hop import HOST_HOP_RANKS_ENV, host_hop_sender_ranks


@pytest.mark.parametrize(
    "raw, num_ranks, expected",
    [
        (None, 8, set()),
        ("", 8, set()),
        ("0", 2, {0}),
        ("1,3,5", 8, {1, 3, 5}),
        (" 1, 3 ,5,", 8, {1, 3, 5}),
    ],
)
def test_sender_ranks_come_from_the_env(monkeypatch, raw, num_ranks, expected):
    if raw is None:
        monkeypatch.delenv(HOST_HOP_RANKS_ENV, raising=False)
    else:
        monkeypatch.setenv(HOST_HOP_RANKS_ENV, raw)
    assert host_hop_sender_ranks(num_ranks) == expected


@pytest.mark.parametrize("raw, num_ranks", [("7", 8), ("-1", 8), ("0", 1), ("1,x", 8)])
def test_a_rank_with_no_outbound_boundary_is_rejected(monkeypatch, raw, num_ranks):
    monkeypatch.setenv(HOST_HOP_RANKS_ENV, raw)
    with pytest.raises(ValueError, match=HOST_HOP_RANKS_ENV):
        host_hop_sender_ranks(num_ranks)
