# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from models.demos.common.prefill.runners.runner_utils import compute_layer_split


@pytest.mark.parametrize(
    "num_layers, num_ranks, mtp_levels, remainder_last, expected",
    [
        (78, 1, 0, True, [78]),
        (78, 2, 0, True, [39, 39]),
        (78, 3, 0, True, [26, 26, 26]),
        (78, 4, 0, True, [19, 19, 20, 20]),
        (78, 5, 0, True, [15, 15, 16, 16, 16]),
        (78, 6, 0, True, [13] * 6),
        (78, 7, 0, True, [11] * 6 + [12]),
        (78, 8, 0, True, [9, 9] + [10] * 6),
        (78, 4, 1, True, [19, 20, 20, 19]),
        (61, 4, 0, False, [16, 15, 15, 15]),
        (61, 8, 0, False, [8] * 5 + [7] * 3),
        (78, 4, 0, False, [20, 20, 19, 19]),
        (78, 4, 1, False, [20, 20, 20, 18]),
    ],
)
def test_even_split_places_the_remainder(monkeypatch, num_layers, num_ranks, mtp_levels, remainder_last, expected):
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    ranges = compute_layer_split(num_layers, num_ranks, mtp_levels=mtp_levels, remainder_last=remainder_last)
    assert [count for _, count in ranges] == expected
    assert [start for start, _ in ranges] == [sum(expected[:r]) for r in range(num_ranks)]
