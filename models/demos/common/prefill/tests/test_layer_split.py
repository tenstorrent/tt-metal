# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from models.demos.common.prefill.runners.runner_utils import compute_layer_split, ffn_handoff_flags


@pytest.mark.parametrize(
    "num_layers, num_ranks, tail_levels, expected",
    [
        (78, 1, 0, [78]),
        (78, 2, 0, [39, 39]),
        (78, 3, 0, [26, 26, 26]),
        (78, 4, 0, [19, 19, 20, 20]),
        (78, 5, 0, [15, 15, 16, 16, 16]),
        (78, 6, 0, [13] * 6),
        (78, 7, 0, [11] * 6 + [12]),
        (78, 8, 0, [9, 9] + [10] * 6),
        (78, 4, 1, [19, 20, 20, 19]),
        (61, 4, 0, [15, 15, 15, 16]),
        (61, 4, 1, [15, 15, 16, 15]),
    ],
)
def test_even_split_gives_the_remainder_to_the_last_ranks(monkeypatch, num_layers, num_ranks, tail_levels, expected):
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    ranges = compute_layer_split(num_layers, num_ranks, tail_levels=tail_levels)
    assert [count for _, count in ranges] == expected
    assert [start for start, _ in ranges] == [sum(expected[:r]) for r in range(num_ranks)]


@pytest.mark.parametrize(
    "env, num_ranks, expected",
    [
        (None, 4, [False, False, False, False]),
        ("1,0,1", 4, [True, False, True, False]),
        ("0", 2, [False, False]),
        ("1", 2, [True, False]),
    ],
)
def test_ffn_handoff_flags(monkeypatch, env, num_ranks, expected):
    # One flag per boundary, plus False for the last rank, which has no downstream stage.
    if env is None:
        monkeypatch.delenv("PREFILL_PP_FFN_HANDOFF", raising=False)
    else:
        monkeypatch.setenv("PREFILL_PP_FFN_HANDOFF", env)
    assert ffn_handoff_flags(num_ranks) == expected


def test_ffn_handoff_flags_rejects_a_wrong_boundary_count(monkeypatch, expect_error):
    monkeypatch.setenv("PREFILL_PP_FFN_HANDOFF", "1,0")
    with expect_error(ValueError, "must list 3 boundaries"):
        ffn_handoff_flags(4)
