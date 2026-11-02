# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from models.demos.common.prefill.runners.runner_utils import compute_layer_split


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
    "num_ranks, tail_levels, expected",
    [
        (1, 0, [78]),
        # Every rank starts on a `full` layer, so no boundary needs the indexer top-k stream.
        (2, 0, [38, 40]),
        (4, 0, [18, 20, 20, 20]),
        # The MTP tail weighs on the last rank, which keeps the even split.
        (4, 1, [19, 20, 20, 19]),
    ],
)
def test_glm53_default_split_balances_cost(monkeypatch, num_ranks, tail_levels, expected):
    from models.demos.common.prefill.adapter import get_adapter

    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    adapter = get_adapter("glm_5_3")
    ranges = compute_layer_split(
        78, num_ranks, adapter.layer_split_boundaries(78), tail_levels, cost_model=adapter.layer_split_cost_model(78)
    )
    assert [count for _, count in ranges] == expected
    types = adapter.load_hf_config().indexer_types
    if tail_levels == 0:
        assert all(types[start] == "full" for start, _ in ranges)


def test_explicit_layer_counts_override_the_cost_model(monkeypatch):
    from models.demos.common.prefill.adapter import get_adapter

    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "39,39")
    adapter = get_adapter("glm_5_3")
    ranges = compute_layer_split(78, 2, None, 0, cost_model=adapter.layer_split_cost_model(78))
    assert [count for _, count in ranges] == [39, 39]
