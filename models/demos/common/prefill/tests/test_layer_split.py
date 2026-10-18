# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""compute_layer_split's cost-balanced split: optimal, contiguous, and on the allowed starts. Host only."""

import itertools
import random

import pytest

from models.demos.common.prefill.runners.runner_utils import _balanced_counts, compute_layer_split


def _worst(costs, counts, tail):
    out, s = [], 0
    for i, n in enumerate(counts):
        out.append(sum(costs[s : s + n]) + (tail if i == len(counts) - 1 else 0.0))
        s += n
    return max(out)


def _brute(costs, ranks, tail, valid=None):
    n, best = len(costs), None
    for cuts in itertools.combinations(range(1, n), ranks - 1):
        if valid is not None and any(c not in valid for c in cuts):
            continue
        counts = [b - a for a, b in zip((0, *cuts), (*cuts, n))]
        w = _worst(costs, counts, tail)
        best = w if best is None else min(best, w)
    return best


@pytest.mark.parametrize("seed", range(40))
def test_balanced_counts_is_optimal(seed):
    rng = random.Random(seed)
    n, ranks = rng.randint(4, 12), rng.randint(1, 4)
    costs = [rng.choice([0.5, 1.0, 1.0, 1.3]) for _ in range(n)]
    tail = rng.choice([0.0, 0.0, 2.5, 6.9])
    valid = None if seed % 3 else set(rng.sample(range(1, n), k=min(n - 1, ranks + 2)))
    if valid is not None and _brute(costs, ranks, tail, valid) is None:
        with pytest.raises(ValueError):
            _balanced_counts(costs, ranks, tail, valid)
        return
    counts = _balanced_counts(costs, ranks, tail, valid)
    assert sum(counts) == n and len(counts) == ranks and min(counts) >= 1
    starts = list(itertools.accumulate(counts[:-1]))
    assert valid is None or all(s in valid for s in starts)
    assert _worst(costs, counts, tail) == pytest.approx(_brute(costs, ranks, tail, valid))


def test_glm53_mtp7_four_ranks(monkeypatch):
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS", raising=False)
    costs = [0.57] * 3 + [1.0] * 75
    split = compute_layer_split(78, 4, None, 7, layer_costs=(costs, 6 * 1.08 + 0.4))
    assert [n for _, n in split] == [22, 21, 21, 14]
    assert [s for s, _ in split] == [0, 22, 43, 64]


def test_override_and_default_unchanged(monkeypatch):
    monkeypatch.setenv("PREFILL_PP_LAYER_COUNTS", "20,20,20,18")
    assert compute_layer_split(78, 4, None, 7, layer_costs=([1.0] * 78, 7.0)) == [(0, 20), (20, 20), (40, 20), (60, 18)]
    monkeypatch.delenv("PREFILL_PP_LAYER_COUNTS")
    assert [n for _, n in compute_layer_split(78, 4, None, 0)] == [19, 19, 20, 20]
