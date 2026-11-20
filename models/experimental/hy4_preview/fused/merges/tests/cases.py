# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Survivor lists for the top-k merge tests: per chip and user 2048 fused words [bf16 score | stamp] in rank order,
as the index-key scan leaves them, built to have many exact ties within and across chips."""

import torch

from models.experimental.hy4_preview.fused.merges import topk_merge as tm

KINDS = ("ties", "dense", "one_value", "short", "ragged", "exact", "edge", "empty_chips", "negative")


def _lists(chips, seed, kind):
    """(score bits int64 [chips, 2048], seen bool [chips, 2048]) of one user."""
    g = torch.Generator().manual_seed(seed)
    n = tm.LIST
    seen = torch.ones(chips, n, dtype=torch.bool)
    rand = lambda *shape: torch.rand(*shape, generator=g)
    if kind == "ties":  # a coarse grid of values of both signs: long runs of equal scores on every chip
        grid = torch.tensor([-3.0, -1.5, -0.75, -0.25, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0])
        scores = grid[torch.randint(0, grid.numel(), (chips, n), generator=g)]
    elif kind == "dense":  # every chip's list inside one or two high-byte buckets, some 8 entries per key
        scores = 8.0 + 20.0 * rand(chips, n)
    elif kind == "one_value":
        scores = torch.full((chips, n), 0.37109375)
    elif kind == "negative":
        scores = -(0.5 + 3.0 * rand(chips, n))
    elif kind == "edge":  # keys around the visibility limit and across exponent boundaries
        scores = -torch.pow(2.0, torch.randint(90, 98, (chips, n), generator=g).float()) * (1 + rand(chips, n))
        scores[:, ::3] = torch.randn(chips, n, generator=g)[:, ::3] * 1e-3
    else:
        scores = torch.randn(chips, n, generator=g) * torch.pow(
            2.0, torch.randint(-3, 4, (chips, n), generator=g).float()
        )
        scores = (scores * 8).round() / 8  # ties
        if kind == "short":  # fewer than 2048 visible entries in total, some chips with none
            counts = torch.randint(0, max(2, 2 * (n - 1) // chips), (chips,), generator=g)
            counts[torch.randperm(chips, generator=g)[: max(1, chips // 4)]] = 0
            while int(counts.sum()) >= n:
                counts = counts // 2
        elif kind == "exact":  # exactly 2048 visible entries in total
            cuts = torch.sort(torch.randint(0, n + 1, (chips - 1,), generator=g)).values
            counts = torch.diff(torch.cat([torch.tensor([0]), cuts, torch.tensor([n])]))
        elif kind == "empty_chips":
            counts = torch.full((chips,), n)
            counts[torch.randperm(chips, generator=g)[: chips // 2]] = 0
            counts[0] = n
        else:  # ragged
            counts = torch.randint(0, n + 1, (chips,), generator=g)
            counts[0] = n
        seen = torch.arange(n)[None, :] < counts[:, None]
    return tm.bf16_bits(scores), seen


def make(chips: int, users: int, seed: int, kinds=None) -> torch.Tensor:
    """int64 [chips, users, 2048]: user u's lists are of kind kinds[u % len(kinds)]."""
    kinds = KINDS if kinds is None else kinds
    g = torch.Generator().manual_seed(seed + 99)
    out = torch.zeros(chips, users, tm.LIST, dtype=torch.int64)
    for u in range(users):
        bits, seen = _lists(chips, 1000 * seed + u, kinds[u % len(kinds)])
        # rows the query does not see: NEG with a stamp, or -inf padding (stamp 0)
        padding = torch.rand(chips, tm.LIST, generator=g) < 0.5
        bits = torch.where(seen, bits, torch.where(padding, tm.NINF_BITS, tm.NEG_BITS))
        stamps = torch.stack([torch.randperm(1 << 15, generator=g)[: tm.LIST] for _ in range(chips)])
        stamps = torch.where(~seen & padding, 0, stamps)
        out[:, u] = tm.rank_order((bits << 16) | stamps)
    return out


def selected_ids(words: torch.Tensor, keep: torch.Tensor) -> list:
    """Per user the sorted (chip, stripe row) pairs of the entries of rank < keep: the selection as ids."""
    chips, users, _ = words.shape
    out = []
    for u in range(users):
        ids = []
        for c in range(chips):
            rows = tm.stamp_row(words[c, u, : int(keep[c, u])] & 0xFFFF)
            ids += [(c, int(r)) for r in rows]
        out.append(sorted(ids))
    return out
