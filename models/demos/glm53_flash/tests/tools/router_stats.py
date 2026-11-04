# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-EP-row load statistics from a router dump (``tests/test_router_stats_mesh.py``):

    python tests/tools/router_stats.py <stats.pt> [--users 8] [--steps 2000] [--redundant 4]

For every MoE layer: the expert popularity (share of selections), and the per-step load of the busiest EP row
(the number of the step's selections whose expert lives on that row: the row's chips each run that many experts)
under three placements of the E experts over the ``rows`` EP rows (E / rows per row):
  contiguous  -- today's ``[r E/rows, (r+1) E/rows)``;
  balanced    -- greedy by popularity (the most popular expert to the lightest row), static;
  redundant   -- balanced plus the ``--redundant`` most popular experts of every row also placed on a second row,
                 each selection of a duplicated expert going to the lighter of its two rows at that step (dynamic).
A step draws ``--users`` tokens at random from the dump (one token per user, k experts each): the decode batch of a
``users``-user server. Prints the mean busiest-row load and the mean over rows per layer, and the totals.
"""

from __future__ import annotations

import argparse

import torch


def placement_contiguous(E: int, rows: int) -> torch.Tensor:
    return torch.arange(E) // (E // rows)


def placement_balanced(pop: torch.Tensor, rows: int) -> torch.Tensor:
    """Greedy: experts by decreasing popularity, each to the row with the smallest popularity sum that still has
    room (E / rows experts per row)."""
    E = pop.numel()
    cap = E // rows
    row_of = torch.full((E,), -1, dtype=torch.long)
    load = torch.zeros(rows)
    count = torch.zeros(rows, dtype=torch.long)
    for e in torch.argsort(pop, descending=True).tolist():
        order = torch.argsort(load).tolist()
        r = next(r for r in order if count[r] < cap)
        row_of[e] = r
        load[r] += pop[e]
        count[r] += 1
    return row_of


def step_loads(ids: torch.Tensor, row_of: torch.Tensor, rows: int, second: torch.Tensor | None = None) -> torch.Tensor:
    """``ids`` [steps, users, k] -> [steps, rows] loads = the DISTINCT experts of the step on each row (the weights
    stream once per expert and the 32-row tile covers every user, so the expert phase scales with distinct experts).
    With ``second`` (the duplicate row per expert, -1 = none) every distinct duplicated expert goes to the lighter
    of its rows, in decreasing-popularity order of the step's experts (greedy, dynamic)."""
    S = ids.shape[0]
    flat = ids.reshape(S, -1)
    out = torch.zeros(S, rows)
    for s in range(S):
        experts = torch.unique(flat[s])
        if second is None:
            out[s] = torch.bincount(row_of[experts], minlength=rows).float()
            continue
        load = torch.zeros(rows)
        for e in experts.tolist():
            r1, r2 = int(row_of[e]), int(second[e])
            r = r1 if r2 < 0 or load[r1] <= load[r2] else r2
            load[r] += 1
        out[s] = load
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--redundant", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = torch.load(args.stats)
    E, k, rows = int(d["E"]), int(d["k"]), int(d["rows"])
    layers = d["layers"]
    g = torch.Generator().manual_seed(args.seed)
    print(f"{len(layers)} MoE layers, E {E}, k {k}, EP rows {rows}; {args.users} users per step, {args.steps} steps")
    print(
        f"{'layer':>5} {'tokens':>7} {'top1 %':>7} {'top8 %':>7} {'contig max':>11} {'balanced max':>13} {'redundant max':>14} {'mean':>6}"
    )
    tot = torch.zeros(4)
    for layer in sorted(layers):
        ids = layers[layer].to(torch.long)  # [tokens, k]
        n = ids.shape[0]
        pop = torch.bincount(ids.reshape(-1), minlength=E).float()
        share = pop / pop.sum()
        top = torch.sort(share, descending=True).values
        pick = torch.randint(0, n, (args.steps, args.users), generator=g)
        step_ids = ids[pick]  # [steps, users, k]
        contig = placement_contiguous(E, rows)
        bal = placement_balanced(pop, rows)
        # redundancy: the most popular experts of every balanced row get a second row (the lightest other row)
        second = torch.full((E,), -1, dtype=torch.long)
        if args.redundant > 0:
            row_pop = torch.zeros(rows)
            for e in range(E):
                row_pop[bal[e]] += pop[e]
            for r in range(rows):
                members = [e for e in range(E) if bal[e] == r]
                hot = sorted(members, key=lambda e: -pop[e])[: args.redundant]
                for e in hot:
                    other = sorted((rr for rr in range(rows) if rr != r), key=lambda rr: row_pop[rr])[0]
                    second[e] = other
        l_c = step_loads(step_ids, contig, rows)
        l_b = step_loads(step_ids, bal, rows)
        l_r = step_loads(step_ids, bal, rows, second)
        mean = l_c.sum(1).mean().item() / rows
        row = torch.tensor([l_c.max(1).values.mean(), l_b.max(1).values.mean(), l_r.max(1).values.mean(), mean])
        tot += row
        print(
            f"{layer:5d} {n:7d} {top[0] * 100:7.2f} {top[:8].sum() * 100:7.1f} {row[0]:11.2f} {row[1]:13.2f} {row[2]:14.2f} {row[3]:6.2f}"
        )
    tot /= len(layers)
    print(
        f"\nmean busiest-row load per MoE layer: contiguous {tot[0]:.2f}, balanced {tot[1]:.2f}, redundant {tot[2]:.2f} vs the mean row {tot[3]:.2f}"
    )


if __name__ == "__main__":
    main()
