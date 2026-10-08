# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer routed-expert placement for the M3 EP MoE (kagent/m3-prefill-moe). Pure torch; no ttnn import.

Why: on the 4x4 prefill mesh each chip hosts 8 of the 128 experts (column g = dispatch group serves labels
[32 g, 32 g + 32), chip row r of that column serves 32 g + 8 r + [0, 8), the ``ExpertMapping`` column-major
default). The chip whose 8 experts drew the most (token, expert) rows sets the critical path of dispatch ->
experts -> combine -> TP reduce-scatter for the whole mesh.

How (exact): a placement is a per-layer permutation ``perm`` of the 128 labels with ``perm[new_label] = expert``:
the expert whose checkpoint id is ``perm[n]`` is served at the slot of label ``n``. It is applied at load time by
permuting the router's output columns (gate weight columns and the e_score_correction_bias, bf16 bytes moved,
never rounded) and the cached bf4 expert weight shards (bytes moved between devices / local slots). The router
then emits new labels for the same expert choices (the logits of each expert are computed from the same weight
column, the top-k is over the same values; only exact fp32 score ties could resolve differently, because the
top-k tie order is by index), and everything downstream (masked_bincount, offset_cumsum, dispatch, the expert
FFN, combine, post_combine_reduce) runs the default label -> chip mapping unchanged.

Exactness classes:
  * ``column`` placements keep every expert in its original column (``perm[n] // 32 == n // 32``). Each column then
    holds the same expert set, post_combine_reduce sums the same top-k slots in the same order, and the TP
    reduce-scatter adds identical partials: KV and logits are expected bit-exact (class A; verify).
  * ``global`` placements may move an expert to another column; the per-column partial sums group differently
    before the bf16 reduce-scatter (class B, NLL gate).

File format (torch.save): {"perm": {global_layer_idx: LongTensor[128]}, "mode": str, "meta": dict}.

CLI:
  python -m models.demos.minimax_m3.utils.expert_placement build --routing <routing.pt> --calib 0:51200 \
      --mode column --out placement.pt            # placement from calibration rows
  python -m models.demos.minimax_m3.utils.expert_placement eval --routing <routing.pt> --rows 51200:56320 \
      [--placement placement.pt]                   # routed rows per chip, max/mean, per layer
"""

from __future__ import annotations

import argparse
import statistics

import torch

NUM_EXPERTS = 128
EXPERTS_PER_CHIP = 8
DISPATCH_GROUP_SIZE = 4  # mesh rows (SP)
NUM_DISPATCH_GROUPS = 4  # mesh cols (TP)
EXPERTS_PER_GROUP = EXPERTS_PER_CHIP * DISPATCH_GROUP_SIZE


def label_position(n: int):
    """(group/column g, chip row r, local slot l) that serves label n under the default ExpertMapping."""
    return n // EXPERTS_PER_GROUP, (n % EXPERTS_PER_GROUP) // EXPERTS_PER_CHIP, n % EXPERTS_PER_CHIP


def identity() -> torch.Tensor:
    return torch.arange(NUM_EXPERTS, dtype=torch.int64)


def validate(perm: torch.Tensor, column_preserving: bool | None = None) -> bool:
    """Assert ``perm`` is a permutation of the labels; returns whether it keeps every expert in its column."""
    perm = torch.as_tensor(perm, dtype=torch.int64)
    assert perm.shape == (NUM_EXPERTS,), perm.shape
    assert torch.equal(torch.sort(perm).values, identity()), "placement is not a permutation of the 128 experts"
    col_ok = bool(torch.all(perm // EXPERTS_PER_GROUP == identity() // EXPERTS_PER_GROUP))
    if column_preserving:
        assert col_ok, "placement moves experts across columns but was declared column-preserving"
    return col_ok


def inverse(perm: torch.Tensor) -> torch.Tensor:
    """label_of[expert] for perm[label] = expert."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), dtype=perm.dtype)
    return inv


def expert_counts(indices: torch.Tensor) -> torch.Tensor:
    """(token, expert) rows per expert from a [tokens, topk] id tensor (negative / >= 128 ignored)."""
    idx = indices.to(torch.int64).flatten()
    idx = idx[(idx >= 0) & (idx < NUM_EXPERTS)]
    return torch.bincount(idx, minlength=NUM_EXPERTS).to(torch.float64)


def chip_loads(counts: torch.Tensor, perm: torch.Tensor | None = None) -> torch.Tensor:
    """Routed rows per chip [g, r] (column, row) when expert perm[n] is served at label n's slot."""
    c = counts if perm is None else counts[perm]
    return c.view(NUM_DISPATCH_GROUPS, DISPATCH_GROUP_SIZE, EXPERTS_PER_CHIP).sum(-1)


def _partition(loads: list[float], n_bins: int, bin_size: int, iters: int = 2000) -> list[list[int]]:
    """Split len(loads) == n_bins * bin_size items into n_bins bins of exactly bin_size items minimizing the
    largest bin sum: cardinality-constrained LPT, then pairwise-swap descent on the heaviest bin."""
    order = sorted(range(len(loads)), key=lambda i: -loads[i])
    bins = [[] for _ in range(n_bins)]
    sums = [0.0] * n_bins
    for i in order:
        cand = [b for b in range(n_bins) if len(bins[b]) < bin_size]
        b = min(cand, key=lambda b: sums[b])
        bins[b].append(i)
        sums[b] += loads[i]
    for _ in range(iters):
        hi = max(range(n_bins), key=lambda b: sums[b])
        best = None  # (new max of the pair, lo, a, c)
        for lo in range(n_bins):
            if lo == hi:
                continue
            for a in bins[hi]:
                for c in bins[lo]:
                    d = loads[a] - loads[c]
                    if d <= 0:
                        continue
                    new_pair_max = max(sums[hi] - d, sums[lo] + d)
                    if new_pair_max < sums[hi] - 1e-9 and (best is None or new_pair_max < best[0]):
                        best = (new_pair_max, lo, a, c)
        if best is None:
            break
        _, lo, a, c = best
        bins[hi].remove(a)
        bins[hi].append(c)
        bins[lo].remove(c)
        bins[lo].append(a)
        d = loads[a] - loads[c]
        sums[hi] -= d
        sums[lo] += d
    return bins


def solve(counts: torch.Tensor, mode: str = "column", active_cost: float = 0.0) -> torch.Tensor:
    """Placement for one layer from calibration ``counts`` [128] (routed rows per expert).

    mode "column": balance the 4 chips of each column using only that column's 32 experts (bit-exact class).
    mode "global": balance all 16 chips over all 128 experts (class B), then order the 16 groups so that the
    column totals are balanced as well. ``active_cost`` adds a per-expert constant (in rows) to model the
    weight streaming every active expert pays regardless of its token count.
    """
    loads = [float(x) + (active_cost if x > 0 else 0.0) for x in counts.tolist()]
    perm = torch.empty(NUM_EXPERTS, dtype=torch.int64)
    if mode == "column":
        for g in range(NUM_DISPATCH_GROUPS):
            ids = list(range(g * EXPERTS_PER_GROUP, (g + 1) * EXPERTS_PER_GROUP))
            bins = _partition([loads[i] for i in ids], DISPATCH_GROUP_SIZE, EXPERTS_PER_CHIP)
            for r, b in enumerate(bins):
                for l, j in enumerate(sorted(ids[k] for k in b)):
                    perm[g * EXPERTS_PER_GROUP + r * EXPERTS_PER_CHIP + l] = j
    elif mode == "global":
        bins = _partition(loads, NUM_DISPATCH_GROUPS * DISPATCH_GROUP_SIZE, EXPERTS_PER_CHIP)
        gsum = [sum(loads[i] for i in b) for b in bins]
        cols = _partition(gsum, NUM_DISPATCH_GROUPS, DISPATCH_GROUP_SIZE)  # balance the column totals too
        for g, col in enumerate(cols):
            for r, bi in enumerate(col):
                for l, j in enumerate(sorted(bins[bi])):
                    perm[g * EXPERTS_PER_GROUP + r * EXPERTS_PER_CHIP + l] = j
    else:
        raise ValueError(mode)
    validate(perm, column_preserving=(mode == "column"))
    return perm


# ---- file I/O ------------------------------------------------------------------------------------------------
_CACHE = {}


def load(path: str) -> dict:
    if path not in _CACHE:
        d = torch.load(path, map_location="cpu", weights_only=False)
        perms = {int(k): torch.as_tensor(v, dtype=torch.int64) for k, v in d["perm"].items()}
        for p in perms.values():
            validate(p)
        _CACHE[path] = {"perm": perms, "mode": d.get("mode"), "meta": d.get("meta", {})}
    return _CACHE[path]


def perm_for_layer(path: str | None, layer_idx: int | None) -> torch.Tensor | None:
    """The layer's placement, or None (default placement) when there is no file, no entry or an identity."""
    if not path or layer_idx is None:
        return None
    d = load(path)
    p = d["perm"].get(int(layer_idx))
    if p is None:
        return None
    if torch.equal(p, identity()) and not d["meta"].get("force_shuffle"):
        return None  # meta force_shuffle: run identity layers through the load-time shuffle (machinery test)
    return p


# ---- CLI -----------------------------------------------------------------------------------------------------
def _span(s: str) -> slice:
    a, b = s.split(":")
    return slice(int(a), int(b))


def _layer_stats(idx: torch.Tensor, rows: slice, perm=None):
    cnt = expert_counts(idx[rows])
    ch = chip_loads(cnt, perm)
    col = ch.sum(1)
    return ch.max().item() / ch.mean().item(), col.max().item() / col.mean().item(), ch


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--routing", required=True)
    b.add_argument("--calib", required=True, help="token span a:b of the routing capture used as statistics")
    b.add_argument("--mode", default="column", choices=("column", "global"))
    b.add_argument("--active-cost", type=float, default=0.0)
    b.add_argument("--out", required=True)
    i = sub.add_parser("identity", help="identity placement for every MoE layer with force_shuffle (must be bit-exact)")
    i.add_argument("--layers", default="3-59")
    i.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--routing", required=True)
    e.add_argument("--rows", required=True)
    e.add_argument("--placement")
    e.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.cmd == "identity":
        lo, hi = (int(x) for x in a.layers.split("-"))
        perms = {L: identity() for L in range(lo, hi + 1)}
        torch.save({"perm": perms, "mode": "identity", "meta": {"force_shuffle": True}}, a.out)
        print(f"wrote identity placement (force_shuffle) for layers {lo}-{hi} -> {a.out}")
        return
    cap = torch.load(a.routing, map_location="cpu", weights_only=False)
    layers = cap["layers"]
    if a.cmd == "build":
        perms = {}
        for L in sorted(layers):
            perms[L] = solve(expert_counts(layers[L][_span(a.calib)]), a.mode, a.active_cost)
        torch.save(
            {"perm": perms, "mode": a.mode, "meta": {"routing": a.routing, "calib": a.calib, "ac": a.active_cost}},
            a.out,
        )
        print(f"wrote {len(perms)} layer placements ({a.mode}) -> {a.out}")
        return
    plc = load(a.placement)["perm"] if a.placement else {}
    base, new, cbase, cnew = [], [], [], []
    for L in sorted(layers):
        r0, c0, ch0 = _layer_stats(layers[L], _span(a.rows))
        r1, c1, ch1 = _layer_stats(layers[L], _span(a.rows), plc.get(L))
        base.append(r0)
        new.append(r1)
        cbase.append(c0)
        cnew.append(c1)
        if a.verbose:
            print(
                f"L{L:02d} chip max/mean {r0:.3f} -> {r1:.3f}  col max/mean {c0:.3f} -> {c1:.3f}  max rows "
                f"{ch0.max().item():.0f} -> {ch1.max().item():.0f} (mean {ch0.mean().item():.0f})"
            )
    q = (
        lambda xs: f"median {statistics.median(xs):.3f} mean {statistics.mean(xs):.3f} [min {min(xs):.3f}, max {max(xs):.3f}]"
    )  # noqa: E731
    print(f"per-chip routed rows max/mean: default {q(base)}")
    if plc:
        print(f"per-chip routed rows max/mean: placed  {q(new)}")
    print(f"per-column max/mean: default {q(cbase)}" + (f"; placed {q(cnew)}" if plc else ""))


if __name__ == "__main__":
    main()
