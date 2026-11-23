# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Compare MoE routing between two runs' PREFILL_DUMP_ROUTING_DIR dumps of the same tokens (CPU only).

    python routing_compare.py <dump_dir_a> <dump_dir_b> [--experts-per-chip 8]

Per (layer, chunk start) present in both: expert load imbalance (busiest expert's tokens / mean), the busiest
chip's tokens / mean with experts placed in contiguous blocks of --experts-per-chip (an assumption about the
expert-parallel placement), experts with no token, and how many of each token's top-k experts the two runs share.
The dumps hold the gate's per-token top-k in chip order; both runs use the same chunking and SP layout, so rows
correspond token for token.
"""

import argparse
import json
from pathlib import Path

import torch


def stats(d: dict, experts_per_chip: int) -> dict:
    ids = d["expert_ids"]
    n_exp = int(d["num_routed_experts"])
    real = ids[(ids < n_exp).all(dim=1)]
    counts = torch.bincount(real.flatten().long(), minlength=n_exp).float()
    chips = counts.view(-1, experts_per_chip).sum(dim=1)
    return {
        "tokens": int(real.shape[0]),
        "expert_max_over_mean": round(float(counts.max() / counts.mean()), 3),
        "chip_max_over_mean": round(float(chips.max() / chips.mean()), 3),
        "experts_unused": int((counts == 0).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--experts-per-chip", type=int, default=8)
    args = ap.parse_args()
    a_files = {p.name: p for p in Path(args.a).glob("routing_layer*.pt")}
    b_files = {p.name: p for p in Path(args.b).glob("routing_layer*.pt")}
    rows = []
    for name in sorted(set(a_files) & set(b_files)):
        da, db = torch.load(a_files[name]), torch.load(b_files[name])
        ia, ib = da["expert_ids"].long(), db["expert_ids"].long()
        assert ia.shape == ib.shape, f"{name}: {ia.shape} vs {ib.shape}"
        shared = (ia.unsqueeze(2) == ib.unsqueeze(1)).any(dim=2).sum(dim=1).float()
        rows.append(
            {
                "file": name,
                "a": stats(da, args.experts_per_chip),
                "b": stats(db, args.experts_per_chip),
                "mean_shared_topk": round(float(shared.mean()), 3),
                "topk": int(ia.shape[1]),
            }
        )
    print(json.dumps({"a": args.a, "b": args.b, "compared": rows}, indent=1))


if __name__ == "__main__":
    main()
