# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Diff two OLMO_LAYER_PROBE dumps: first (decode step, layer, stage) whose output differs, with devices and rows.

    python models/demos/olmo_galaxy/scripts/compare_layer_probes.py A.pt B.pt [--layers 64] [--show 12]

Records are in execution order: 6 stages per layer per decode step.
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()
    a, b = torch.load(args.a, weights_only=False), torch.load(args.b, weights_only=False)
    per_step = args.layers * 6
    print(f"{len(a)} vs {len(b)} records ({len(a) / per_step:.2f} decode steps)")
    shown = 0
    for i, (x, y) in enumerate(zip(a, b)):
        bad = [d for d, (p, q) in enumerate(zip(x["hashes"], y["hashes"])) if p != q]
        if not bad:
            continue
        rows = sorted({r for d in bad for r in (x["rows"][d] != y["rows"][d]).nonzero().flatten().tolist()})
        step = i // per_step
        print(
            f"step {step} layer {x['layer']} {x['stage']}: {len(bad)}/{len(x['hashes'])} devices differ {bad[:8]} rows {rows[:12]}"
        )
        shown += 1
        if shown >= args.show:
            break
    if shown == 0:
        print("all recorded stages identical")


if __name__ == "__main__":
    main()
