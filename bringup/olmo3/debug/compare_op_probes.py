# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Find the first op whose output differs between two OLMO_OP_PROBE dumps (same prompts, eager decode step).

    python models/demos/olmo_galaxy/scripts/compare_op_probes.py A.pt B.pt [--show 15]

Each dump is a list of {op, shape, hashes (md5 per device), rows (per-row sums per device)}. Ops are matched by
position, so both runs must come from the same commit and config. Prints the first differing ops with the
devices and rows that differ.
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    a, b = torch.load(args.a, weights_only=False), torch.load(args.b, weights_only=False)
    print(f"{len(a)} vs {len(b)} op outputs")
    shown = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra["op"] != rb["op"] or ra["shape"] != rb["shape"]:
            print(f"op #{i}: sequence mismatch {ra['op']}{ra['shape']} vs {rb['op']}{rb['shape']}")
            break
        bad = [d for d, (x, y) in enumerate(zip(ra["hashes"], rb["hashes"])) if x != y]
        if not bad:
            continue
        rows = sorted({r for d in bad for r in (ra["rows"][d] != rb["rows"][d]).nonzero().flatten().tolist()})
        print(
            f"op #{i} {ra['op']} shape={ra['shape']}: {len(bad)}/{len(ra['hashes'])} devices differ {bad[:8]}, rows {rows[:12]}"
        )
        shown += 1
        if shown >= args.show:
            break
    if shown == 0:
        print("all op outputs identical")


if __name__ == "__main__":
    main()
