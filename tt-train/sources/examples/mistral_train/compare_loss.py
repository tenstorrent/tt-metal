# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare a ttml loss log with torch reference logs and assert the stated tolerance.

    python compare_loss.py --ttml loss.json --torch loss_torch.json [--torch-bf16 loss_torch_bf16.json] --tol 0.10

The ttml value compared is ``host_loss`` (fp32 on the host from the device logits) when the log has it, else the
bf16 device loss. Tolerance derivation: ``doc/composite/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ttml", required=True)
    p.add_argument("--torch", required=True, help="fp64 reference, unrounded parameters")
    p.add_argument("--torch-bf16", default=None, help="fp64 reference with bf16-rounded parameters and moments")
    p.add_argument("--tol", type=float, required=True, help="max |ttml - reference| allowed, per step")
    p.add_argument("--ttml2", default=None, help="a repeated ttml run (determinism)")
    args = p.parse_args()
    a = json.load(open(args.ttml))["steps"]
    t = json.load(open(args.torch))["steps"]
    tb = json.load(open(args.torch_bf16))["steps"] if args.torch_bf16 else None
    key = "host_loss" if "host_loss" in a[0] else "loss"
    ref = tb if tb else t
    ref_name = "torch fp64, bf16-rounded params" if tb else "torch fp64"
    print(f"ttml value: {key}; reference for the tolerance: {ref_name}; tol {args.tol}")
    head = (
        "step | ttml bf16 loss | ttml host fp32 | torch fp64"
        + (" | torch fp64 bf16-params" if tb else "")
        + " | ttml - ref"
    )
    print(head)
    diffs = []
    for i, (x, y) in enumerate(zip(a, ref)):
        d = x[key] - y["loss"]
        diffs.append(d)
        row = f"{x['step']:4d} | {x['loss']:.4f} | {x.get('host_loss', float('nan')):.4f} | {t[i]['loss']:.4f}"
        if tb:
            row += f" | {tb[i]['loss']:.4f}"
        print(row + f" | {d:+.4f}")
    mx = max(abs(d) for d in diffs)
    mean = sum(diffs) / len(diffs)
    print(
        f"max |diff| {mx:.4f} (tol {args.tol}), mean {mean:+.4f}, signs +{sum(d > 0 for d in diffs)}/-{sum(d < 0 for d in diffs)}"
    )
    if tb:
        d2 = [x["loss"] - y["loss"] for x, y in zip(tb, t)]
        print(
            f"bf16-params replay vs unrounded torch: max |diff| {max(abs(d) for d in d2):.4f}, mean {sum(d2)/len(d2):+.4f}"
        )
    if args.ttml2:
        b = json.load(open(args.ttml2))["steps"]
        same = all(x["loss"] == y["loss"] for x, y in zip(a, b))
        print(f"repeated ttml run identical: {same}")
        if not same:
            sys.exit("determinism failed")
    if mx > args.tol:
        sys.exit(f"FAIL: max |diff| {mx:.4f} > tol {args.tol}")
    print("PASS")


if __name__ == "__main__":
    main()
