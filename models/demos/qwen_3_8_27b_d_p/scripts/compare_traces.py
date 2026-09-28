#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC two golden traces against each other, tensor by tensor.

Built for comparing the two generator backends -- ``--backend reference`` (the vendored fp16
reference) against ``--backend hf`` (upstream transformers) -- on the same prompt and ISL. Both
write the same filenames, keys and shapes, so a mismatch in any of those is reported as a failure
rather than skipped.

Storage dtype does not have to match: bf16-vs-fp16 storage costs ~1.6e-6 PCC, far below the
backend difference, so a bf16 HF trace is directly comparable to an fp16 reference trace.

    python3 models/demos/qwen_3_8_27b_d_p/scripts/compare_traces.py \\
        $QWEN35_GOLDEN_ROOT/longbook_10240 $QWEN35_GOLDEN_ROOT/longbook_10240_hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    # float64: an fp32 dot product over ~1e7 elements accumulates enough error to report PCC > 1.
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="baseline trace dir")
    ap.add_argument("b", type=Path, help="trace dir to compare against it")
    ap.add_argument("--bound", type=float, default=0.99, help="fail below this PCC (default 0.99)")
    ap.add_argument("--top", type=int, default=12, help="how many worst rows to print")
    args = ap.parse_args()

    ma, mb = (json.loads((p / "metadata.json").read_text()) for p in (args.a, args.b))
    if ma["token_ids"] != mb["token_ids"]:
        print("ERROR: the two traces were generated from different tokens", file=sys.stderr)
        return 2
    for field in ("num_layers", "n_tokens"):
        assert ma[field] == mb[field], f"{field}: {ma[field]} != {mb[field]}"
    print(f"A  {args.a}\n   backend={ma.get('backend', 'reference')} dtype={ma['dtype']}")
    print(f"B  {args.b}\n   backend={mb.get('backend', 'reference')} dtype={mb['dtype']}")
    print(f"   {ma['n_tokens']} tokens, {ma['num_layers']} layers\n")

    rows: list[tuple[str, float]] = [
        (
            "final_hidden",
            pcc(
                load_file(str(args.a / "final_hidden.safetensors"))["final_hidden"],
                load_file(str(args.b / "final_hidden.safetensors"))["final_hidden"],
            ),
        )
    ]
    for idx in range(ma["num_layers"]):
        ta, tb = (load_file(str(p / "kv_cache" / f"layer_{idx}.safetensors")) for p in (args.a, args.b))
        assert ta.keys() == tb.keys(), f"layer {idx}: key sets differ, {sorted(ta)} vs {sorted(tb)}"
        for k in ta:
            assert ta[k].shape == tb[k].shape, f"layer {idx} {k}: {tuple(ta[k].shape)} != {tuple(tb[k].shape)}"
            rows.append((k, pcc(ta[k], tb[k])))

    ordered = sorted(rows, key=lambda r: r[1])
    worst = ordered[0][1]
    print(f"{'worst tensors':32s} {'PCC':>12s}")
    print("-" * 46)
    for name, value in ordered[: args.top]:
        print(f"{name:32s} {value:12.8f}")
    print("-" * 46)
    e2e = dict(rows)["final_hidden"]
    below = [n for n, v in rows if v < args.bound]
    print(f"{'e2e (final_hidden)':32s} {e2e:12.8f}")
    print(f"{'worst of ' + str(len(rows)):32s} {worst:12.8f}")
    print(f"\n{len(rows) - len(below)} / {len(rows)} at or above {args.bound}")
    if below:
        print(f"below bound: {', '.join(below[:10])}{' …' if len(below) > 10 else ''}")
    return 0 if not below else 1


if __name__ == "__main__":
    sys.exit(main())
