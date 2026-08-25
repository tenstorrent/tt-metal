# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rank tt-metal's existing models by architectural similarity to a new one.

Comparison is HF-config vs HF-config only. Nothing about Tenstorrent hardware
enters the similarity metric. The TT consequences are printed in a clearly
separated section, downstream of the verdict, and they only ever say "this
donor's recipe is worth reading" plus arithmetic on the target's own shapes.

Usage:
  python -m models.tooling.arch_donor.compare <target-config.json | hf-repo-id> [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request

from models.tooling.arch_donor import corpus as C
from models.tooling.arch_donor import signature as S

BLOCKS = ("attention", "mlp", "norm", "embed", "global")
SHAPE_KEYS = {
    "attention": ("hidden", "n_q", "n_kv", "head_dim"),
    "mlp": ("ffn", "dense_ffn", "moe_ffn", "n_experts", "top_k"),
    "norm": (),
    "embed": (),
    "global": ("layers", "vocab"),
}


# ------------------------------------------------------------------ comparison
def compare_block(block: str, tgt: S.Signature, don: S.Signature) -> dict:
    sev = S.SEVERITY.get(block, {})
    diffs = []
    for fld, want in tgt.mech[block].items():
        got = don.mech[block].get(fld)
        if got != want:
            diffs.append({"field": fld, "target": want, "donor": got, "severity": sev.get(fld, "dataflow")})
    unknowns = [d for d in diffs if S.UNKNOWN in (str(d["target"]), str(d["donor"]))]
    dataflow = [d for d in diffs if d["severity"] == "dataflow" and d not in unknowns]
    if not diffs:
        verdict = "identical"
    elif len(dataflow) == 1:
        # one mechanism away: the donor is the right skeleton plus one swap
        verdict = "near"
    elif dataflow:
        verdict = "different"
    elif unknowns:
        verdict = "unverified"
    else:
        verdict = "compatible"
    return {"verdict": verdict, "diffs": diffs, "n_dataflow_diffs": len(dataflow)}


def shape_distance(block: str, tgt: S.Signature, don: S.Signature) -> float:
    """Mean absolute log-ratio over the block's dimensions. 0.0 == identical."""
    keys = SHAPE_KEYS[block]
    ratios = []
    for k in keys:
        a, b = tgt.shape.get(block, {}).get(k, 0), don.shape.get(block, {}).get(k, 0)
        if a and b:
            ratios.append(abs(math.log(a / b)))
        elif a or b:
            ratios.append(math.log(4))  # one side absent: large but finite
    return sum(ratios) / len(ratios) if ratios else 0.0


VERDICT_RANK = {"identical": 0, "compatible": 1, "near": 2, "unverified": 3, "different": 4}


def rank_for_block(block: str, tgt: S.Signature, donors: list[S.Signature], galaxy_only: bool) -> list[dict]:
    rows = []
    for d in donors:
        if galaxy_only and not d.galaxy_class:
            continue
        cmp = compare_block(block, tgt, d)
        rows.append(
            {
                "donor": d,
                "block": block,
                **cmp,
                "shape_dist": shape_distance(block, tgt, d),
            }
        )
    rows.sort(
        key=lambda r: (VERDICT_RANK[r["verdict"]], r["n_dataflow_diffs"], C.TIER_RANK[r["donor"].tier], r["shape_dist"])
    )
    return rows


# ----------------------------------------------------------------- consequence
# Downstream of similarity, and only ever arithmetic on the TARGET's own shapes.
def divisibility(tgt: S.Signature, factors=(2, 4, 8, 16, 32)) -> dict[int, list[str]]:
    a, m = tgt.shape["attention"], tgt.shape["mlp"]
    out = {}
    for f in factors:
        problems = []
        if a["n_q"] % f:
            problems.append(f"n_q={a['n_q']}")
        if a["n_kv"] % f:
            problems.append(f"n_kv={a['n_kv']}")
        if a["hidden"] % f or (a["hidden"] // f) % 32:
            problems.append(f"hidden={a['hidden']}")
        ffn = m["moe_ffn"] or m["ffn"]
        if ffn % f or (ffn // f) % 32:
            problems.append(f"ffn={ffn}")
        out[f] = problems
    return out


def load_target(spec: str) -> S.Signature:
    if os.path.exists(spec):
        base = os.path.basename(spec).replace(".json", "")
        # a bare "config.json" is named by its directory, as the corpus does
        name = os.path.basename(os.path.dirname(os.path.abspath(spec))) if base == "config" else base
        return S.from_path(spec, name=name)
    url = f"https://huggingface.co/{spec}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as r:
        cfg = json.load(r)
    return S.build(cfg, name=spec.split("/")[-1], source=url)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="path to a config.json, or a HuggingFace repo id")
    ap.add_argument("--all-sizes", action="store_true", help="include sub-Galaxy donors")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tgt = load_target(args.target)
    tgt_path = os.path.realpath(tgt.source) if os.path.exists(tgt.source) else None
    donors = [
        d
        for d in C.build_corpus()
        if d.name != tgt.name and (tgt_path is None or os.path.realpath(os.path.join(C.REPO, d.source)) != tgt_path)
    ]
    galaxy_only = not args.all_sizes

    per_block = {b: rank_for_block(b, tgt, donors, galaxy_only) for b in BLOCKS}

    if args.json:
        print(
            json.dumps(
                {
                    "target": tgt.to_dict(),
                    "blocks": {
                        b: [
                            {
                                "donor": r["donor"].name,
                                "tier": r["donor"].tier,
                                "verdict": r["verdict"],
                                "shape_dist": round(r["shape_dist"], 3),
                                "diffs": r["diffs"],
                            }
                            for r in rows[: args.top]
                        ]
                        for b, rows in per_block.items()
                    },
                },
                indent=2,
                default=str,
            )
        )
        return 0

    p = tgt.params
    print(f"\nTARGET  {tgt.name}   ({tgt.model_type}, {tgt.architectures})")
    print(
        f"  {p.get('total_B')}B total / {p.get('active_no_embed_B')}B active"
        f"   {p.get('gflop_per_token')} GFLOP/token   KV {p.get('kv_bytes_per_token_bf16', 0)/1024:.0f} KiB/token bf16"
    )
    for b in ("attention", "mlp"):
        print(f"  {b:10s} {tgt.mech[b]}")
        print(f"  {'':10s} {tgt.shape[b]}")
    print(
        f"  {'norm':10s} {tgt.mech['norm']}    quant={tgt.mech['global']['quant']}"
        f"   layers={tgt.shape['global']['layers']}  vision={tgt.has_vision}"
    )
    for n in tgt.notes:
        print(f"  ! {n}")

    print(f"\nDONOR RANKING  (corpus={len(donors)}, {'Galaxy-class only' if galaxy_only else 'all sizes'})")
    for b in BLOCKS:
        rows = per_block[b]
        if not rows:
            continue
        print(f"\n  [{b}]")
        for r in rows[: args.top]:
            d = r["donor"]
            delta = (
                ""
                if not r["diffs"]
                else "  ".join(f"{x['field']}:{x['donor']}→{x['target']}({x['severity'][:4]})" for x in r["diffs"][:4])
            )
            print(f"    {r['verdict']:11s} {d.name:26s} {d.tier:10s} shapeΔ={r['shape_dist']:.2f}  {delta}")

    print("\nSUMMARY")
    best_overall: dict[str, int] = {}
    for b in BLOCKS:
        for r in per_block[b]:
            if r["verdict"] in ("identical", "compatible"):  # "near" needs work, so it does not count
                best_overall[r["donor"].name] = best_overall.get(r["donor"].name, 0) + 1
    if best_overall:
        top = sorted(best_overall.items(), key=lambda kv: -kv[1])
        print("  reusable-block count by donor: " + ", ".join(f"{n} ({c}/{len(BLOCKS)})" for n, c in top[:4]))
        winner = next(d for d in donors if d.name == top[0][0])
        print(f"  -> closest overall: {winner.name} [{winner.tier}]  recipe lives in {winner.impl_dir}/")
    gaps = [b for b in BLOCKS if per_block[b] and per_block[b][0]["verdict"] in ("near", "different")]
    if gaps:
        print(f"  -> no exact galaxy donor for: {', '.join(gaps)}  (adapt the nearest, or new work)")
        for b in gaps:
            for x in per_block[b][0]["diffs"]:
                if x["severity"] == "dataflow":
                    print(f"       {b}.{x['field']}: target={x['target']!r} vs nearest donor={x['donor']!r}")

    print("\nCONSEQUENCE  (arithmetic on the target's own shapes; not a similarity input)")
    for f, probs in divisibility(tgt).items():
        print(f"    split by {f:2d}: {'OK  (tile-aligned)' if not probs else 'blocked by ' + ', '.join(probs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
