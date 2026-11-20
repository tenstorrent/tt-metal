# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""X.9 (owner 2026-11-19): the hot-expert census over texts, from test_expert_load_census.py's side records. Host only.

Reads <art>/census/census_p<p>_k<k>_t<text>.json (one per text) and prints:
  1. a table per text: routed_experts_combine ms a chunk (measured, DS41-299's fit, the balanced fit), the busiest chip's
     tiles vs balanced, the hot-8 overlap between consecutive chunks;
  2. the hot-8 overlap between texts (each text's counts summed over its chunks), mean over layers;
  3. a static replica set chosen on the picking texts (per layer the R experts with the largest share of rows, the
     picking texts' shares summed) and how it covers the held-out texts: the share of each held-out chunk's hot-8
     experts inside the set and the share of the held-out chunk's routed rows the set's experts carry.

    PYTHONPATH=$PWD python -m models.demos.deepseek_v41_flash_d_p.tests.perf.census_texts --spec $BRINGUP_SPEC \\
        [--pos 145408] [--chunks 4] [--pick prose,code,math] [--held agentic,multilingual] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HOT = 8


def hot_set(rows: dict, n: int = HOT) -> list[int]:
    return [e for e, _ in sorted(rows.items(), key=lambda kv: (-kv[1], int(kv[0])))[:n]]


def overlap(a, b) -> float:
    return len(set(a) & set(b)) / float(max(len(a), 1))


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def chunk_rows(rec: dict) -> list[dict]:
    """[chunk] -> {layer: {expert: rows}}."""
    return [
        {int(L["layer"]): {int(e): int(v) for e, v in L["expert_rows"].items()} for L in ch["layers"]}
        for ch in rec["chunks"]
    ]


def summed(rec: dict) -> dict:
    out: dict = {}
    for ch in chunk_rows(rec):
        for li, d in ch.items():
            o = out.setdefault(li, {})
            for e, v in d.items():
                o[e] = o.get(e, 0) + v
    return out


def analyse(recs: dict, pick: list[str], held: list[str], sizes=(8, 16, 32)) -> dict:
    res = {"texts": {}, "cross": {}, "static": {}}
    for t, rec in recs.items():
        chs = rec["chunks"]
        crows = chunk_rows(rec)
        lay = sorted(crows[0])
        ov = [
            mean(overlap(hot_set(crows[j][li]), hot_set(crows[j + 1][li])) for j in range(len(crows) - 1)) for li in lay
        ]
        res["texts"][t] = {
            "ms": mean(sum(L["program_us_max"] for L in c["layers"]) / 1e3 for c in chs),
            "fit_ms": mean(sum(L["fit_us"] for L in c["layers"]) / 1e3 for c in chs),
            "fit_bal_ms": mean(sum(L["fit_balanced_us"] for L in c["layers"]) / 1e3 for c in chs),
            "tiles_max": mean(sum(L["tiles_max"] for L in c["layers"]) for c in chs),
            "tiles_bal": mean(sum(L["tiles_balanced"] for L in c["layers"]) for c in chs),
            "hot8_chunks": mean(ov) if len(crows) > 1 else float("nan"),
            "hot8_chunks_min_layer": min(ov) if len(crows) > 1 else float("nan"),
            "hot_share": mean(
                sum(sorted(d[li].values(), reverse=True)[:HOT]) / max(sum(d[li].values()), 1)
                for d in crows
                for li in lay
            ),
        }
    sums = {t: summed(r) for t, r in recs.items()}
    names = sorted(recs)
    for a in names:
        for b in names:
            if a < b:
                common = sorted(set(sums[a]) & set(sums[b]))
                res["cross"][f"{a}/{b}"] = mean(overlap(hot_set(sums[a][li]), hot_set(sums[b][li])) for li in common)
    pk = [t for t in pick if t in sums]
    if pk:
        layers = sorted(set.intersection(*[set(sums[t]) for t in pk]))
        for R in sizes:
            chosen = {}
            for li in layers:
                share: dict = {}
                for t in pk:
                    tot = max(sum(sums[t][li].values()), 1)
                    for e, v in sums[t][li].items():
                        share[e] = share.get(e, 0.0) + v / tot
                chosen[li] = set(hot_set(share, R))
            cov = {}
            for t in list(held) + pk:
                if t not in recs:
                    continue
                hot_in, rows_in, top1_in = [], [], []
                for d in chunk_rows(recs[t]):
                    for li in layers:
                        if li not in d:
                            continue
                        h = hot_set(d[li])
                        hot_in.append(len(set(h) & chosen[li]) / len(h))
                        top1_in.append(1.0 if h and h[0] in chosen[li] else 0.0)
                        rows_in.append(
                            sum(v for e, v in d[li].items() if e in chosen[li]) / max(sum(d[li].values()), 1)
                        )
                cov[t] = {"hot8_in": mean(hot_in), "top1_in": mean(top1_in), "rows_in": mean(rows_in)}
            res["static"][R] = cov
    return res


def report(res: dict, pick, held) -> str:
    lines = [
        "| text | role | experts ms (measured) | fit ms | balanced fit ms | tiles max sum | tiles balanced sum | "
        "hot-8 overlap chunk to chunk (min layer) | hot-8 share of rows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t, r in sorted(res["texts"].items()):
        role = "pick" if t in pick else ("held-out" if t in held else "-")
        lines.append(
            f"| {t} | {role} | {r['ms']:.2f} | {r['fit_ms']:.2f} | {r['fit_bal_ms']:.2f} | {r['tiles_max']:.0f} | "
            f"{r['tiles_bal']:.0f} | {r['hot8_chunks']:.3f} ({r['hot8_chunks_min_layer']:.3f}) | {r['hot_share']:.3f} |"
        )
    lines += ["", "| texts | hot-8 overlap (summed over chunks, mean over layers) |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in sorted(res["cross"].items())]
    lines += [
        "",
        "| static set R / layer (chosen on " + ",".join(pick) + ") | text | chunk hot-8 in set | hottest in set | "
        "rows on the set's experts |",
        "|---|---|---|---|---|",
    ]
    for R, cov in res["static"].items():
        for t, c in cov.items():
            lines.append(f"| {R} | {t} | {c['hot8_in']:.3f} | {c['top1_in']:.3f} | {c['rows_in']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pos", type=int, default=145408)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--pick", default="prose,code,math")
    ap.add_argument("--held", default="agentic,multilingual")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    d = Path(load_spec(a.spec).art) / "census"
    recs = {
        q.stem.split("_t", 1)[1]: json.loads(q.read_text())
        for q in sorted(d.glob(f"census_p{a.pos}_k{a.chunks}_t*.json"))
    }
    pick, held = a.pick.split(","), a.held.split(",")
    res = analyse(recs, pick, held)
    print(report(res, pick, held))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
