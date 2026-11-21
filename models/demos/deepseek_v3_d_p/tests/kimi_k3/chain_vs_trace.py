# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The GPU vLLM trace scored against the FP64 chain (``fp64_chain.py``) per context position. CPU only, no ttnn.

For every chain segment the chain's manifest lists as complete (the chain writes a segment's files atomically and
lists it only after both are written, so a listed segment is whole), the scorer computes per-row measures of the
trace against the FP64 products and caches them as ``<score>/segment_<i>.pt``; a rerun scores only segments it has
not cached, so it can follow a run in progress. Then it summarizes per window of context positions
(``windows.tsv``, ``metrics.json``) with each metric's trend (last window over first, and the worst window).

Measures, GPU (trace) against FP64 (chain):

* MoE routing, layers 1 and 2 (all rows): recall of the FP64 top-16 set, the identical-set fraction, the FP64
  routing weight on the experts the GPU misses (``dropped``), and each miss's rank in the FP64 order (best first);
  the FP64 16th-17th choice margin of rows whose sets differ against all rows (near-ties).
* MoE routing, layer 3 (sampled rows only; the chain routes them for its cross-check): recall from the chain's
  segment checks.
* DSA layer 3 keys: KV latent, k_pe and index key PCC and NRMSE per window (exact, from per-row sums).
* DSA layer 3 top-2048 (rows with a full selection, row >= 2047): recall of the FP64 selection, and each missed
  key's rank in the FP64 order and its spread share ``(score - cut) / (top - cut)`` in FP64 scores (near 0: a near-tie
  at the cut), as histograms so windows combine exactly.

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace --chain G --trace T --out S
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, atomic_save, segment_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import DSA_LAYER, MOE_LAYERS, state_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import Trace, missed_keys, topk_recall

SHARE_BINS = 100  # spread-share histogram bins over [0, 1]
KEYS = ("latent", "kpe", "index_key")


def key_sums(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per row ``[n, 7]`` FP64: sum x, sum y, sum xx, sum yy, sum xy, sum (x - y)^2, elements, with x the trace (got)
    and y the FP64 value (want); PCC and NRMSE over any set of rows follow exactly from the column sums."""
    x, y = got.double(), want.double()
    count = torch.full((x.shape[0],), float(x.shape[1]), dtype=torch.float64)
    return torch.stack(
        [x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1), (x * y).sum(1), ((x - y) ** 2).sum(1), count], 1
    )


def pcc_nrmse(sums: torch.Tensor) -> tuple[float, float]:
    """PCC and NRMSE (||x - y|| / ||y||) of the rows whose ``key_sums`` are given."""
    sx, sy, sxx, syy, sxy, see, n = sums.sum(0).tolist()
    cov = sxy - sx * sy / n
    var = (sxx - sx * sx / n) * (syy - sy * sy / n)
    return (cov / var**0.5 if var > 0 else float("nan")), (see / syy) ** 0.5 if syy > 0 else float("nan")


def score_segment(chain: Path, index: int, trace: Trace, checks: dict) -> dict:
    """Per-row measures of one chain segment's rows against the trace."""
    seg = torch.load(segment_path(chain, index), weights_only=True)
    keys = torch.load(state_path(chain, index), weights_only=True)["keys"]
    lo, hi = seg["start"], seg["end"]
    out: dict = {"start": lo, "end": hi}
    for l in MOE_LAYERS:
        want = seg[f"routing_ids_{l}"].long()
        got = trace.rows(f"expert_ids_layer_{l}", lo, hi).long()
        recall = topk_recall(got, want)
        rows = torch.arange(want.shape[0]).unsqueeze(1) << 32
        missed = ~torch.isin(rows + want, rows + got)
        rank = torch.arange(want.shape[1]).expand_as(want)[missed]  # the FP64 order is best first
        out[f"routing_{l}"] = {
            "recall": recall.float(),
            "dropped": (seg[f"routing_weights_{l}"].double() * missed).sum(1).float(),
            "margin": seg[f"routing_margin_{l}"].float(),
            "miss_rank_hist": torch.bincount(rank, minlength=want.shape[1]),
        }
    lora = keys["latent"].shape[1]
    kv = trace.rows(f"kv_post_transform_layer_{DSA_LAYER}", lo, hi)
    trace_keys = {
        "latent": kv[:, :lora],
        "kpe": kv[:, lora:],
        "index_key": trace.rows(f"indexer_k_layer_{DSA_LAYER}", lo, hi),
    }
    out["keys"] = {k: key_sums(trace_keys[k], keys[k]) for k in KEYS}
    want = seg["topk_indices_3"].long()
    got = trace.rows(f"dsa_topk_indices_layer_{DSA_LAYER}", lo, hi).long()
    topk = want.shape[1]
    full = torch.arange(lo, hi) >= topk - 1  # rows with a full selection on both sides
    recall = topk_recall(got, want)
    m = missed_keys(got[full], want[full], seg["topk_scores_3"][full].double())
    out["topk"] = {
        "recall": recall.float(),
        "full": full,
        "miss_rank_hist": torch.bincount(m["rank"], minlength=topk),
        "miss_share_hist": torch.histc(m["spread_share"].clamp(0, 1).float(), SHARE_BINS, 0, 1).long(),
    }
    out["routing_3_sampled"] = {
        "rows": int(checks.get("routing_sampled_rows", 0)),
        "recall_sum": float(checks.get("routing_recall_3_sampled", 0.0)) * int(checks.get("routing_sampled_rows", 0)),
    }
    return out


def score_landed(chain: Path, trace: Trace, out: Path, log=print) -> list[dict]:
    """Score (or read from ``out``'s cache) every segment the chain's manifest lists as complete."""
    manifest = json.loads((chain / MANIFEST).read_text())
    out.mkdir(parents=True, exist_ok=True)
    identity = {"chain": str(chain), "chain_token_ids_sha256": manifest["token_ids_sha256"], "trace": str(trace.path)}
    id_path = out / "identity.json"
    if id_path.is_file() and json.loads(id_path.read_text()) != identity:
        raise ValueError(f"{out} holds scores of another chain or trace: {json.loads(id_path.read_text())}")
    id_path.write_text(json.dumps(identity, indent=1))
    results = []
    for index, entry in enumerate(manifest["segments"]):
        cached = out / f"segment_{index:06d}.pt"
        if cached.is_file():
            results.append(torch.load(cached, weights_only=True))
            continue
        began = time.perf_counter()
        r = score_segment(chain, index, trace, entry.get("checks", {}))
        atomic_save(r, cached)
        results.append(r)
        log(
            f"scored segment {index + 1}/{len(manifest['segments'])} rows [{r['start']}, {r['end']}) {time.perf_counter() - began:.1f} s"
        )
    return results


def _hist_quantile(hist: torch.Tensor, q: float, scale: float = 1.0) -> float:
    total = int(hist.sum())
    if not total:
        return float("nan")
    return float(torch.searchsorted(hist.cumsum(0), torch.tensor(q * total)).item()) * scale


def summarize(results: list[dict], window: int) -> dict:
    """Per-window and overall measures from the per-segment results (windows aligned to segment boundaries)."""
    count = results[-1]["end"] if results else 0
    groups: list[list[dict]] = []
    for r in results:
        if not groups or r["start"] // window != groups[-1][0]["start"] // window:
            groups.append([])
        groups[-1].append(r)

    def measures(rs: list[dict]) -> dict:
        m = {"start": rs[0]["start"], "end": rs[-1]["end"]}
        for l in MOE_LAYERS:
            cat = {k: torch.cat([r[f"routing_{l}"][k] for r in rs]) for k in ("recall", "dropped", "margin")}
            differ = cat["recall"] < 1
            hist = sum(r[f"routing_{l}"]["miss_rank_hist"] for r in rs)
            m[f"routing_{l}_recall_mean"] = float(cat["recall"].mean())
            m[f"routing_{l}_set_equal"] = float((~differ).float().mean())
            m[f"routing_{l}_dropped_mean"] = float(cat["dropped"].mean())
            m[f"routing_{l}_dropped_p99"] = float(cat["dropped"].quantile(0.99))
            m[f"routing_{l}_miss_rank_median"] = _hist_quantile(hist, 0.5)
            m[f"routing_{l}_miss_top_half_fraction"] = (
                float(hist[: hist.numel() // 2].sum() / hist.sum()) if hist.sum() else float("nan")
            )
            m[f"routing_{l}_margin_median_differing"] = (
                float(cat["margin"][differ].median()) if differ.any() else float("nan")
            )
            m[f"routing_{l}_margin_median_all"] = float(cat["margin"].median())
        n3 = sum(r["routing_3_sampled"]["rows"] for r in rs)
        m["routing_3_sampled_rows"] = n3
        m["routing_3_sampled_recall_mean"] = (
            sum(r["routing_3_sampled"]["recall_sum"] for r in rs) / n3 if n3 else float("nan")
        )
        for k in KEYS:
            m[f"{k}_pcc"], m[f"{k}_nrmse"] = pcc_nrmse(torch.cat([r["keys"][k] for r in rs]))
        recall = torch.cat([r["topk"]["recall"][r["topk"]["full"]] for r in rs])
        rank = sum(r["topk"]["miss_rank_hist"] for r in rs)
        share = sum(r["topk"]["miss_share_hist"] for r in rs)
        m["topk_rows"] = int(recall.numel())
        m["topk_recall_mean"] = float(recall.mean()) if recall.numel() else float("nan")
        m["topk_recall_p01"] = float(recall.quantile(0.01)) if recall.numel() else float("nan")
        m["topk_misses_per_row"] = float(rank.sum()) / recall.numel() if recall.numel() else float("nan")
        m["topk_miss_rank_p10"] = _hist_quantile(rank, 0.1)
        m["topk_miss_rank_median"] = _hist_quantile(rank, 0.5)
        m["topk_miss_top_half_fraction"] = (
            float(rank[: rank.numel() // 2].sum() / rank.sum()) if rank.sum() else float("nan")
        )
        m["topk_miss_share_median"] = _hist_quantile(share, 0.5, 1 / SHARE_BINS)
        m["topk_miss_share_p90"] = _hist_quantile(share, 0.9, 1 / SHARE_BINS)
        return m

    windows = [measures(g) for g in groups]
    trend = {}
    for key in windows[0] if windows else []:
        if key in ("start", "end") or key.endswith("_rows"):
            continue
        values = [(w[key], w["start"]) for w in windows if w[key] == w[key]]
        if len(values) < 2:
            continue
        first, last = values[0][0], values[-1][0]
        worst = max(values) if "recall" not in key and "pcc" not in key and "set_equal" not in key else min(values)
        trend[key] = {
            "first": first,
            "last": last,
            "last_over_first": last / first if first else float("nan"),
            "worst": worst[0],
            "worst_at": worst[1],
        }
    return {
        "count": count,
        "window": window,
        "overall": measures(results) if results else {},
        "windows": windows,
        "trend": trend,
    }


def write_summary(out: Path, summary: dict) -> None:
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    windows = summary["windows"]
    if windows:
        columns = list(windows[0])
        (out / "windows.tsv").write_text(
            "\t".join(columns) + "\n" + "".join("\t".join(f"{w[c]:.6g}" for c in columns) + "\n" for w in windows)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain", type=Path, required=True, help="an fp64_chain run directory (complete or running)")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="score directory (per-segment cache and summaries)")
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    results = score_landed(args.chain, Trace(args.trace), args.out, log)
    summary = summarize(results, args.window)
    write_summary(args.out, summary)
    log(f"summarized {summary['count']} rows in {len(summary['windows'])} windows -> {args.out}")
    print(json.dumps(summary["overall"], indent=1))


if __name__ == "__main__":
    main()
