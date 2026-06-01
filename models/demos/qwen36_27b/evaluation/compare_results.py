# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Compare benchmark evaluation results against published Qwen3.6-27B scores.

Usage:
    python -m models.demos.qwen36_27b.evaluation.compare_results eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_published_scores() -> dict:
    scores_path = Path(__file__).parent / "published_scores.json"
    with open(scores_path) as f:
        return json.load(f)


def compare(results_path: str, format: str = "table"):
    with open(results_path) as f:
        eval_data = json.load(f)

    published = load_published_scores()
    benchmarks = published.get("benchmarks", {})
    results = eval_data.get("results", {})

    header_info = (
        f"Model: {eval_data.get('model', 'unknown')}\n"
        f"Layers: {eval_data.get('max_layers', 'all')}\n"
        f"Limit: {eval_data.get('limit', 'full')}\n"
        f"Dummy weights: {eval_data.get('dummy_weights', False)}\n"
    )

    if format == "table":
        print(header_info)
        print(f"{'Benchmark':<20} | {'Category':<12} | {'Published':>10} | {'BFP4_B':>10} | {'Delta':>8} | {'Retention':>10}")
        print("-" * 82)

        for key, info in benchmarks.items():
            name = info.get("name", key)
            category = info.get("category", "")
            pub_score = info.get("published_score", 0)

            result = results.get(key, {})
            score = result.get("score")

            if score is not None:
                delta = score - pub_score
                retention = (score / pub_score * 100) if pub_score > 0 else 0
                print(f"{name:<20} | {category:<12} | {pub_score:>10.1f} | {score:>10.1f} | {delta:>+8.1f} | {retention:>9.1f}%")
            else:
                status = result.get("status", result.get("error", "not run"))
                print(f"{name:<20} | {category:<12} | {pub_score:>10.1f} | {'--':>10} | {'--':>8} | {status}")

        scored = [(k, r["score"]) for k, r in results.items() if r.get("score") is not None]
        if scored:
            avg_retention = sum(
                r["score"] / benchmarks[k]["published_score"] * 100
                for k, r in results.items()
                if r.get("score") is not None and k in benchmarks
            ) / len(scored)
            print(f"\nAverage retention: {avg_retention:.1f}% across {len(scored)} benchmarks")

    elif format == "json":
        comparison = {}
        for key, info in benchmarks.items():
            result = results.get(key, {})
            score = result.get("score")
            pub_score = info.get("published_score", 0)
            comparison[key] = {
                "name": info.get("name"),
                "published": pub_score,
                "measured": score,
                "delta": (score - pub_score) if score is not None else None,
                "retention_pct": (score / pub_score * 100) if score is not None and pub_score > 0 else None,
            }
        print(json.dumps(comparison, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Compare Qwen3.6-27B benchmark results")
    parser.add_argument("results_file", help="Path to evaluation results JSON")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="Output format")
    args = parser.parse_args()
    compare(args.results_file, args.format)


if __name__ == "__main__":
    main()
