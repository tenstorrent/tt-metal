# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-11 device run: benchmark accuracy, mixed-workload throughput and image request latency.

One process, one device open, the default model path (``TTDecider.from_pretrained``: stage-8
selected precision C0, vision tower loaded). Phases, in order:

1. ``warmup``: two requests per bucket present in the subsets and per image row (program compile).
2. ``accuracy``: the 600 frozen examples (``benchmark/datasets.py``) in file order, back to back,
   each through ``TTDecider.predict(state, question)``. Per example: answer, probabilities,
   correctness, seq_len, bucket, request latency. Prompts over 8192 tokens are recorded as
   rejected (``ValueError`` from ``bucket_for``); none are truncated.
3. ``throughput``: after ``IDLE_S`` s idle, the same 600 prompts in a seeded shuffled (mixed) order,
   back to back. Wall time, decisions/s, latency percentiles; the choices are compared with phase 2
   (determinism).
4. ``image``: request latency of golden image rows (``TTDecider.predict(..., images=[png])``),
   burst (``BURST_RUNS`` passes, each after ``IDLE_S`` s idle) and sustained (``SUSTAINED_RUNS``
   back to back), after ``WARMUP`` warm-ups.

Per-bucket text latency comes from the stage-6 harness (``tests/perf/test_model_perf.py``), run
separately. Usage::

    python -m models.demos.pplx_decider_v1_27b.benchmark.run_benchmark \
        --subsets $STAGE11/subsets --out $STAGE11/run
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import ttnn
from models.demos.pplx_decider_v1_27b.benchmark.datasets import SEED, load_subsets
from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider

IDLE_S = 10.0
WARMUP = 2
BURST_RUNS = 5
SUSTAINED_RUNS = 7
IMAGE_GOLDEN = Path("/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision/e2e/prompts.jsonl")
IMAGE_ROWS = ("v01_dominant_color", "v04_tallest_bar")  # fewest (256) and most (1024) patches of the golden set


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    return s[min(len(s) - 1, max(0, int(-(-q * len(s) // 100)) - 1))]


def latency_stats(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "mean_ms": statistics.fmean(ms),
        "p50_ms": percentile(ms, 50),
        "p90_ms": percentile(ms, 90),
        "p99_ms": percentile(ms, 99),
        "min_ms": min(ms),
        "max_ms": max(ms),
    }


def run_example(decider: TTDecider, ex: dict) -> dict:
    rec = {k: ex[k] for k in ("id", "benchmark", "seq_len", "bucket", "count", "label")}
    start = time.perf_counter()
    try:
        answer = decider.predict(ex["state"], ex["question"])
    except ValueError as err:  # > 8192 tokens: rejected like the app, never truncated
        rec.update(rejected=True, error=str(err), correct=False, prediction=None)
        return rec
    rec["latency_ms"] = (time.perf_counter() - start) * 1e3
    rec.update(
        rejected=False,
        prediction=answer["choice"],
        probabilities=answer["probabilities"],
        confidence=answer["confidence"],
        correct=answer["choice"] == ex["label"],
    )
    return rec


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subsets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    examples = load_subsets(args.subsets)
    golden = [json.loads(line) for line in IMAGE_GOLDEN.read_text().splitlines()]
    image_rows = {r["id"]: r for r in golden if r["id"] in IMAGE_ROWS}

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    summary = {"time_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        t = time.perf_counter()
        decider = TTDecider.from_pretrained(device, vision=True)
        summary["load_s"] = time.perf_counter() - t
        summary["policy"] = decider.model.config.optimizations.policy.name
        print(f"STAGE11 loaded in {summary['load_s']:.1f} s, policy {summary['policy']}", flush=True)

        # 1. warm-up: every bucket present + every image row
        t = time.perf_counter()
        for bucket in sorted({e["bucket"] for e in examples if e["bucket"] is not None}):
            ex = next(e for e in examples if e["bucket"] == bucket)
            for _ in range(WARMUP):
                decider.predict(ex["state"], ex["question"])
        for r in image_rows.values():
            for _ in range(WARMUP):
                decider.predict(r["row"]["state"], r["row"]["question"], images=r["row"]["images"])
        summary["warmup_s"] = time.perf_counter() - t

        # 2. accuracy pass, file order, back to back
        t = time.perf_counter()
        acc = []
        for i, ex in enumerate(examples):
            acc.append(run_example(decider, ex))
            if (i + 1) % 100 == 0:
                print(f"STAGE11 accuracy {i + 1}/{len(examples)}", flush=True)
        summary["accuracy_wall_s"] = time.perf_counter() - t
        write_jsonl(args.out / "predictions_accuracy.jsonl", acc)
        for name in dict.fromkeys(e["benchmark"] for e in acc):
            rows = [r for r in acc if r["benchmark"] == name]
            print(f"STAGE11 {name}: {sum(r['correct'] for r in rows)}/{len(rows)}", flush=True)

        # 3. throughput pass: idle, then seeded shuffled order, back to back
        order = list(range(len(examples)))
        random.Random(SEED).shuffle(order)
        time.sleep(IDLE_S)
        t = time.perf_counter()
        thr = [run_example(decider, examples[i]) for i in order]
        wall = time.perf_counter() - t
        write_jsonl(args.out / "predictions_throughput.jsonl", thr)
        first = {r["id"]: r for r in acc}
        served = [r for r in thr if not r["rejected"]]
        diffs = [
            max(abs(r["probabilities"][k] - first[r["id"]]["probabilities"][k]) for k in r["probabilities"])
            for r in served
        ]
        summary["throughput"] = {
            "order": f"random.Random({SEED}).shuffle over the 600 examples",
            "requests": len(thr),
            "served": len(served),
            "wall_s": wall,
            "decisions_per_s": len(served) / wall,
            "latency": latency_stats([r["latency_ms"] for r in served]),
            "latency_by_bucket": {
                str(b): latency_stats([r["latency_ms"] for r in served if r["bucket"] == b])
                for b in sorted({r["bucket"] for r in served})
            },
            "same_choice_as_accuracy_pass": sum(r["prediction"] == first[r["id"]]["prediction"] for r in served),
            "max_abs_prob_diff_vs_accuracy_pass": max(diffs),
        }
        acc_served = [r for r in acc if not r["rejected"]]
        summary["accuracy_pass"] = {
            "wall_s": summary["accuracy_wall_s"],
            "decisions_per_s": len(acc_served) / summary["accuracy_wall_s"],
            "latency": latency_stats([r["latency_ms"] for r in acc_served]),
            "rejected": len(acc) - len(acc_served),
        }
        tp = summary["throughput"]
        print(
            f"STAGE11 throughput {tp['served']} in {wall:.1f} s = {tp['decisions_per_s']:.3f}/s, "
            f"p50 {tp['latency']['p50_ms']:.1f} p90 {tp['latency']['p90_ms']:.1f} p99 {tp['latency']['p99_ms']:.1f} ms; "
            f"same choice {tp['same_choice_as_accuracy_pass']}/{tp['served']}, max |dp| {tp['max_abs_prob_diff_vs_accuracy_pass']:.2e}",
            flush=True,
        )

        # 4. image request latency
        summary["image"] = {}
        for rid, r in image_rows.items():
            row = r["row"]

            def request():
                s = time.perf_counter()
                out = decider.predict(row["state"], row["question"], images=row["images"])
                return (time.perf_counter() - s) * 1e3, out

            for _ in range(WARMUP):
                request()
            burst = []
            for _ in range(BURST_RUNS):
                time.sleep(IDLE_S)
                burst.append(request()[0])
            sustained, answer = [], None
            for _ in range(SUSTAINED_RUNS):
                ms, answer = request()
                sustained.append(ms)
            summary["image"][rid] = {
                "seq_len": r["seq_len"],
                "bucket": r["bucket"],
                "patches": r["patches"],
                "image_tokens": r["image_tokens"],
                "expected": r["meta"].get("expected"),
                "answer": answer,
                "burst": {**latency_stats(burst), "median_ms": statistics.median(burst), "samples_ms": burst},
                "sustained": {
                    **latency_stats(sustained),
                    "median_ms": statistics.median(sustained),
                    "samples_ms": sustained,
                },
            }
            print(
                f"STAGE11 image {rid} patches {r['patches']}: burst median {statistics.median(burst):.1f} ms, "
                f"sustained median {statistics.median(sustained):.1f} ms, choice {answer['choice']}",
                flush=True,
            )
    finally:
        summary["time_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
