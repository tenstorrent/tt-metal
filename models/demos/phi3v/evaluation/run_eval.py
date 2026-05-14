#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark evaluation runner for Phi-3.5-vision-instruct on TT-Metal.

Usage (inside Docker container):
  cd /tt-metal
  HF_HOME=/home/yito/.cache/huggingface \
  HF_MODEL=microsoft/Phi-3.5-vision-instruct \
  python models/demos/phi3v/evaluation/run_eval.py \
      --benchmarks mmmu scienceqa mmbench mathvista ai2d chartqa textvqa pope \
      --num-samples 500 \
      --output-dir eval_results_phi3v/

Available benchmarks:
  mmmu, mmbench, scienceqa, mathvista, ai2d, chartqa, textvqa, pope

Options:
  --benchmarks   Space-separated list of benchmarks (default: all)
  --num-samples  Limit samples per benchmark for quick checks (default: all)
  --output-dir   Directory to save per-sample JSON results
  --max-new-tokens  Max tokens to generate per sample (default: 50)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn

from models.demos.phi3v.evaluation.benchmarks import BENCHMARK_REGISTRY, REFERENCE_SCORES
from models.demos.phi3v.evaluation.model_runner import Phi3VRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Phi-3.5-vision benchmark evaluation")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_REGISTRY.keys()),
        choices=list(BENCHMARK_REGISTRY.keys()),
        help="Benchmarks to run",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Limit samples per benchmark")
    parser.add_argument("--output-dir", default="eval_results_phi3v", help="Directory for result JSON files")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate per sample")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--dtype", choices=["bfp8", "bf16"], default="bfp8", help="Weight dtype")
    return parser.parse_args()


def print_summary(results: dict):
    print("\n" + "=" * 72)
    print(f"{'Benchmark':<15} {'Metric':<15} {'TT Score':>10} {'Reference':>10} {'Δ':>8}")
    print("-" * 72)
    for name, result in results.items():
        ref = REFERENCE_SCORES.get(name)
        tt_val = result.score * 100
        if ref:
            metric, ref_val = ref
            delta = tt_val - ref_val
            delta_str = f"{delta:+.1f}"
        else:
            metric, ref_val, delta_str = result.metric, "-", "-"
        print(f"{name:<15} {metric:<15} {tt_val:>10.1f} {str(ref_val):>10} {delta_str:>8}")
    print("=" * 72)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3.5-vision-instruct")
    weight_dtype = ttnn.bfloat16 if args.dtype == "bf16" else ttnn.bfloat8_b
    logger.info(f"Model: {hf_model}")
    logger.info(f"Weight dtype: {args.dtype}")
    logger.info(f"Benchmarks: {args.benchmarks}")
    logger.info(f"Samples per benchmark: {args.num_samples or 'all'}")

    runner = Phi3VRunner(
        hf_model=hf_model,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        dtype=weight_dtype,
    )

    try:
        runner.setup()
        results = {}
        t_start = time.time()

        for bench_name in args.benchmarks:
            bench_cls = BENCHMARK_REGISTRY[bench_name]
            bench = bench_cls()
            logger.info(f"\n{'='*60}\nStarting {bench.name}\n{'='*60}")

            try:
                result = bench.run(
                    runner,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                )
                results[bench.name] = result
                bench.save_results(result, args.output_dir)
            except Exception as e:
                logger.error(f"Benchmark {bench_name} failed: {e}")
                import traceback

                traceback.print_exc()

        total_time = time.time() - t_start
        logger.info(f"\nAll benchmarks completed in {total_time/60:.1f} minutes")

        print_summary(results)

        summary = {
            name: {
                "metric": r.metric,
                "score": r.score * 100,
                "num_samples": r.num_samples,
                "elapsed_sec": r.elapsed_sec,
                "reference": REFERENCE_SCORES.get(name, (r.metric, None))[1],
            }
            for name, r in results.items()
        }
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")

    finally:
        runner.teardown()


if __name__ == "__main__":
    main()
