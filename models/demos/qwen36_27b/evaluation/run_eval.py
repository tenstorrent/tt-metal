# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark evaluation runner for Qwen3.6-27B on TT P150a.

Supports both lm-eval-harness tasks and custom evaluations.

Usage:
    # Run specific lm-eval benchmarks:
    python -m models.demos.qwen36_27b.evaluation.run_eval \
        --benchmarks mmlu_pro,gpqa_diamond \
        --limit 10 \
        --output results.json

    # Run all 10 benchmarks:
    python -m models.demos.qwen36_27b.evaluation.run_eval --all --output results.json

    # Quick pipeline test with dummy weights:
    python -m models.demos.qwen36_27b.evaluation.run_eval \
        --benchmarks gpqa_diamond --limit 5 --dummy-weights --output test_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


BENCHMARK_KEYS = [
    "mmlu_pro",
    "mmlu_redux",
    "gpqa_diamond",
    "ceval",
    "supergpqa",
    "aime26",
    "livecodebench_v6",
    "hmmt_feb25",
    "hle",
    "imo_answer_bench",
]

LM_EVAL_TASKS = {
    "mmlu_pro": "mmlu_pro",
    "gpqa_diamond": "gpqa_diamond_zeroshot",
    "ceval": "ceval-valid",
}

CUSTOM_EVAL_BENCHMARKS = {
    "mmlu_redux",
    "supergpqa",
    "aime26",
    "livecodebench_v6",
    "hmmt_feb25",
    "hle",
    "imo_answer_bench",
}


def load_published_scores() -> dict:
    scores_path = Path(__file__).parent / "published_scores.json"
    with open(scores_path) as f:
        return json.load(f)


def run_lm_eval_benchmark(
    task_name: str,
    max_layers: int | None = None,
    dummy_weights: bool = False,
    limit: int | None = None,
) -> dict:
    """Run a benchmark using lm-evaluation-harness."""
    from lm_eval import evaluator

    import models.demos.qwen36_27b.evaluation.lm_eval_wrapper  # registers the model

    model_args = []
    if max_layers is not None:
        model_args.append(f"max_layers={max_layers}")
    if dummy_weights:
        model_args.append("dummy_weights=True")

    results = evaluator.simple_evaluate(
        model="qwen36_tt",
        model_args=",".join(model_args) if model_args else None,
        tasks=[task_name],
        limit=limit,
        batch_size=1,
    )

    task_results = results.get("results", {})
    return task_results


def run_custom_benchmark(
    benchmark_key: str,
    max_layers: int | None = None,
    dummy_weights: bool = False,
    limit: int | None = None,
) -> dict:
    """Run a custom benchmark evaluation (for benchmarks not in lm-eval)."""
    import torch
    import ttnn
    from transformers import AutoTokenizer

    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.load_weights import load_state_dict, create_dummy_state_dict
    from models.demos.qwen36_27b.tt.model import TtQwen36Model
    from models.demos.qwen36_27b.tt.generator import Qwen36Generator

    config = Qwen36ModelConfig()
    if max_layers is not None:
        config.num_hidden_layers = max_layers

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    if dummy_weights:
        state_dict = create_dummy_state_dict(config, num_layers=config.num_hidden_layers)
    else:
        state_dict = load_state_dict(config, max_layers=max_layers)

    device = ttnn.open_device(device_id=0)
    try:
        model = TtQwen36Model(device, state_dict, config)
        generator = Qwen36Generator(model, config, tokenizer=tokenizer)
        del state_dict

        if benchmark_key == "mmlu_redux":
            return _eval_mmlu_redux(generator, tokenizer, config, limit)
        elif benchmark_key == "supergpqa":
            return _eval_supergpqa(generator, tokenizer, config, limit)
        elif benchmark_key == "hle":
            return _eval_hle(generator, tokenizer, config, limit)
        elif benchmark_key in ("aime26", "hmmt_feb25", "imo_answer_bench", "livecodebench_v6"):
            return _eval_generation_benchmark(generator, tokenizer, config, benchmark_key, limit)
        else:
            return {"error": f"Unknown benchmark: {benchmark_key}", "score": None}
    finally:
        ttnn.close_device(device)


def _eval_multichoice_hf(generator, tokenizer, config, dataset_name, dataset_config, split, limit, question_key, choices_key, answer_key, answer_index_key=None):
    """Generic multi-choice evaluation using HuggingFace datasets."""
    from datasets import load_dataset
    import torch
    import torch.nn.functional as F

    try:
        ds = load_dataset(dataset_name, dataset_config, split=split)
    except TypeError:
        ds = load_dataset(dataset_name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for idx, item in enumerate(ds):
        question = item[question_key]
        choices = item[choices_key]

        if answer_index_key and answer_index_key in item:
            answer_idx = item[answer_index_key]
        else:
            answer_raw = item[answer_key]
            if isinstance(answer_raw, int):
                answer_idx = answer_raw
            elif isinstance(answer_raw, str) and len(answer_raw) == 1 and answer_raw.isalpha():
                answer_idx = ord(answer_raw.upper()) - ord("A")
            else:
                answer_idx = 0

        best_ll = float("-inf")
        best_idx = -1

        for i, choice in enumerate(choices):
            prompt = f"{question}\nAnswer: {choice}"
            input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
            if input_ids.shape[1] > config.max_seq_len:
                input_ids = input_ids[:, -config.max_seq_len:]

            generator.reset()
            logits = generator.get_logits_for_sequence(input_ids)
            log_probs = F.log_softmax(logits[:, :config.vocab_size], dim=-1)

            choice_ids = tokenizer.encode(choice, add_special_tokens=False)
            ll = 0.0
            start = input_ids.shape[1] - len(choice_ids)
            for j, tid in enumerate(choice_ids):
                pos = start + j - 1
                if pos >= 0:
                    ll += log_probs[pos, tid].item()

            if ll > best_ll:
                best_ll = ll
                best_idx = i

        if best_idx == answer_idx:
            correct += 1
        total += 1

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(ds)}] accuracy so far: {correct/total*100:.1f}%")

    accuracy = correct / total * 100 if total > 0 else 0
    return {"score": accuracy, "correct": correct, "total": total}


def _eval_mmlu_redux(generator, tokenizer, config, limit):
    """Evaluate on MMLU-Redux dataset."""
    try:
        return _eval_multichoice_hf(
            generator, tokenizer, config,
            "edinburgh-dawg/mmlu-redux", None, "test", limit,
            "question", "choices", "answer",
        )
    except Exception as e:
        return {"error": str(e), "score": None}


def _eval_supergpqa(generator, tokenizer, config, limit):
    """Evaluate on SuperGPQA dataset."""
    try:
        return _eval_multichoice_hf(
            generator, tokenizer, config,
            "m-a-p/SuperGPQA", None, "test", limit,
            "question", "options", "answer",
        )
    except Exception as e:
        return {"error": str(e), "score": None}


def _eval_hle(generator, tokenizer, config, limit):
    """Evaluate on Humanity's Last Exam (HLE) dataset."""
    try:
        return _eval_multichoice_hf(
            generator, tokenizer, config,
            "cais/hle", None, "test", limit,
            "question", "choices", "answer",
        )
    except Exception as e:
        return {"error": str(e), "score": None}


def _eval_generation_benchmark(generator, tokenizer, config, benchmark_key, limit):
    """Placeholder for generation-based benchmarks (AIME, HMMT, LiveCodeBench, IMO)."""
    return {
        "score": None,
        "status": "not_implemented",
        "note": f"Generation-based benchmark '{benchmark_key}' requires specialized evaluation infrastructure. "
                "Use the lm-eval wrapper with appropriate task plugins when available.",
    }


def format_results_table(results: dict, published_scores: dict) -> str:
    """Format comparison table."""
    lines = []
    lines.append(f"{'Benchmark':<20} | {'Category':<12} | {'Published':>10} | {'BFP4_B':>10} | {'Delta':>8}")
    lines.append("-" * 72)

    benchmarks = published_scores.get("benchmarks", {})
    for key in BENCHMARK_KEYS:
        info = benchmarks.get(key, {})
        name = info.get("name", key)
        category = info.get("category", "")
        published = info.get("published_score", 0)

        result = results.get(key, {})
        score = result.get("score")

        if score is not None:
            delta = score - published
            lines.append(f"{name:<20} | {category:<12} | {published:>10.1f} | {score:>10.1f} | {delta:>+8.1f}")
        else:
            status = result.get("status", result.get("error", "pending"))
            lines.append(f"{name:<20} | {category:<12} | {published:>10.1f} | {'--':>10} | {status}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-27B benchmark evaluation")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated list of benchmark keys (e.g., mmlu_pro,gpqa_diamond)")
    parser.add_argument("--all", action="store_true",
                        help="Run all 10 benchmarks")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per benchmark")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Use only first N layers")
    parser.add_argument("--dummy-weights", action="store_true",
                        help="Use dummy random weights (for pipeline testing)")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if args.all:
        benchmark_keys = BENCHMARK_KEYS
    elif args.benchmarks:
        benchmark_keys = [b.strip() for b in args.benchmarks.split(",")]
    else:
        parser.error("Specify --benchmarks or --all")

    published = load_published_scores()
    all_results = {}

    for key in benchmark_keys:
        print(f"\n{'='*60}")
        print(f"Running benchmark: {key}")
        print(f"{'='*60}")
        t0 = time.time()

        if key in LM_EVAL_TASKS:
            lm_task = LM_EVAL_TASKS[key]
            try:
                task_results = run_lm_eval_benchmark(
                    lm_task, args.max_layers, args.dummy_weights, args.limit
                )
                if lm_task in task_results:
                    metrics = task_results[lm_task]
                    score = metrics.get("acc,none", metrics.get("acc_norm,none", metrics.get("exact_match,none")))
                    if score is not None:
                        score = score * 100
                    all_results[key] = {"score": score, "raw_metrics": metrics}
                else:
                    all_results[key] = {"score": None, "raw": task_results}
            except Exception as e:
                print(f"Error running lm-eval task {lm_task}: {e}")
                all_results[key] = {"error": str(e), "score": None}
        else:
            result = run_custom_benchmark(key, args.max_layers, args.dummy_weights, args.limit)
            all_results[key] = result

        elapsed = time.time() - t0
        score = all_results[key].get("score")
        print(f"  Result: {score if score is not None else 'N/A'} (elapsed: {elapsed:.1f}s)")

    output_data = {
        "model": "Qwen3.6-27B (BFP4_B on P150a)",
        "max_layers": args.max_layers,
        "dummy_weights": args.dummy_weights,
        "limit": args.limit,
        "results": all_results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print(f"\n{'='*72}")
    print("BENCHMARK COMPARISON: Qwen3.6-27B Published vs BFP4_B (P150a)")
    print(f"{'='*72}")
    print(format_results_table(all_results, published))


if __name__ == "__main__":
    main()
