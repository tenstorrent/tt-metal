# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
MMLU-Redux benchmark for Qwen3.6-27B on TT P150a.

Evaluates 100 random samples from MMLU-Redux using log-prob multi-choice scoring.
Reports accuracy, TTFT, prefill throughput, and decode TPS grouped by prompt length bins.

Usage:
    python -u models/demos/qwen36_27b/evaluation/run_mmlu_bench.py [--samples 100] [--seed 42]
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import ttnn
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import load_state_dict
from models.demos.qwen36_27b.tt.model import TtQwen36Model
from models.demos.qwen36_27b.tt.generator import Qwen36Generator
from models.demos.qwen36_27b.tt.deltanet import USE_FUSED_KERNEL, USE_FULL_FUSED_KERNEL

LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "The following are multiple choice questions (with answers) about {subject_fmt}.\n\n"
)

SEQ_BINS = [(0, 64), (64, 96), (96, 128), (128, 9999)]
SEQ_BIN_LABELS = ["0-64", "64-96", "96-128", "128+"]

MODEL_PATH = "/tmp/qwen36_model/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"


def load_mmlu_redux(num_samples: int, seed: int) -> pd.DataFrame:
    """Download MMLU-Redux CSVs and return a random sample."""
    files = list_repo_files("edinburgh-dawg/mmlu-redux", repo_type="dataset")
    csv_files = sorted([f for f in files if f.endswith("test.csv")])

    all_rows = []
    for csv_file in csv_files:
        subject = csv_file.split("/")[0]
        path = hf_hub_download("edinburgh-dawg/mmlu-redux", csv_file, repo_type="dataset")
        df = pd.read_csv(path, header=None, skiprows=1)
        df.columns = [
            "question", "choices", "answer", "error_type",
            "source", "correct_answer", "potential_reason",
        ]
        df["subject"] = subject
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)
    sample_df = all_df.sample(n=min(num_samples, len(all_df)), random_state=seed)
    return sample_df.reset_index(drop=True)


def format_mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    subject_fmt = subject.replace("_", " ")
    prompt = SYSTEM_PROMPT.format(subject_fmt=subject_fmt)
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{LETTERS[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def get_seq_bin(token_len: int) -> int:
    for i, (lo, hi) in enumerate(SEQ_BINS):
        if lo <= token_len < hi:
            return i
    return len(SEQ_BINS) - 1


def main():
    parser = argparse.ArgumentParser(description="MMLU-Redux benchmark on TT device")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="/tmp/mmlu_bench_result.json")
    args = parser.parse_args()

    print(f"[Config] USE_FUSED_KERNEL={USE_FUSED_KERNEL}, USE_FULL_FUSED_KERNEL={USE_FULL_FUSED_KERNEL}")
    print(f"[Config] samples={args.samples}, seed={args.seed}")

    config = Qwen36ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    answer_token_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        answer_token_ids.append(ids[0])
    print(f"[Config] Answer token IDs: {dict(zip(LETTERS, answer_token_ids))}")

    print("[Data] Loading MMLU-Redux...")
    t0 = time.time()
    sample_df = load_mmlu_redux(args.samples, args.seed)
    print(f"[Data] Loaded {len(sample_df)} samples in {time.time() - t0:.1f}s")

    prompts = []
    for _, row in sample_df.iterrows():
        choices = ast.literal_eval(row["choices"])
        answer_idx = int(row["answer"])
        prompt_text = format_mmlu_prompt(row["question"], choices, row["subject"])
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompts.append({
            "text": prompt_text,
            "input_ids": input_ids,
            "token_len": len(input_ids),
            "answer_idx": answer_idx,
            "subject": row["subject"],
        })

    token_lens = [p["token_len"] for p in prompts]
    print(f"[Data] Token lengths: min={min(token_lens)}, max={max(token_lens)}, "
          f"mean={sum(token_lens)/len(token_lens):.0f}")

    print("[Weights] Loading...")
    t0 = time.time()
    state_dict = load_state_dict(config, model_path=MODEL_PATH)
    print(f"[Weights] Loaded in {time.time() - t0:.1f}s")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen36Model(device, state_dict, config)
        generator = Qwen36Generator(model, config, tokenizer=tokenizer)
        del state_dict
        print(f"[Model] Built in {time.time() - t0:.1f}s")

        # Warmup
        print("[Warmup] Running short prefill+decode...")
        generator.reset()
        warm_ids = torch.tensor([tokenizer.encode("Hello", add_special_tokens=False)], dtype=torch.long)
        logits = generator.prefill(warm_ids)
        logits_cpu = ttnn.to_torch(logits).float().reshape(-1)
        tok = torch.argmax(logits_cpu[:config.vocab_size]).item()
        generator.decode_one_token(torch.tensor([[tok]], dtype=torch.long))
        print("[Warmup] Done")

        # Evaluation loop
        results = []
        correct = 0
        total = 0
        t_eval_start = time.time()

        for idx, p in enumerate(prompts):
            generator.reset()

            input_ids = torch.tensor([p["input_ids"]], dtype=torch.long)
            if input_ids.shape[1] > config.max_seq_len:
                input_ids = input_ids[:, -config.max_seq_len:]

            # Prefill (TTFT measurement)
            t_prefill_start = time.perf_counter()
            logits_tt = generator.prefill(input_ids)
            t_prefill_end = time.perf_counter()
            ttft_s = t_prefill_end - t_prefill_start

            logits_cpu = ttnn.to_torch(logits_tt).float().reshape(-1)
            last_logits = logits_cpu[:config.vocab_size]
            log_probs = F.log_softmax(last_logits, dim=-1)

            answer_lps = [log_probs[tid].item() for tid in answer_token_ids]
            pred_idx = max(range(len(answer_lps)), key=lambda i: answer_lps[i])

            is_correct = pred_idx == p["answer_idx"]
            if is_correct:
                correct += 1
            total += 1

            # 1-token decode (decode TPS measurement)
            greedy_tok = torch.argmax(last_logits).item()
            t_decode_start = time.perf_counter()
            generator.decode_one_token(torch.tensor([[greedy_tok]], dtype=torch.long))
            t_decode_end = time.perf_counter()
            decode_latency_s = t_decode_end - t_decode_start

            result = {
                "idx": idx,
                "subject": p["subject"],
                "token_len": p["token_len"],
                "seq_bin": get_seq_bin(p["token_len"]),
                "correct": is_correct,
                "pred": LETTERS[pred_idx],
                "ref": LETTERS[p["answer_idx"]],
                "ttft_ms": ttft_s * 1000,
                "prefill_tps": p["token_len"] / ttft_s if ttft_s > 0 else 0,
                "decode_latency_ms": decode_latency_s * 1000,
                "decode_tps": 1.0 / decode_latency_s if decode_latency_s > 0 else 0,
            }
            results.append(result)

            elapsed = time.time() - t_eval_start
            acc = correct / total * 100
            lp_str = " ".join(f"{LETTERS[i]}={lp:.2f}" for i, lp in enumerate(answer_lps))
            print(
                f"  [{idx+1:>3}/{len(prompts)}] "
                f"{'OK' if is_correct else 'X '} "
                f"pred={LETTERS[pred_idx]} ref={LETTERS[p['answer_idx']]} "
                f"acc={acc:.0f}% "
                f"len={p['token_len']:>3} "
                f"ttft={ttft_s*1000:.0f}ms "
                f"dec={decode_latency_s*1000:.0f}ms "
                f"({elapsed:.0f}s) [{lp_str}]",
                flush=True,
            )

        total_time = time.time() - t_eval_start

        # Aggregate by sequence length bin
        bin_stats = defaultdict(lambda: {
            "correct": 0, "total": 0,
            "ttft_ms": [], "prefill_tps": [], "decode_tps": [],
        })
        for r in results:
            b = bin_stats[r["seq_bin"]]
            b["total"] += 1
            if r["correct"]:
                b["correct"] += 1
            b["ttft_ms"].append(r["ttft_ms"])
            b["prefill_tps"].append(r["prefill_tps"])
            b["decode_tps"].append(r["decode_tps"])

        # Print results table
        print(f"\n{'='*80}")
        print(f" MMLU-Redux Benchmark — Qwen3.6-27B (Full Fused DeltaNet) on P150a")
        print(f"{'='*80}")
        print(f" Overall: {len(results)} samples, Accuracy = {correct/total*100:.1f}%")
        print(f" Total eval time: {total_time:.0f}s ({total_time/len(results):.1f}s/sample)")
        print()
        print(f" {'Seq Length':>10}  {'Samples':>7}  {'Accuracy':>8}  {'Avg TTFT(ms)':>12}  {'Prefill TPS':>11}  {'Decode TPS':>10}")
        print(f" {'─'*10}  {'─'*7}  {'─'*8}  {'─'*12}  {'─'*11}  {'─'*10}")

        for i, label in enumerate(SEQ_BIN_LABELS):
            b = bin_stats[i]
            if b["total"] == 0:
                continue
            acc = b["correct"] / b["total"] * 100
            avg_ttft = sum(b["ttft_ms"]) / len(b["ttft_ms"])
            avg_prefill = sum(b["prefill_tps"]) / len(b["prefill_tps"])
            avg_decode = sum(b["decode_tps"]) / len(b["decode_tps"])
            print(
                f" {label:>10}  {b['total']:>7}  {acc:>7.1f}%  {avg_ttft:>12.0f}  {avg_prefill:>11.1f}  {avg_decode:>10.2f}"
            )

        # Overall row
        avg_ttft_all = sum(r["ttft_ms"] for r in results) / len(results)
        avg_prefill_all = sum(r["prefill_tps"] for r in results) / len(results)
        avg_decode_all = sum(r["decode_tps"] for r in results) / len(results)
        print(f" {'─'*10}  {'─'*7}  {'─'*8}  {'─'*12}  {'─'*11}  {'─'*10}")
        print(
            f" {'ALL':>10}  {len(results):>7}  {correct/total*100:>7.1f}%  "
            f"{avg_ttft_all:>12.0f}  {avg_prefill_all:>11.1f}  {avg_decode_all:>10.2f}"
        )
        print(f"{'='*80}")

        # Save JSON
        output = {
            "model": "Qwen3.6-27B",
            "kernel": f"fused={USE_FUSED_KERNEL}, full_fused={USE_FULL_FUSED_KERNEL}",
            "weights_dtype": str(config.weights_dtype),
            "dataset": "MMLU-Redux",
            "num_samples": len(results),
            "seed": args.seed,
            "overall_accuracy": correct / total * 100,
            "avg_ttft_ms": avg_ttft_all,
            "avg_prefill_tps": avg_prefill_all,
            "avg_decode_tps": avg_decode_all,
            "total_eval_time_s": total_time,
            "bins": {},
            "per_sample": results,
        }
        for i, label in enumerate(SEQ_BIN_LABELS):
            b = bin_stats[i]
            if b["total"] == 0:
                continue
            output["bins"][label] = {
                "samples": b["total"],
                "accuracy": b["correct"] / b["total"] * 100,
                "avg_ttft_ms": sum(b["ttft_ms"]) / len(b["ttft_ms"]),
                "avg_prefill_tps": sum(b["prefill_tps"]) / len(b["prefill_tps"]),
                "avg_decode_tps": sum(b["decode_tps"]) / len(b["decode_tps"]),
            }

        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
