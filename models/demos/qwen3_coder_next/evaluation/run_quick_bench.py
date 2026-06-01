# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Quick benchmark: Qwen3-Coder-Next with fused DeltaNet kernel on MMLU-Redux.

Standard MMLU evaluation: present question + labeled choices, then check
which answer letter (A/B/C/D) gets the highest next-token log probability.
"""

import json
import time
import sys

import torch
import torch.nn.functional as F
import ttnn

from datasets import load_dataset
from transformers import AutoTokenizer

from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.load_weights import load_state_dict
from models.demos.qwen3_coder_next.tt.model import TtQwen3CoderNextModel
from models.demos.qwen3_coder_next.tt.generator import Qwen3CoderNextGenerator
from models.demos.qwen3_coder_next.tt.deltanet import USE_FUSED_KERNEL

LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "The following are multiple choice questions (with answers) about {subject_fmt}.\n\n"
)


def format_mmlu_prompt(question, choices, subject):
    subject_fmt = subject.replace("_", " ")
    prompt = SYSTEM_PROMPT.format(subject_fmt=subject_fmt)
    prompt += f"{question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{LETTERS[i]}. {choice}\n"
    prompt += "Answer:"
    return prompt


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    subject = sys.argv[2] if len(sys.argv) > 2 else "high_school_mathematics"

    print(f"[Config] Fused DeltaNet kernel: {'ENABLED' if USE_FUSED_KERNEL else 'DISABLED'}")
    print(f"[Config] Subject: {subject}, Limit: {limit}")

    config = Qwen3CoderNextConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    answer_token_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        answer_token_ids.append(ids[0])
    print(f"[Config] Answer token IDs: {dict(zip(LETTERS, answer_token_ids))}")

    print("[Weights] Loading...")
    t0 = time.time()
    state_dict = load_state_dict(config)
    print(f"[Weights] Loaded in {time.time() - t0:.1f}s")

    device = ttnn.open_device(device_id=0)
    try:
        print("[Model] Building...")
        t0 = time.time()
        model = TtQwen3CoderNextModel(device, state_dict, config)
        generator = Qwen3CoderNextGenerator(model, config, tokenizer=tokenizer)
        del state_dict
        build_time = time.time() - t0
        print(f"[Model] Built in {build_time:.1f}s")

        ds = load_dataset("edinburgh-dawg/mmlu-redux", subject, split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))

        correct = 0
        total = 0
        t_eval_start = time.time()

        for idx, item in enumerate(ds):
            question = item["question"]
            choices = item["choices"]
            answer_raw = item["answer"]
            if isinstance(answer_raw, str) and len(answer_raw) == 1:
                answer_idx = ord(answer_raw.upper()) - ord("A")
            else:
                answer_idx = int(answer_raw)

            prompt = format_mmlu_prompt(question, choices, subject)
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long
            )
            if input_ids.shape[1] > config.max_seq_len:
                input_ids = input_ids[:, -config.max_seq_len:]

            generator.reset()
            logits = generator.get_logits_for_sequence(input_ids)
            last_logits = logits[-1, :config.vocab_size]
            log_probs = F.log_softmax(last_logits, dim=-1)

            answer_lps = [log_probs[tid].item() for tid in answer_token_ids]
            pred_idx = max(range(len(answer_lps)), key=lambda i: answer_lps[i])

            is_correct = pred_idx == answer_idx
            if is_correct:
                correct += 1
            total += 1

            elapsed = time.time() - t_eval_start
            acc = correct / total * 100
            lp_str = " ".join(f"{LETTERS[i]}={lp:.2f}" for i, lp in enumerate(answer_lps))
            print(f"  [{idx+1}/{len(ds)}] {'OK' if is_correct else 'X '} "
                  f"pred={LETTERS[pred_idx]} ref={LETTERS[answer_idx]} "
                  f"acc={acc:.0f}% ({elapsed:.0f}s) [{lp_str}]")

        total_time = time.time() - t_eval_start
        accuracy = correct / total * 100

        print(f"\n{'='*60}")
        print(f"MMLU-Redux ({subject}) — {total} questions")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"  Time: {total_time:.0f}s ({total_time/total:.1f}s/question)")
        print(f"  Fused kernel: {'YES' if USE_FUSED_KERNEL else 'NO'}")
        print(f"{'='*60}")

        result = {
            "benchmark": f"mmlu_redux_{subject}",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "time_s": total_time,
            "fused_kernel": USE_FUSED_KERNEL,
        }
        with open("/tmp/eval_quick_result.json", "w") as f:
            json.dump(result, f, indent=2)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
