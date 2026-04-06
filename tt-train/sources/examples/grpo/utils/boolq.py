from datasets import load_dataset
from typing import List, Tuple
import numpy as np
from .setup import (
    InferenceCtx,
)

system_prompt = "You are a wordy professor. Explain in 3 long sentences before saying Yes or No."


def get_boolq(ctx: InferenceCtx, split="train", shuffle_seed=None) -> Tuple[List[str], List[str]]:
    data = load_dataset("google/boolq", split=split)
    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    questions = [ex["question"] for ex in data]
    passages = [ex["passage"] for ex in data]
    answers = ["yes" if ex["answer"] else "no" for ex in data]

    prompts = [
        ctx.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": f"Question: {q}? Context: {p}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q, p in zip(questions, passages)
    ]

    return prompts, answers


def compute_rewards_advantages(ctx: InferenceCtx, answers: List[str], completions: List[List[int]]):
    assert len(answers) == len(completions)

    completions_strs = [ctx.tokenizer.decode(c, skip_special_tokens=True) for c in completions]

    rewards_np = np.zeros(len(completions), dtype=np.float32)
    for i, (text, ground_truth) in enumerate(zip(completions_strs, answers)):
        clean = text.strip().lower()
        accuracy = 2.0 if clean.startswith(ground_truth.lower()) else -1.0
        brevity = -0.1 * (len(text) / 20) ** 2
        rewards_np[i] = accuracy + brevity

    advantages_np = np.zeros_like(rewards_np)
    G = ctx.group_size
    for start in range(0, len(rewards_np), G):
        end = min(start + G, len(rewards_np))
        rg = rewards_np[start:end]
        mu = float(rg.mean())
        advantages_np[start : start + G] = rg - mu

    return rewards_np, advantages_np
