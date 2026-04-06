from datasets import load_dataset
from typing import List, Tuple
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
