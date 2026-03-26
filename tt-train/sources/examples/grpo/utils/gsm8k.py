from datasets import load_dataset
from typing import List, Tuple
from .setup import (
    InferenceCtx,
)
from string import Template

system_prompt = "You are a helpful math tutor. Show your reasoning step by step and end with exactly one final line in this format: #### <number>"

user_prompt_template_str = """Question: There are 48 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 64 trees. How many trees did the grove workers plant today?
Answer: There are 48 trees originally.
Then there were 64 trees after some more were planted.
So there must have been 64 - 48 = 16.
#### 16

Question: If there are 13 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 13 cars.
2 more cars arrive.
13 + 2 = 15.
#### 15

Question: $question
Answer:
"""


def extract_hash_answer(text: str) -> float | None:
    if "####" not in text:
        return None
    s = text.split("####")[1].strip()

    if s is None:
        return float("nan")

    import re

    number_re = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")

    if not number_re.fullmatch(s):
        return float("nan")

    return float(s.replace(",", ""))


def get_gsm8k(ctx: InferenceCtx, split="train", shuffle_seed=None) -> Tuple[List[str], List[float]]:
    data = load_dataset("openai/gsm8k", "main")[split]
    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    dataset = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )

    questions = [ex["question"] for ex in dataset]
    answers = [ex["answer"] for ex in dataset]

    t = Template(user_prompt_template_str)

    prompts = [
        ctx.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": t.substitute(question=q)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    return prompts, answers


def compute_rewards_advantages(ctx: InferenceCtx, answers: List[float], completions: List[List[int]]):
    assert len(answers) == len(completions)

    completions_strs = [ctx.tokenizer.decode(c, skip_special_tokens=True) for c in completions]

    guesses: List[float | None] = [extract_hash_answer(s) for s in completions_strs]

    rewards_np = np.zeros(len(completions), dtype=np.float32)
    for i, (g, a) in enumerate(zip(guesses, answers)):
        if g is None or a is None:
            rewards_np[i] = 0.0
            continue
        rewards_np[i] = 1.0 if abs(g - a) < 1e-3 else 0.0  # or -1.0 for wrong if you prefer

    advantages_np = np.zeros_like(rewards_np)
    G = ctx.group_size
    for start in range(0, len(rewards_np), G):
        end = min(start + G, len(rewards_np))
        rg = rewards_np[start:end]
        mu = float(rg.mean())
        advantages_np[start : start + G] = rg - mu

    return rewards_np, advantages_np
