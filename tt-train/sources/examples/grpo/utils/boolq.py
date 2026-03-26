import re
import numpy as np
from typing import List
from .setup import InferenceCtx


system_prompt = "Answer the question. You can explain if you want, but you don't have to."


def get_boolq(ctx: InferenceCtx, split="train", shuffle_seed=None):
    from datasets import load_dataset
    from string import Template

    user_prompt_template_str = """Passage: $passage

Question: $question
Answer:"""

    data = load_dataset("google/boolq")[split]
    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    t = Template(user_prompt_template_str)

    prompts = []
    answers = []
    for ex in data:
        user_content = t.substitute(passage=ex["passage"], question=ex["question"])
        prompt_str = ctx.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_str)
        answers.append(bool(ex["answer"]))

    return prompts, answers


def _correctness_reward(response_text: str, label: bool) -> float:
    response_text = response_text.lower().strip()
    expected = "yes" if label else "no"
    if response_text.startswith(expected):
        return 2.0
    return 0.0


def _absolute_brevity_reward(response_text: str) -> float:
    return -len(response_text) * 0.01


def compute_rewards_advantages(
    ctx: InferenceCtx,
    answers: List[bool],
    completions: List[List[int]],
):
    assert len(answers) == len(completions)

    completions_strs = [ctx.tokenizer.decode(c, skip_special_tokens=True) for c in completions]

    rewards_np = np.zeros(len(completions), dtype=np.float32)
    for i, (text, label) in enumerate(zip(completions_strs, answers)):
        rewards_np[i] = _correctness_reward(text, label) + _absolute_brevity_reward(text)

    advantages_np = np.zeros_like(rewards_np)
    G = ctx.group_size
    for start in range(0, len(rewards_np), G):
        end = min(start + G, len(rewards_np))
        rg = rewards_np[start:end]
        mu = float(rg.mean())
        advantages_np[start:end] = rg - mu

    return rewards_np, advantages_np
