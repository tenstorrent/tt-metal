#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from math import ceil
from ttml.common.utils import (
    no_grad,
)
import ttml
from ttml.autograd import Function
import ttnn
from typing import List, Sequence, Iterator, Tuple
from eval_utils import setup, get_gsm8k, extract_hash_answer
from batched_inference import completion_batched_multiple_prompts, compute_nlog_probs, InferenceCtx, deallocate_tensors
from ttml_operators import Min, Clip, Exp
import numpy as np
import time

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


def iter_batched_completions(
    ctx: InferenceCtx,
    prompts: Sequence[List[int]],
    answers: Sequence[float],
    batch_size: int = 32,
) -> Iterator[tuple[List[List[int]], List[float], List[List[int]]]]:
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])
        answers_batch = list(answers[start:end])

        start_time = time.perf_counter()
        completions_batch = completion_batched_multiple_prompts(ctx, prompt_batch)
        print(f"batch of completions done! {time.perf_counter() - start_time} s")

        answers_batch_yield = [item for item in answers_batch for _ in range(ctx.group_size)]
        prompt_batch_yield = [item for item in prompt_batch for _ in range(ctx.group_size)]

        assert len(prompt_batch_yield) == len(answers_batch_yield) == len(completions_batch)
        yield prompt_batch_yield, answers_batch_yield, completions_batch


def iter_micro_batch(prompts, answers, completions, micro_batch_size=16):
    for start in range(0, len(completions), micro_batch_size):
        end = min(start + micro_batch_size, len(completions))

        yield prompts[start:end], answers[start:end], completions[start:end]


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


def compute_grpo_loss(
    nlog_probs_old: ttml.autograd.Tensor,
    nlog_probs_new: ttml.autograd.Tensor,
    mask: ttml.autograd.Tensor,
    adv_tt: ttml.autograd.Tensor,
    B: int,
    Tp: int,
    completions_batch_len: int,
) -> Tuple[ttml.autograd.Tensor, list]:
    ratio = Exp.apply(nlog_probs_old - nlog_probs_new)
    eps = 0.2
    clipped_ratio = Clip.apply(ratio, 1.0 - eps, 1.0 + eps)

    surr1 = ratio * adv_tt
    surr2 = clipped_ratio * adv_tt
    surr = Min.apply(surr1, surr2)

    mask_np = mask.to_numpy()
    tokens_per_completion = np.maximum(mask_np.sum(axis=1, keepdims=True), 1.0)
    weight_np = (mask_np / tokens_per_completion).astype(np.float32)
    weight_tt = ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16)

    weighted_surr = surr * weight_tt
    weighted_surr_4d = ttml.ops.reshape.reshape(weighted_surr, [1, 1, B, Tp])
    loss = ttml.ops.unary.mean(weighted_surr_4d) * (-float(Tp) / completions_batch_len)

    return loss


def train_grpo():
    start_training = time.perf_counter()
    ctx = setup(
        yaml_config_path="training_grpo_accuracy_unsloth_llama_3_2_1b_instruct.yaml",
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
        setup_optimizer=True,
    )

    prompts, answers = get_gsm8k(ctx, system_prompt, user_prompt_template_str, split="train", shuffle_seed=42)
    prompts = [ctx.tokenizer.encode(s) for s in prompts]

    prompts_to_train = 32
    prompts, answers = prompts[:prompts_to_train], answers[:prompts_to_train]

    base_lr = 1e-6
    warmup_steps = 20
    batch = 0
    micro_batch_size = 16
    num_mini_epochs = 2

    for prompts_batch, answers_batch, completions_batch in iter_batched_completions(
        ctx, prompts, answers, batch_size=4
    ):
        start_batch = time.perf_counter()
        batch += 1
        warmup_factor = min(1.0, batch / warmup_steps)
        ctx.optimizer.set_lr(base_lr * warmup_factor)
        ctx.optimizer.zero_grad()

        rewards_np, advantages_np = compute_rewards_advantages(ctx, answers_batch, completions_batch)

        # Cache old log probs once
        probs_old_list = []
        ctx.tt_model.eval()
        with no_grad():
            for p, _, c in iter_micro_batch(prompts_batch, answers_batch, completions_batch, micro_batch_size):
                nlog_old, mask, Tp = compute_nlog_probs(ctx, p, c)
                nlog_old.set_requires_grad(False)
                mask.set_requires_grad(False)
                probs_old_list.append((nlog_old, mask, Tp))

        # --- Mini epoch loop ---
        for mini_epoch in range(num_mini_epochs):
            ctx.optimizer.zero_grad()
            ctx.tt_model.train()

            for i, (p, ans, c) in enumerate(
                iter_micro_batch(prompts_batch, answers_batch, completions_batch, micro_batch_size),
            ):
                B = len(c)
                adv_slice = advantages_np[i * micro_batch_size : i * micro_batch_size + B]

                adv_tt = ttml.autograd.Tensor.from_numpy(
                    adv_slice.reshape((B, 1)), ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16
                )
                adv_tt.set_requires_grad(False)

                nlog_old, mask_old, Tp = probs_old_list[i]
                nlog_probs_new, mask_new, _ = compute_nlog_probs(ctx, p, c)

                loss = compute_grpo_loss(nlog_old, nlog_probs_new, mask_old, adv_tt, B, Tp, len(prompts_batch))
                loss.backward(retain_graph=False)

                deallocate_tensors([nlog_probs_new, mask_new, adv_tt, loss])

            ctx.optimizer.step()

        # Deallocate cached old log probs after all mini epochs are done
        for nlog_old, mask_old, _ in probs_old_list:
            deallocate_tensors([nlog_old, mask_old])

        print(f"reward_mean={rewards_np.mean():.4f}, reward_std={rewards_np.std():.4f}")
        print(f"batch={batch} done! elapsed={time.perf_counter() - start_batch:.2f} s")

    print(f"training process done! {time.perf_counter() - start_training} s")


if __name__ == "__main__":
    train_grpo()
