#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
    print(f"iter_micro_batch, {len(completions)=}")
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


def train_grpo():
    ctx = setup(
        yaml_config_path="training_grpo_accuracy_unsloth_llama_3_2_1b_instruct.yaml",
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
        setup_optimizer=True,
    )

    prompts, answers = get_gsm8k(ctx, system_prompt, user_prompt_template_str, split="train", shuffle_seed=42)
    prompts = [ctx.tokenizer.encode(s) for s in prompts]

    prompts_to_train = 256
    prompts, answers = prompts[:prompts_to_train], answers[:prompts_to_train]

    base_lr = 1e-6
    warmup_steps = 20
    step = 0

    for prompts_batch, answers_batch, completions_batch in iter_batched_completions(
        ctx, prompts, answers, batch_size=4
    ):
        step += 1
        warmup_factor = min(1.0, step / warmup_steps)
        ctx.optimizer.set_lr(base_lr * warmup_factor)
        step_start = time.perf_counter()

        all_rewards = []
        ctx.optimizer.zero_grad()
        for p, ans, c in iter_micro_batch(prompts_batch, answers_batch, completions_batch):
            mb_start = time.perf_counter()
            B = len(c)

            r_np, adv_np = compute_rewards_advantages(ctx, ans, c)
            nonzero_adv = np.count_nonzero(adv_np)
            print(
                f"microbatch:  adv: nonzero={nonzero_adv}/{len(adv_np)}, min={adv_np.min():.4f}, max={adv_np.max():.4f}"
            )
            all_rewards.append(r_np)

            adv_tt = ttml.autograd.Tensor.from_numpy(
                adv_np.reshape((B, 1)), ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16
            )
            adv_tt.set_requires_grad(False)

            ctx.tt_model.eval()
            with no_grad():
                nlog_probs_old, mask, Tp = compute_nlog_probs(ctx, p, c)

            ctx.tt_model.train()
            nlog_probs_new, mask_new, _ = compute_nlog_probs(ctx, p, c)

            ratio = Exp.apply(nlog_probs_old - nlog_probs_new)
            eps = 0.2
            clipped_ratio = Clip.apply(ratio, 1.0 - eps, 1.0 + eps)

            surr1 = ratio * adv_tt
            surr2 = clipped_ratio * adv_tt

            surr = Min.apply(surr1, surr2)

            # After masked_surr = surr * mask [B, Tp]
            # Build per-token weight that incorporates 1/|o| normalization
            # mask is constant (no grad needed), so we can go to numpy
            mask_np = mask.to_numpy()  # [B, Tp]
            tokens_per_completion = mask_np.sum(axis=1, keepdims=True)  # [B, 1]
            tokens_per_completion = np.maximum(tokens_per_completion, 1.0)  # avoid div-by-zero

            # weight[b,t] = mask[b,t] / |o_b|
            # When we sum(surr * weight) over all elements, we get:
            #   sum_b [ (1/|o_b|) * sum_t surr[b,t]*mask[b,t] ]
            # Then divide by B (num completions) for the outer average.
            weight_np = (mask_np / tokens_per_completion).astype(np.float32)

            weight_tt = ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16)

            # surr * weight_tt gives per-token values already normalized by 1/|o|
            # ttml.ops.unary.mean gives sum_all / (B * Tp)
            # We need sum_all / B, so multiply by Tp to compensate
            # where B = len(c)

            weighted_surr = surr * weight_tt
            weighted_surr_4d = ttml.ops.reshape.reshape(weighted_surr, [1, 1, B, Tp])
            loss = ttml.ops.unary.mean(weighted_surr_4d) * (-float(Tp))

            loss.backward(retain_graph=False)
            mb_elapsed = time.perf_counter() - mb_start
            print(f"microbatch done! elapsed={mb_elapsed:.2f}s")

            deallocate_tensors(
                [
                    nlog_probs_old,
                    nlog_probs_new,
                    mask,
                    mask_new,
                    adv_tt,
                    ratio,
                    clipped_ratio,
                    surr1,
                    surr2,
                    surr,
                    weight_tt,
                    weighted_surr,
                    weighted_surr_4d,
                    loss,
                ]
            )

        ctx.optimizer.step()
        step_elapsed = time.perf_counter() - step_start
        all_rewards_np = np.concatenate(all_rewards)
        print(f"reward_mean={all_rewards_np.mean():.4f}, reward_std={all_rewards_np.std():.4f}")
        print(f"step={step} done! elapsed={step_elapsed:.2f}s")

    print("training process done!")


if __name__ == "__main__":
    train_grpo()
