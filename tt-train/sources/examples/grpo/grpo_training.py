#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Sequence, Iterator

import os
from safetensors.numpy import save_file
import ttnn
import ttml
import time
from ttml.common.utils import no_grad

from utils.setup import InferenceCtx, setup_inference, setup_grpo_config, setup_training_optimizer
from utils.gsm8k import get_gsm8k
from utils.inference import completion_batched_multiple_prompts, deallocate_tensors
from utils.loss import compute_nlog_probs, compute_grpo_loss, compute_rewards_advantages


def save_grpo_checkpoint(model, checkpoint_dir: str, step: int) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensors = {}
    for name, param in model.parameters().items():
        tensors[name] = param.to_numpy(ttnn.DataType.FLOAT32)

    filepath = os.path.join(checkpoint_dir, f"grpo_step_{step}.safetensors")
    save_file(tensors, filepath)
    print(f"Saved checkpoint ({len(tensors)} tensors) to {filepath}")
    return filepath


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


def train_grpo():
    start_training = time.perf_counter()

    yaml_config_path = "tt-train/configs/training_configs/training_grpo_gsm8k_unsloth_llama_3_2_1b_instruct.yaml"
    ctx = setup_inference(
        yaml_config_path,
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )
    grpo_cfg = setup_grpo_config(yaml_config_path)
    optimizer = setup_training_optimizer(yaml_config_path, ctx.tt_model)

    prompts, answers = get_gsm8k(ctx, split="train", shuffle_seed=42)
    prompts = [ctx.tokenizer.encode(s) for s in prompts]

    prompts, answers = prompts[: grpo_cfg.prompts_to_train], answers[: grpo_cfg.prompts_to_train]

    batch = 0

    for prompts_batch, answers_batch, completions_batch in iter_batched_completions(
        ctx, prompts, answers, batch_size=grpo_cfg.completions_batch_size
    ):
        start_batch = time.perf_counter()
        batch += 1

        warmup_factor = min(1.0, batch / grpo_cfg.warmup_steps)
        optimizer.set_lr(grpo_cfg.base_lr * warmup_factor)

        rewards_np, advantages_np = compute_rewards_advantages(ctx, answers_batch, completions_batch)

        # Cache old log probs once
        probs_old_list = []
        ctx.tt_model.eval()
        with no_grad():
            for p, _, c in iter_micro_batch(prompts_batch, answers_batch, completions_batch, grpo_cfg.micro_batch_size):
                nlog_old, mask, Tp = compute_nlog_probs(ctx, p, c)
                nlog_old.set_requires_grad(False)
                mask.set_requires_grad(False)
                probs_old_list.append((nlog_old, mask, Tp))

        # --- Mini epoch loop ---
        for mini_epoch in range(grpo_cfg.num_mini_epochs):
            optimizer.zero_grad()
            ctx.tt_model.train()

            for i, (p, ans, c) in enumerate(
                iter_micro_batch(prompts_batch, answers_batch, completions_batch, grpo_cfg.micro_batch_size),
            ):
                B = len(c)
                adv_slice = advantages_np[i * grpo_cfg.micro_batch_size : i * grpo_cfg.micro_batch_size + B]

                adv_tt = ttml.autograd.Tensor.from_numpy(
                    adv_slice.reshape((B, 1)), ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16
                )
                adv_tt.set_requires_grad(False)

                nlog_old, mask_old, Tp = probs_old_list[i]
                nlog_probs_new, mask_new, _ = compute_nlog_probs(ctx, p, c)

                loss = compute_grpo_loss(
                    nlog_old, nlog_probs_new, mask_old, adv_tt, B, Tp, len(prompts_batch), grpo_cfg.clip_eps
                )
                loss.backward(retain_graph=False)

                deallocate_tensors([nlog_probs_new, mask_new, adv_tt, loss])

            optimizer.step()
            print("optimizer.step() called.")
            print(f"{mini_epoch=} done!")

        # Deallocate cached old log probs after all mini epochs are done
        for nlog_old, mask_old, _ in probs_old_list:
            deallocate_tensors([nlog_old, mask_old])

        print(f"reward_mean={rewards_np.mean():.4f}, reward_std={rewards_np.std():.4f}")
        print(f"batch={batch} done! elapsed={time.perf_counter() - start_batch:.2f} s")

        if batch % 100 == 1:
            save_grpo_checkpoint(
                ctx.tt_model,
                checkpoint_dir="generated/tt-train/grpo_checkpoints",
                step=batch,
            )
    print(f"training process done! {time.perf_counter() - start_training} s")
    save_grpo_checkpoint(
        ctx.tt_model,
        checkpoint_dir="generated/tt-train/grpo_checkpoints",
        step=batch,
    )


if __name__ == "__main__":
    train_grpo()
