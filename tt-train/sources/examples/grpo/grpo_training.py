#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import (
    initialize_device,
    set_seed,
    create_optimizer,
    get_tt_metal_home,
    no_grad,
)
from ttml.common.config import load_config
import ttnn
import ttml
import numpy as np
import datasets
import time
from typing import List, TypeAlias, Any
from eval_utils import setup, extract_hash_answer, get_gsm8k

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


def iter_completions_rewards_advantages(
    ctx: InferenceCtx,
    prompts: Sequence[str],
    batch_size: int = 32,
) -> Iterator[tuple[int, List[int], List[List[int]], List[float]]]:
    """
    Yields aligned results one-by-one:
      i, prompts[i] as a list of tokens, list of ctx.group_size completions, where each completion is a list of tokens, and a list of advantages for
    """
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        prompt_batch = list(prompts[start:end])

        batch_completions = completion_batched_multiple_prompts(ctx, prompt_batch)

        for offset in range(batch_size):
            i = start + offset
            yield i, prompts[i], batch_completions[offset * ctx.group_size : (offset + 1) * ctx.batch_size]


def iter_micro_batch():
    for start in range(0, len(prompts), micro_batch_size):
        end = min(start + micro_batch_size, len(prompts))

        yield start, end


def compute_nlog_probs(inputs_np, targets_np, B, T) -> Any:
    """
    Takes np.arrays 'inputs_np', 'targets_np', returns a ttml tensor 'tokens_nlog', where
    for every i, j \in [0, B-1]x[0, T-2]
    tokens_nlog[i,j] = -log(prob(token[i,j])), where
    token[i,j] = vocab[targets_np[i,j]]
    """
    assert inputs_np.shape == (B, T - 1)
    assert targets_np.shape == (B, T - 1)

    X = inputs_np.astype(np.uint32)

    X_tt = ttml.autograd.Tensor.from_numpy(
        X.reshape(B, 1, 1, T - 1),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    mask_tensor = generate_causal_mask(T - 1, 0)  # [1, 1, T-1, T-1]
    logits = tt_model(X_tt, mask_tensor)  # [B, 1, T-1, V]

    targets_tt = ttml.autograd.Tensor.from_numpy(
        targets_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    tokens_nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)

    tokens_nlog = ttml.ops.reshape.reshape(tokens_nlog, [B, T - 1])

    assert tokens_nlog.shape() == [B, T - 1]
    return tokens_nlog


def train_grpo():
    ctx = setup(
        yaml_config_path="training_grpo_accuracy_unsloth_llama_3_2_1b_instruct.yaml",
        hf_model_id="unsloth/Llama-3.2-1B-Instruct",
        load_pretrained=True,
    )

    prompts, answers = get_gsm8k(ctx, system_prompt, user_prompt_template_str, split="train")
    prompts = [ctx.tokenizer.encode(s) for s in prompts]

    prompts_to_train = 2
    mini_batches = 2

    for i, p, cs in iter_completions_rewards_advantages(ctx, prompts, batch_size=2):
        for micro_batch_start, micro_batch_end in iter_micro_batch():
            micro_batch_len = micro_batch_end - micro_batch_start

            ctx.tt_model.eval()
            nlog_probs_old = compute_nlog_probs()

            ctx.tt_model.train()
            for mini_batch in range(mini_batches):
                nlog_probs_cur = compute_nlog_probs()

    print(prompts_tokens[0])


if __name__ == "__main__":
    train_grpo()
