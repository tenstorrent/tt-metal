# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time

from utils.setup import setup_inference
from utils.gsm8k import get_gsm8k
from utils.inference import completion_batched_multiple_prompts

# =========== Config ==========
YAML_CONFIG = "tt-train/sources/examples/grpo/grpo_model_accuracy.yaml"
HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT = None  # set to a .safetensors path to load trained weights
MAX_BATCH = 32
# =============================


def capitals_via_ttml():
    ctx = setup_inference(
        yaml_config_path=YAML_CONFIG,
        hf_model_id=HF_MODEL_ID,
        checkpoint_path=CHECKPOINT,
        device_id=0,
    )

    prompts, answers = get_gsm8k(ctx, split="train", shuffle_seed=42)
    user_prompts = prompts[:MAX_BATCH]

    print(user_prompts)

    # Tokenize all prompts upfront; completion_batched_multiple_prompts expects List[List[int]]
    tokenized_prompts = [ctx.tokenizer.encode(p) for p in user_prompts]

    print("\n=== Running generation via tt-train ===")

    completions = []
    start_time = time.perf_counter()
    for i in range(0, len(tokenized_prompts), MAX_BATCH):
        batch = tokenized_prompts[i : i + MAX_BATCH]
        completions.extend(completion_batched_multiple_prompts(ctx, batch))

    print(f"{len(completions)} completions done, elapsed time: {time.perf_counter() - start_time} s")

    for i in range(len(completions)):
        print(f"completion {i} length = {len(completions[i])} tokens")

    print()
    for prompt_text, completion_tokens in zip(user_prompts, completions):
        answer = ctx.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        print(f"Q: {prompt_text}")
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    capitals_via_ttml()
