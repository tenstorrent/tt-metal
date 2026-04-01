# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttml
import time

from utils.setup import setup_inference
from utils.gsm8k import get_gsm8k
from utils.inference_tr import (
    completion_batched_multiple_prompts_tr,
    setup_tt_transformers_inference,
    sync_ttml_to_tt_transformers,
)

# =========== Config ==========
YAML_CONFIG = "tt-train/sources/examples/grpo/example_completion.yaml"
HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT = None  # set to a .safetensors path to load trained weights
MAX_SEQ_LEN = 2048
MAX_BATCH = 32
MAX_NEW_TOKS = 512
TEMPERATURE = 0.0  # greedy
# =============================


def capitals_via_ttml_weights():
    ttml_ctx = setup_inference(
        yaml_config_path=YAML_CONFIG,
        hf_model_id=HF_MODEL_ID,
        checkpoint_path=CHECKPOINT,
        device_id=0,
    )

    # ── Convert weights and build tt-transformers model ────────────────────────
    print("\n=== Converting weights and building tt-transformers model ===")
    mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
    tr_ctx = setup_tt_transformers_inference(
        mesh_device=mesh_device,
        tokenizer=ttml_ctx.tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH,
        max_tokens_to_complete=MAX_NEW_TOKS,
        temperature=TEMPERATURE,
        instruct=False,
    )

    start_time = time.perf_counter()
    sync_ttml_to_tt_transformers(ttml_ctx.tt_model, tr_ctx.model)
    print(f"sync, elapsed time: {time.perf_counter() - start_time} s")

    prompts, answers = get_gsm8k(ttml_ctx, split="train", shuffle_seed=42)
    user_prompts = prompts[:32]

    print(user_prompts)

    print("\n=== Running generation via tt-transformers ===")

    start_time = time.perf_counter()
    completions = []
    for i in range(0, len(user_prompts), MAX_BATCH):
        completions.extend(completion_batched_multiple_prompts_tr(tr_ctx, user_prompts[i : i + MAX_BATCH]))

    print(f"{len(completions)} completions done, elapsed time: {time.perf_counter() - start_time} s")

    for i in range(32):
        print(f"completion {i} length = {len(completions[i])} tokens")

    print()
    for prompt_text, completion_tokens in zip(user_prompts, completions):
        answer = tr_ctx.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        print(f"Q: {prompt_text}")
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    capitals_via_ttml_weights()
