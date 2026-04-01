# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
example_completion_tr.py — weight-transfer demo

Flow:
  1. Load a ttml LlamaCompositeKV model from HF weights (via setup_inference)
  2. Extract its parameters and convert them to a Meta-format state dict
     (using ttml_to_tt_transformers_state_dict from inference_tr.py)
  3. Hand that state dict to create_tt_model so tt-transformers skips file I/O
  4. Run batched generation and print answers
"""

import ttml
import time

from utils.setup import setup_inference
from utils.inference_tr import (
    completion_batched_multiple_prompts_tr,
    setup_tt_transformers_inference,
    sync_ttml_to_tt_transformers,
)

# =========== Config ==========
YAML_CONFIG = "tt-train/sources/examples/grpo/grpo_model_accuracy.yaml"
HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT = None  # set to a .safetensors path to load trained weights
MAX_SEQ_LEN = 2048
MAX_BATCH = 32
MAX_NEW_TOKS = 512
TEMPERATURE = 0.0  # greedy
# =============================


def capitals_via_ttml_weights():
    # ── Load ttml model (also calls initialize_device internally) ─────────────
    print("=== Loading ttml Llama model ===")
    from ttml.models.llama import LlamaConfig  # import only for the cfg type

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
        instruct=True,
    )

    start_time = time.perf_counter()
    sync_ttml_to_tt_transformers(ttml_ctx.tt_model, tr_ctx.model)
    print(f"sync, elapsed time: {time.perf_counter() - start_time} s")

    # ── Run inference ──────────────────────────────────────────────────────────
    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom of Great Britain and Northern Ireland is",
        "The capital of Czech Republic is",
    ] * 64

    print("\n=== Running generation via tt-transformers ===")
    prompts_tokenized = [tr_ctx.model_args.encode_prompt(p, instruct=True) for p in user_prompts]

    start_time = time.perf_counter()
    completions = []
    for i in range(0, len(user_prompts), MAX_BATCH):
        completions.extend(completion_batched_multiple_prompts_tr(tr_ctx, prompts_tokenized[i : i + MAX_BATCH]))

    print(f"{len(completions)} completions, elapsed time: {time.perf_counter() - start_time} s")

    for i in range(256):
        assert completions[i] == completions[i % 4]

    # print()
    # for prompt_text, completion_tokens in zip(user_prompts, completions):
    #    answer = tr_ctx.tokenizer.decode(completion_tokens, skip_special_tokens=True)
    #    print(f"Q: {prompt_text}")
    #    print(f"A: {answer}")
    #    print()


if __name__ == "__main__":
    capitals_via_ttml_weights()
