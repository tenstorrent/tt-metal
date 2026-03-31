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
import ttnn
import torch

from utils.setup import setup_inference
from utils.inference_tr import (
    TrInferenceCtx,
    ttml_to_tt_transformers_state_dict,
    completion_batched_multiple_prompts_tr,
)
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision


# ── Config ────────────────────────────────────────────────────────────────────
YAML_CONFIG = "tt-train/sources/examples/grpo/grpo_model_accuracy.yaml"
HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
CHECKPOINT = None  # set to a .safetensors path to load trained weights
MAX_SEQ_LEN = 1024
MAX_BATCH = 32
MAX_NEW_TOKS = 64
TEMPERATURE = 0.0  # greedy
# ─────────────────────────────────────────────────────────────────────────────


def setup_tr_from_ttml(ttml_ctx, ttml_llama_cfg) -> TrInferenceCtx:
    """
    Convert a loaded ttml Llama model into a TrInferenceCtx backed by
    tt-transformers.  Reuses the device that ttml's AutoContext already opened,
    so there is no double device-open.

    Args:
        ttml_ctx:       InferenceCtx returned by setup_inference().
        ttml_llama_cfg: The LlamaConfig used when building the ttml model.
    """
    # ── Step 1: extract weights from the ttml model to CPU tensors ───────────
    print("[1/4] Extracting ttml model weights to CPU …")
    state_dict = ttml_to_tt_transformers_state_dict(
        ttml_ctx.tt_model,
        ttml_llama_cfg,
    )
    print(f"      Extracted {len(state_dict)} tensors.")

    # ── Step 2: reuse the device that initialize_device() already opened ─────
    # ttml stores its device in a C++ singleton; we grab the underlying handle
    # so we don't open a second device on the same chip.
    print("[2/4] Obtaining device handle from ttml AutoContext …")
    mesh_device = ttml.autograd.AutoContext.get_instance().get_device()

    # ── Step 3: build the tt-transformers model with our state dict ───────────
    print("[3/4] Building tt-transformers Transformer (no file I/O) …")
    paged_cfg = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    optimizations = lambda args: DecodersPrecision.performance(args.n_layers, args.model_name)

    # create_tt_model uses state_dict as-is when it is truthy, skipping
    # load_state_dict() which would otherwise download / read from disk.
    model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=MAX_BATCH,
        optimizations=optimizations,
        max_seq_len=MAX_SEQ_LEN,
        paged_attention_config=paged_cfg,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,  # <── inject our converted weights
    )

    # ── Step 4: wire up the Generator / page table ───────────────────────────
    print("[4/4] Wiring up Generator and page table …")
    tokenizer = model_args.tokenizer

    permutation = torch.randperm(paged_cfg.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        MAX_BATCH,
        paged_cfg.max_num_blocks // MAX_BATCH,
    )

    generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer)

    return TrInferenceCtx(
        generator=generator,
        model_args=model_args,
        model=model,
        tokenizer=tokenizer,
        tt_kv_cache=[tt_kv_cache],
        page_table=page_table,
        paged_attention_config=paged_cfg,
        mesh_device=mesh_device,
        max_tokens_to_complete=MAX_NEW_TOKS,
        temperature=TEMPERATURE,
        top_p=0.9,
        group_size=1,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH,
        instruct=True,
    )


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

    # setup_inference stores the LlamaConfig inside the TransformerConfig wrapper;
    # ttml_to_tt_transformers_state_dict needs num_hidden_layers and
    # num_key_value_heads, which live on the TransformerConfig.
    # We pass transformer_config directly since it has those attributes.
    ttml_llama_cfg = ttml_ctx.transformer_config

    # ── Convert weights and build tt-transformers model ────────────────────────
    print("\n=== Converting weights and building tt-transformers model ===")
    tr_ctx = setup_tr_from_ttml(ttml_ctx, ttml_llama_cfg)

    # ── Run inference ──────────────────────────────────────────────────────────
    user_prompts = [
        "The capital of France is",
        "The capital of Portugal is",
        "The capital of United Kingdom of Great Britain and Northern Ireland is",
        "The capital of Czech Republic is",
    ]

    print("\n=== Running generation via tt-transformers ===")
    prompts_tokenized = [tr_ctx.model_args.encode_prompt(p, instruct=True) for p in user_prompts]

    completions = completion_batched_multiple_prompts_tr(tr_ctx, prompts_tokenized)

    print()
    for prompt_text, completion_tokens in zip(user_prompts, completions):
        answer = tr_ctx.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        print(f"Q: {prompt_text}")
        print(f"A: {answer}")
        print()


if __name__ == "__main__":
    capitals_via_ttml_weights()
