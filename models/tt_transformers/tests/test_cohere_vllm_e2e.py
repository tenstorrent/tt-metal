# SPDX-FileCopyrightText: © 2026 Canada Quant Labs (org-internal — bounty tt-metal#49307 track)
# SPDX-License-Identifier: Apache-2.0
#
# Command-R (c4ai-command-r-v01) vLLM-generator e2e smoke — P6(3).
#
# Drives the EXACT call path vLLM's TT plugin uses (minus the server):
#   CohereForCausalLM.initialize_vllm_model (the P5a model_type=="cohere"
#   family dispatch -> ModelArgs -> Transformer, use_paged_kv_cache=True)
#   -> allocate_vllm_kv_cache (paged cache, vLLM side)
#   -> prefill_forward_text -> greedy decode loop via decode_forward.
# Numerical correctness is NOT re-gated here — P6 full-model PCC >= 0.99
# already proves the math (results/2026-08-28-command-r-p6-fullmodel-pcc.md
# in tt-rd); this test proves the generator plumbing + real text out.
#
# Env knobs: COHERE_PROMPT, COHERE_DECODE_TOKENS (default 16),
#            COHERE_MAX_SEQ (default 8192 — config.json max_position_embeddings;
#            the rope 8192-vs-131072 question stays open, do not exceed).
#
# Run on the QB2 host via the side-load pattern (same as test_cohere_pcc.py):
#   python_env/bin/python -m pytest models/tt_transformers/tests/test_cohere_vllm_e2e.py -v -s

import os

import pytest
import torch
from loguru import logger

import ttnn  # noqa: F401  (import order matters for tt-metal env)
from models.tt_transformers.tt.generator_vllm import CohereForCausalLM, allocate_vllm_kv_cache

MESH_DEVICES = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}


@torch.no_grad()
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((1, 4), id="qb2-4xbh")],
    indirect=True,
)
def test_cohere_vllm_e2e_decode(mesh_device):
    from transformers import AutoConfig, AutoTokenizer

    hf_model = os.environ.get("HF_MODEL", "CohereLabs/c4ai-command-r-v01")
    max_seq_len = int(os.environ.get("COHERE_MAX_SEQ", "8192"))
    max_batch = 1
    block_size = 64
    max_num_blocks = max_seq_len // block_size
    decode_tokens = int(os.environ.get("COHERE_DECODE_TOKENS", "16"))
    prompt = os.environ.get("COHERE_PROMPT", "The capital of France is")

    logger.info(f"[cohere-e2e] hf_config for {hf_model}")
    hf_config = AutoConfig.from_pretrained(hf_model)

    logger.info(
        f"[cohere-e2e] CohereForCausalLM.initialize_vllm_model batch={max_batch} max_seq={max_seq_len}"
    )
    gen = CohereForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch, max_seq_len)
    model_args = gen.model_args[0]
    logger.info(f"[cohere-e2e] model_args.model_name={model_args.model_name} n_layers={model_args.n_layers}")

    # Paged KV cache, exactly what vLLM's allocate path builds per layer:
    # (max_num_blocks, n_local_kv_heads, block_size, head_dim) — read the
    # per-device shape off the constructed attention layer (attention.py
    # init_kv_cache uses the same fields).
    attn0 = gen.model[0].layers[0].attention
    kv_shape = (max_num_blocks, attn0.n_local_kv_heads, block_size, attn0.head_dim)
    logger.info(f"[cohere-e2e] allocate_vllm_kv_cache shape={kv_shape} x {model_args.n_layers} layers")
    kv_cache = allocate_vllm_kv_cache(
        kv_shape,
        torch.bfloat16,
        model_args.n_layers,
        dp_model=gen.model,
        tt_cache_path=gen.cache_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids  # [1, S]
    prompt_len = int(tokens.shape[1])
    logger.info(f"[cohere-e2e] prompt={prompt!r} tokens={prompt_len}")

    page_table = torch.arange(max_num_blocks, dtype=torch.int32).unsqueeze(0).repeat(max_batch, 1)

    logger.info("[cohere-e2e] prefill ...")
    logits = gen.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=torch.tensor([prompt_len]),
        sampling_params=None,  # host greedy
        enable_trace=False,
    )
    out_tok = torch.argmax(logits, dim=-1).reshape(-1)  # [B]
    gen_tokens = [int(out_tok[0].item())]
    logger.info(f"[cohere-e2e] prefill done, first token id={gen_tokens[0]!r} text={tokenizer.decode(gen_tokens)!r}")

    logger.info(f"[cohere-e2e] decode x {decode_tokens} ...")
    for i in range(decode_tokens):
        current_pos = torch.tensor([prompt_len + i])
        logits, _log_probs = gen.decode_forward(
            out_tok,
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=kv_cache,
            reset_batch=(i == 0),
            sampling_params=None,
            prompt_tokens=tokens,
            output_tokens=out_tok,
        )
        out_tok = torch.argmax(logits, dim=-1).reshape(-1)
        gen_tokens.append(int(out_tok[0].item()))
        logger.info(f"[cohere-e2e]   +{i}: id={gen_tokens[-1]} text={tokenizer.decode(gen_tokens)!r}")

    text = tokenizer.decode(gen_tokens)
    logger.info(f"[cohere-e2e] FINAL prompt={prompt!r} completion={text!r}")
    assert len(gen_tokens) == decode_tokens + 1
    # Weak smoke gate: greedy continuation must not be a single repeated token
    # (the classic broken-plumbing signature) and must be non-whitespace.
    assert text.strip(), "empty completion"
    assert len(set(gen_tokens)) > 1, f"degenerate repeated-token output: {gen_tokens}"
