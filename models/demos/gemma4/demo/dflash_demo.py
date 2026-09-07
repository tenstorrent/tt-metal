# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative-decode demo for Gemma4-31B on T3K, with a GEMMA4_USE_DFLASH toggle
so the same demo can be run with and without DFlash to compare generation parameters
(tok/s, mean acceptance length, ms/token) under matched conditions (same prompt, same
low-level prefill/decode path, same hardware).

Both paths go through the same directly-instantiated target model (create_tt_model) used
by tests/dflash/*, not the higher-level Gemma4Generator/trace machinery the production
spec-decode demo (text_demo_v2.py::_run_spec_decode) uses -- DFlash's layer_probe-based
context tapping is only proven against that lower-level path so far (see
docs/dflash_design.md's status section). This demo is a first real measurement, not yet
the fully-optimized/traced production path spec-decode already has.

Run with DFlash (eager):   pytest models/demos/gemma4/demo/dflash_demo.py -k 1x8 -s
Run with DFlash (traced):  GEMMA4_USE_TRACE=1 pytest models/demos/gemma4/demo/dflash_demo.py -k 1x8 -s
Run without DFlash:        GEMMA4_USE_DFLASH=0 pytest models/demos/gemma4/demo/dflash_demo.py -k 1x8 -s
"""

from __future__ import annotations

import math
import os
import time

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.dflash.config import Gemma4DFlashDrafterConfig
from models.demos.gemma4.tt.dflash.generate import dflash_generate
from models.demos.gemma4.tt.dflash.lm_head import load_gemma4_lm_head_weight
from models.demos.gemma4.tt.dflash.weights import load_gemma4_dflash_weights
from models.tt_transformers.tt.common import PagedAttentionConfig

# Naturally tokenizes to exactly 32 tokens (tile-aligned) for google/gemma-4-31B-it's
# chat template -- avoids _tile_align_prompt's newline-filler fallback below, which was
# found to bias generation toward repeating the filler token (a real side effect worth
# avoiding for a demo whose point is measuring real generation, not just KV-cache
# mechanics). Override via GEMMA4_DFLASH_PROMPT for a different prompt -- non-tile-aligned
# prompts still work correctly (padding is a documented, logged fallback, not silent) but
# may produce lower-quality continuations because of the injected filler.
DEFAULT_PROMPT = "The capital city of France, a large country located in Western Europe known for its culture, is"
MAX_SEQ_LEN = 128


def _model_path():
    return os.getenv("HF_MODEL", "google/gemma-4-31B-it")


def _tile_align_prompt(input_ids: torch.Tensor, tokenizer) -> tuple[torch.Tensor, int]:
    """Pad the tokenized prompt with real, attended-to filler tokens up to the next
    multiple of 32. Required by both generation paths below: the very first decode-style
    KV write after prefill hits a real Gemma4 kernel bug whenever the prefill context
    length isn't tile-aligned (see tt/dflash/verify.py's module docstring) -- ordinary
    single-token decode hits the identical bug, not just DFlash's verify. Uses the
    tokenizer's own newline token as filler, appended after the chat-templated prompt."""
    ctx_len = input_ids.shape[-1]
    remainder = ctx_len % 32
    if remainder == 0:
        return input_ids, ctx_len
    pad_n = 32 - remainder
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
    filler = torch.full((1, pad_n), newline_id, dtype=input_ids.dtype)
    logger.warning(
        f"Padding prompt from {ctx_len} to {ctx_len + pad_n} tokens (+{pad_n} newline filler) -- "
        "works around a known Gemma4 kernel bug where the first decode-style KV write after a "
        "non-tile-aligned prefill mispredicts once (see tt/dflash/verify.py)."
    )
    return torch.cat([input_ids, filler], dim=-1), ctx_len + pad_n


def _prepare_prompt(prompt, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)
    return _tile_align_prompt(input_ids, tokenizer)


def _make_target_model(mesh_device, model_path):
    page_params = {"page_block_size": 64, "page_max_num_blocks": math.ceil(MAX_SEQ_LEN / 64)}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks"]
    )
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_attention_config,
    )
    page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
        1, paged_attention_config.max_num_blocks
    )
    return model, tt_kv_cache, page_table


def _report(mode, ctx_len, output_ids, elapsed, acceptance_lengths, block_size, tokenizer):
    text = tokenizer.decode(output_ids)
    n_tokens = len(output_ids)
    n_iters = len(acceptance_lengths)
    mean_accept = (sum(acceptance_lengths) / n_iters) if n_iters else 0.0
    tok_s_u = n_tokens / elapsed if elapsed > 0 else 0.0
    ms_per_token = (elapsed * 1000.0 / n_tokens) if n_tokens else 0.0

    logger.info(f"\n== {mode} GENERATION ==\n{text.strip()}\n")
    logger.info(f"{mode} raw token ids: {output_ids}")
    logger.info(f"=== {mode} metrics ===")
    logger.info(f"Prompt tokens: {ctx_len}, generated tokens: {n_tokens}")
    if n_iters:
        logger.info(
            f"Drafter: block_size={block_size}; mean accepted {mean_accept:.2f}/{block_size - 1} "
            f"(tokens/iter: {mean_accept + 1:.2f}); verify iterations: {n_iters}"
        )
    logger.info(f"Decode: {ms_per_token:.2f} ms/token @ {tok_s_u:.2f} tok/s/user")
    return {
        "mode": mode,
        "text": text,
        "n_tokens": n_tokens,
        "n_iters": n_iters,
        "mean_accept": mean_accept,
        "tok_s_u": tok_s_u,
        "ms_per_token": ms_per_token,
        "elapsed": elapsed,
    }


def _run_dflash(prompt, max_generated_tokens, mesh_device, use_trace=False):
    model_path = _model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids, ctx_len = _prepare_prompt(prompt, tokenizer)

    config = Gemma4DFlashDrafterConfig.from_pretrained()
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=mesh_device.shape[1]))
    ccl_manager = CCLManager(mesh_device)
    weights = load_gemma4_dflash_weights(mesh_device, config, mesh_config)
    lm_head_weight = load_gemma4_lm_head_weight(mesh_device, mesh_config)

    model, tt_kv_cache, page_table = _make_target_model(mesh_device, model_path)
    input_ids_padded = torch.nn.functional.pad(input_ids.squeeze(0), (0, MAX_SEQ_LEN - ctx_len), value=0)
    stop_at_eos = os.environ.get("GEMMA4_DFLASH_STOP_AT_EOS", "1") == "1"
    stop_token_ids = [tokenizer.eos_token_id] if stop_at_eos and tokenizer.eos_token_id is not None else None

    t0 = time.perf_counter()
    output_ids, acceptance_lengths = dflash_generate(
        model,
        mesh_device,
        weights,
        lm_head_weight,
        config,
        mesh_config,
        ccl_manager,
        tt_kv_cache,
        page_table,
        input_ids_padded,
        ctx_len,
        max_generated_tokens,
        stop_token_ids=stop_token_ids,
        use_trace=use_trace,
    )
    elapsed = time.perf_counter() - t0
    mode = "DFLASH-TRACE" if use_trace else "DFLASH"
    return _report(mode, ctx_len, output_ids, elapsed, acceptance_lengths, config.block_size, tokenizer)


def _run_plain(prompt, max_generated_tokens, mesh_device):
    """Baseline: ordinary greedy single-token decode, no drafting -- same target model,
    same tile-aligned prompt, same hardware, for an apples-to-apples comparison."""
    model_path = _model_path()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_ids, ctx_len = _prepare_prompt(prompt, tokenizer)
    model, tt_kv_cache, page_table = _make_target_model(mesh_device, model_path)

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    page_table_tt = ttnn.from_torch(
        page_table, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32, mesh_mapper=replicate
    )
    is_mesh = hasattr(mesh_device, "shape")

    input_ids_padded = torch.nn.functional.pad(input_ids.squeeze(0), (0, MAX_SEQ_LEN - ctx_len), value=0)
    embeds, _, _, _, _, _ = model.prepare_inputs_prefill(input_ids_padded.unsqueeze(0), page_table=page_table_tt)
    prefill_logits = model.ttnn_prefill_forward(
        embeds,
        page_table=page_table_tt,
        kv_cache=tt_kv_cache,
        get_last_token=ctx_len - 1,
        input_ids_torch=input_ids_padded.unsqueeze(0),
        embeds_torch=None,
    )
    logits_torch = (
        ttnn.to_torch(ttnn.get_device_tensors(prefill_logits)[0]) if is_mesh else ttnn.to_torch(prefill_logits)
    )
    ttnn.deallocate(prefill_logits)
    tile_start = ((ctx_len - 1) // 32) * 32
    real_first_token = int(
        torch.argmax(logits_torch.float().reshape(1, -1, logits_torch.shape[-1])[0, ctx_len - 1 - tile_start]).item()
    )

    stop_at_eos = os.environ.get("GEMMA4_DFLASH_STOP_AT_EOS", "1") == "1"
    stop_tokens = {tokenizer.eos_token_id} if stop_at_eos and tokenizer.eos_token_id is not None else set()
    output_ids = [real_first_token]
    pos = ctx_len
    stopped = real_first_token in stop_tokens

    t0 = time.perf_counter()
    while len(output_ids) < max_generated_tokens and not stopped:
        x = torch.tensor([[output_ids[-1]]], dtype=torch.int64)
        x_tt = ttnn.from_torch(
            x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, device=mesh_device, mesh_mapper=replicate
        )
        pu = torch.zeros((1, 32), dtype=torch.int64)
        pu[0, 0] = pos
        pos_uint32 = ttnn.from_torch(
            pu, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, device=mesh_device, mesh_mapper=replicate
        )
        pos_int32 = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=replicate,
        )
        logits, _ = model.ttnn_decode_forward(
            x=x_tt, current_pos=pos_uint32, rot_mat_idxs=pos_int32, page_table=page_table_tt, kv_cache=tt_kv_cache
        )
        logits_torch = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]) if is_mesh else ttnn.to_torch(logits)
        ttnn.deallocate(logits)
        next_tok = int(torch.argmax(logits_torch.float().reshape(-1, logits_torch.shape[-1])[0]).item())
        output_ids.append(next_tok)
        pos += 1
        if next_tok in stop_tokens:
            stopped = True
    elapsed = time.perf_counter() - t0
    return _report("PLAIN", ctx_len, output_ids, elapsed, [], 0, tokenizer)


@parametrize_mesh_with_fabric([(1, 8)])
def test_demo_dflash(mesh_device, device_params):
    """GEMMA4_USE_DFLASH=1 (default): DFlash speculative decode. =0: plain greedy decode
    baseline. Run both (separately) to compare tok/s, mean acceptance length, ms/token."""
    prompt = os.environ.get("GEMMA4_DFLASH_PROMPT", DEFAULT_PROMPT)
    max_generated_tokens = int(os.environ.get("GEMMA4_DFLASH_MAX_NEW_TOKENS", 24))
    use_dflash = os.environ.get("GEMMA4_USE_DFLASH", "1") == "1"
    use_trace = os.environ.get("GEMMA4_USE_TRACE", "0") == "1"

    if use_dflash:
        metrics = _run_dflash(prompt, max_generated_tokens, mesh_device, use_trace=use_trace)
    else:
        metrics = _run_plain(prompt, max_generated_tokens, mesh_device)

    assert metrics["n_tokens"] > 0, f"{metrics['mode']} generation produced no tokens"
    return metrics
