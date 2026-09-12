# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SERVING session decoder (tt/dflash2_serving.py) must be lossless across consecutive requests on
ONE set of captures, the way vLLM drives it: eager tap-capturing prefill -> ingest -> begin (seed the
runner's anchor) -> step() until the block budget is met -> end, then the next request re-uses every
trace/buffer (different prompt, different page-table row). Same bar and method as
test_dflash2_lossless (plain greedy reference; the first mismatch must be a near-tie).

Sessions: A = prompt 1 on the identity page table (also performs the one-time capture; eager seed);
B = a different prompt on a PERMUTED page table (exercises refresh_verify_page_table; traced seed);
C, D = prompt 1 again through the traced seed (persistence: D identical to C).

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_dflash2_serving.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS, _get_prompt
from models.demos.blackhole.qwen36.tests.test_spec_lossless import (
    DEFAULT_NEAR_TIE_GAP,
    MAX_NEW,
    NUM_BLOCKS,
    PROMPT_LEN,
    _reference_greedy,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PROMPT_LEN_B = int(os.environ.get("QWEN36_SERVING_PROMPT_LEN_B", 300))
# QWEN36_SERVING_CHAT=1: chat-templated prompts (what vLLM serves) instead of document continuations, to
# compare the drafter's acceptance under the serving text distribution.
CHAT_PROMPTS = (
    "Explain, step by step, how a transformer neural network processes a sequence of tokens to predict the next one.",
    "Give me a Python function that checks whether a string is a palindrome, with a docstring and three test cases.",
)


def _chat_prompt(tokenizer, text, thinking=False):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=thinking, tokenize=False
    )
    return [int(t) for t in tokenizer(rendered, add_special_tokens=False).input_ids]


def _serving_prefill(model, prompt_ids, page_table):
    """What Qwen36DFlashForCausalLM.prefill_forward does: the TP serving prefill with the taps armed.
    Returns (anchor token = greedy argmax of the prompt logits, taps)."""
    T = len(prompt_ids)
    tokens = torch.tensor([list(prompt_ids)], dtype=torch.int32)
    model._dflash_tap = True
    try:
        logits = model.prefill_traced_chunked(tokens, page_table, actual_len=T)
    finally:
        model._dflash_tap = False
    lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
    first = int(lt.reshape(-1)[: model.vocab_size].float().argmax())
    taps = model.take_dflash_eager_taps()
    assert taps is not None, "eager masked prefill captured no taps"
    return first, taps


def _serving_session(dec, model, prompt_ids, page_table, max_new, capture=False):
    first, taps = _serving_prefill(model, prompt_ids, page_table)
    T = len(prompt_ids)
    dec.ingest_prompt(taps, T)
    dec.begin(first, T, page_table)
    if capture:
        dec.capture()
    out = [first]
    while len(out) < max_new:
        committed = dec.step()
        assert committed is not None, "session hit capacity unexpectedly"
        out.extend(committed)
    dec.end()
    return out[:max_new]


def _check(tag, ref, gaps, got, tokenizer, near_tie_gap):
    n = min(len(ref), len(got))
    assert n == MAX_NEW, f"{tag}: expected {MAX_NEW} tokens, got ref={len(ref)} got={len(got)}"
    div = next((i for i in range(n) if got[i] != ref[i]), None)
    logger.info(f"[{tag}] ref  : {ref}")
    logger.info(f"[{tag}] spec : {got}")
    logger.info(f"[{tag}] text : {tokenizer.decode(got)!r}")
    if div is None:
        logger.info(f"[{tag}] PASSED: reproduced plain greedy for all {n} tokens")
        return
    gap = gaps[div]
    assert gap < near_tie_gap, (
        f"{tag}: diverged from plain greedy at token {div} where the reference was CONFIDENT "
        f"(gap {gap:.4f} >= {near_tie_gap}): expected {ref[div]} got {got[div]}"
    )
    logger.info(f"[{tag}] PASSED: lossless up to {div}, near-tie flip (gap={gap:.4f})")


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_serving_sessions_are_lossless(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_serving import DFlash2ServingDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    if os.environ.get("QWEN36_SERVING_CHAT", "0") == "1":
        thinking = os.environ.get("QWEN36_SERVING_THINKING", "0") == "1"
        prompt_a = _chat_prompt(tokenizer, CHAT_PROMPTS[0], thinking)
        prompt_b = _chat_prompt(tokenizer, CHAT_PROMPTS[1], thinking)
        logger.info(f"[serving] chat prompts: {len(prompt_a)} / {len(prompt_b)} tokens (thinking={thinking})")
    else:
        prompt_a = _get_prompt(PROMPT_LEN, tokenizer)[0].tolist()
        prompt_b = _get_prompt(PROMPT_LEN_B, tokenizer)[0].tolist()
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    ident = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    # A "vLLM-like" row for request B: blocks handed out from the top of the pool, block 0 unused.
    perm = torch.cat([torch.arange(NUM_BLOCKS - 1, 0, -1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)])
    perm = perm.reshape(1, NUM_BLOCKS)
    near_tie_gap = float(os.environ.get("QWEN36_SPEC_NEAR_TIE_GAP", DEFAULT_NEAR_TIE_GAP))
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.use_fused_recurrent_decode = True

    ref_a, gaps_a = _reference_greedy(model, prompt_a, ident, kv_shape, MAX_NEW, use_decode_step=True)
    ref_b, gaps_b = _reference_greedy(model, prompt_b, ident, kv_shape, MAX_NEW, use_decode_step=True)

    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    dec = DFlash2ServingDecoder(model, NUM_BLOCKS, ctx_blocks=NUM_BLOCKS)
    dec.alloc()
    try:
        got_a = _serving_session(dec, model, prompt_a, ident, MAX_NEW, capture=True)
        _check("serving-A", ref_a, gaps_a, got_a, tokenizer, near_tie_gap)
        got_b = _serving_session(dec, model, prompt_b, perm, MAX_NEW)
        _check("serving-B(permuted)", ref_b, gaps_b, got_b, tokenizer, near_tie_gap)
        got_c = _serving_session(dec, model, prompt_a, ident, MAX_NEW)
        _check("serving-C", ref_a, gaps_a, got_c, tokenizer, near_tie_gap)
        # A seeded eagerly (before the capture), B/C/D through the verify trace: A and C may differ at a
        # near-tie (both pass the bar above); two traced-seed sessions of the same prompt must be identical.
        got_d = _serving_session(dec, model, prompt_a, ident, MAX_NEW)
        _check("serving-D", ref_a, gaps_a, got_d, tokenizer, near_tie_gap)
        assert got_d == got_c, "the same prompt on the same captures must reproduce the previous traced-seed session"
    finally:
        dec.release()
        model.free_kv_caches()
