# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 spec decode must be LOSSLESS: same bar and method as test_spec_lossless.py (plain greedy
reference vs DFlash2Decoder.generate, first mismatch must be a near-tie), with the DFlash2 drafter
in place of the MTP head. Acceptance is logged, never asserted.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_dflash2_lossless.py -v -s
Env: see test_spec_lossless.py (QWEN36_LOSSLESS_PROMPT_LEN, QWEN36_SPEC_NEAR_TIE_GAP, ...).
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


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_decode_is_lossless(mesh_device):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_decode import DFlash2Decoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    token_ids = _get_prompt(PROMPT_LEN, tokenizer)
    prompt_ids = token_ids[0].tolist()
    assert len(prompt_ids) == PROMPT_LEN
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    pt = torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, NUM_BLOCKS)
    near_tie_gap = float(os.environ.get("QWEN36_SPEC_NEAR_TIE_GAP", DEFAULT_NEAR_TIE_GAP))
    for layer in model.layers:
        if not layer.is_full_attention:
            layer.attention.use_fused_recurrent_decode = True

    ref, gaps = _reference_greedy(model, prompt_ids, pt, kv_shape, MAX_NEW, use_decode_step=True)
    logger.info(f"[dflash2-lossless] reference {len(ref)} tokens: {ref}")
    logger.info(f"[dflash2-lossless] reference text: {tokenizer.decode(ref)!r}")

    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    dec = DFlash2Decoder(model, pt)
    spec = dec.generate(prompt_ids, MAX_NEW)
    logger.info(f"[dflash2-lossless] dflash2   {len(spec)} tokens: {spec}")
    logger.info(f"[dflash2-lossless] dflash2 text:   {tokenizer.decode(spec)!r}")
    dec.log_stats(prefix="dflash2-lossless")
    logger.info(
        f"[dflash2-lossless] ttft={dec.prefill_time:.2f}s decode={len(spec) / max(dec.decode_time, 1e-9):.2f} tok/s "
        f"over {dec.iters} iters"
    )
    model.free_kv_caches()

    n = min(len(ref), len(spec))
    assert n == MAX_NEW, f"expected {MAX_NEW} tokens from both runs, got ref={len(ref)} spec={len(spec)}"
    ties = [i for i in range(n) if gaps[i] < near_tie_gap]
    logger.info(f"[dflash2-lossless] near-tie positions: {len(ties)}/{n} {ties}, min gap {min(gaps[:n]):.4f}")
    div = next((i for i in range(n) if spec[i] != ref[i]), None)
    if div is None:
        logger.info(f"[dflash2-lossless] PASSED: DFlash2 reproduced plain greedy for all {n} tokens")
        return
    gap = gaps[div]
    assert gap < near_tie_gap, (
        f"DFlash2 spec decode diverged from plain greedy at token {div}, where the reference was CONFIDENT:\n"
        f"  expected {ref[div]} ({tokenizer.decode([ref[div]])!r}), got {spec[div]} ({tokenizer.decode([spec[div]])!r})\n"
        f"  reference top-2 gap = {gap:.4f} >= {near_tie_gap}\n"
        f"  ref [:{div + 1}] = {ref[: div + 1]}\n  spec[:{div + 1}] = {spec[: div + 1]}"
    )
    logger.info(f"[dflash2-lossless] PASSED: lossless up to {div}, near-tie flip (gap={gap:.4f} < {near_tie_gap})")
