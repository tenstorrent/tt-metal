# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed-query verify (spec decode) vs the sequential and batch-dim verifies.

The packed verify folds the K+1 candidates into the query-heads dim of ONE
batch=1 forward (head-major packed SDPA with an additive mask) and writes the
new KV loop-free through the persistent staging + paged_fill_cache path. Greedy
argmax must match the sequential single-token verify chain position-for-position
(up to the same near-tie caveat as the batch-dim verify); the loop-free KV write
must also leave the cache state correct across iterations (a chain of packed
verifies, with rollovers across page boundaries, keeps matching sequential).

Requires HF_MODEL (target). The drafter is irrelevant here.
"""

import math
import os

import pytest
import torch
from loguru import logger

from ...tests.test_factory import parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric()
def test_packed_verify_matches_sequential(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = "The capital of France is"
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 24, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    # assistant_model unused — only the verify paths run.
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )
    P = spec.draft_len + 1

    def _prefill():
        generator.prefill_forward_text(in_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos)

    # ── Reference: sequential single-token verify chain (batch=1 takes the
    # plain verify path — packing only engages for multi-token calls).
    _prefill()
    seq = []
    tok, pos = anchor_token, anchor_pos
    for _ in range(3 * P):  # several iterations ⇒ exercises staging rollovers
        logits, h = spec._verify([tok], [pos])
        h.deallocate(True)
        tok = int(torch.argmax(logits[0]))
        seq.append(tok)
        pos += 1

    # ── Packed: chain of packed verifies committing the matched chain tokens.
    _prefill()
    spec._pv_a_prev = -1
    packed = []
    pos = anchor_pos
    chain = [anchor_token] + seq
    it = 0
    while len(packed) < 3 * P:
        tokens = chain[it * P : it * P + P]  # the already-verified greedy chain
        positions = [pos + j for j in range(P)]
        lh, h = spec._verify(tokens, positions)
        h.deallocate(True)
        packed.extend(int(torch.argmax(lh[j])) for j in range(P))
        pos += P
        it += 1
    packed = packed[: len(seq)]

    logger.info(f"sequential: {seq}")
    logger.info(f"packed:     {packed}")
    n_match = sum(1 for a, b in zip(seq, packed) if a == b)
    logger.info(f"match {n_match}/{len(seq)}")
    assert packed == seq, f"packed verify diverges from sequential: packed={packed} seq={seq}"
