# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Serving-shaped single requests must be deterministic: one live row in a padded 32-row decode batch.

A vLLM server on a single-row mesh runs one request at a time through inputs shaped like this: the prefill gets the
request's block table right-padded with ZEROS to the maximum width (block 0 is vLLM's reserved null block), and every
decode step is a 32-row batch in which the request occupies one row (its index depends on what ran before) while the
other rows carry token 0, position -1 and an all-zero block table. The multi-user regression harness never runs that
shape (all 32 rows live, one contiguous page table), so this test replays identical greedy requests with different row
indices and block ids and requires identical tokens every time.

Run with TT_METAL_TRACE_ALLOC_TRACKING=1 to also fail on programs compiled after the traces were captured.
"""

import os

import pytest
import torch
from loguru import logger

from models.common.sampling import SamplingParams
from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
from models.demos.gpt_oss.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.gpt_oss.tests.test_multi_user_regression import (
    BLOCK_SIZE,
    PROMPT_FILES,
    _clear_kv_caches,
    _degenerate,
    _prompts_for,
)
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

PROMPT_TOKENS = [int(t) for t in os.getenv("GPT_OSS_PADDED_BATCH_TOKENS", "300,1500").split(",")]
GEN_TOKENS = 32
BATCH = 32
MAX_SEQ_LEN = 8192
WIDTH = MAX_SEQ_LEN // BLOCK_SIZE  # block-table width the plugin would use for this max_model_len
# (live row index, first block id, tail) per repeat. The block ids stay away from block 0 (vLLM's null block). `tail`
# is what the prefill block table carries beyond the request's own blocks: "zeros" (a fresh vLLM row) or "stale" (a
# reused row: vLLM's BlockTable.add_row rewrites only the first num_blocks entries, so the tail keeps earlier requests'
# block ids -- emulated here with this request's own ids, i.e. blocks the previous occupant freed and this request got).
LAYOUTS = [
    (0, 1, "zeros"),
    (0, 1, "zeros"),
    (5, 700, "stale"),
    (17, 41, "zeros"),
    (31, 2000, "stale"),
    (3, 1, "stale"),
    (9, 3300, "zeros"),
]


def _prompt_with_tokens(tokenizer, num_tokens):
    ids = tokenizer.encode(_prompts_for(max(PROMPT_FILES), 1)[0], add_special_tokens=False)
    return tokenizer.decode(ids[:num_tokens])


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 8)])
def test_padded_decode_batch_is_deterministic(mesh_device, device_params, state_dict):
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("single-row 1x8 serving shape")
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    torch.manual_seed(1234)
    model_args, models, _page_table, tt_kv_cache, tokenizer, _p, _c = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=BATCH,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params={"page_block_size": BLOCK_SIZE, "page_max_num_blocks_per_dp": BATCH * WIDTH},
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    sampling = SamplingParams(temperature=[0.0] * BATCH, top_k=[1] * BATCH, top_p=[1.0] * BATCH)

    def serve(prompt, row, first_block, tail):
        """One request the way the plugin submits it: single-row prefill table, then padded 32-row decode steps."""
        input_tokens, _, decoding_pos, _ = preprocess_inputs_prefill(
            [prompt],
            tokenizer,
            model_args,
            instruct=False,
            max_generated_tokens=GEN_TOKENS,
            max_prefill_len=MAX_SEQ_LEN,
        )
        tokens = torch.stack(input_tokens).view(1, -1)
        prompt_len = int(decoding_pos[0])
        num_blocks = (prompt_len + GEN_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE  # what vLLM allocates for the request
        row_table = torch.zeros(1, WIDTH, dtype=torch.int32)
        own = torch.arange(first_block, first_block + num_blocks, dtype=torch.int32)
        row_table[0, :num_blocks] = own
        if tail == "stale":  # a reused vLLM row: tail columns still hold earlier requests' ids (here: ours, re-issued)
            row_table[0, num_blocks:] = own.flip(0).repeat((WIDTH - num_blocks) // num_blocks + 1)[: WIDTH - num_blocks]
        _clear_kv_caches(models)
        generator.prev_page_table = None
        logits = generator.prefill_forward_text(  # the plugin hands the model the request's row only
            tokens,
            page_table=row_table,
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([prompt_len]),
            enable_trace=True,
            warmup_prefill=False,
        )
        out = [int(torch.argmax(logits, dim=-1)[0])]
        step_tokens = torch.zeros(BATCH, 1, dtype=torch.long)
        positions = torch.full((BATCH,), -1, dtype=torch.long)
        table = torch.zeros(BATCH, WIDTH, dtype=torch.int32)
        table[row, :num_blocks] = own  # the decode table is clean beyond the request's blocks
        for _ in range(GEN_TOKENS - 1):
            step_tokens[row, 0] = out[-1]
            positions[row] = prompt_len + len(out) - 1
            out_tok, _ = generator.decode_forward(
                step_tokens,
                positions,
                enable_trace=True,
                page_table=table,
                kv_cache=tt_kv_cache,
                sampling_params=sampling,
            )
            out.append(int(out_tok[row]))
        return prompt_len, out

    # Warm-up the way the plugin does at start-up: a full 32-row table, prefill trace + decode trace captured here.
    _clear_kv_caches(models)
    warm_tokens, _, warm_pos, _ = preprocess_inputs_prefill(
        [_prompts_for(128, 1)[0]],
        tokenizer,
        model_args,
        instruct=False,
        max_generated_tokens=GEN_TOKENS,
        max_prefill_len=MAX_SEQ_LEN,
    )
    generator.prefill_forward_text(
        torch.stack(warm_tokens).view(1, -1),
        page_table=torch.arange(1, BATCH * WIDTH + 1, dtype=torch.int32).view(BATCH, WIDTH),  # 32 rows, as at start-up
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor([int(warm_pos[0])]),
        enable_trace=True,
        warmup_prefill=True,
    )
    for n in PROMPT_TOKENS:  # compile each padded length before the traces would be replayed against it
        serve(_prompt_with_tokens(tokenizer, n), 0, 1, "zeros")

    failures = []
    for n in PROMPT_TOKENS:
        prompt = _prompt_with_tokens(tokenizer, n)
        results = []
        for row, first_block, tail in LAYOUTS:
            prompt_len, toks = serve(prompt, row, first_block, tail)
            text = tokenizer.decode(toks)
            results.append(toks)
            logger.info(
                f"{n} tokens (prefill len {prompt_len}) row {row} blocks from {first_block} tail {tail}: {text!r}"
            )
        distinct = {tuple(t) for t in results}
        bad = [tokenizer.decode(t) for t in results if _degenerate(t) or t != results[0]]
        if len(distinct) > 1 or bad:
            failures.append((n, len(distinct), bad[:3]))
        logger.info(f"{n} tokens: {len(distinct)} distinct output(s) over {len(results)} serving-shaped requests")
    assert not failures, "serving-shaped requests are not deterministic:\n" + "\n".join(
        f"{n} tokens: {d} distinct outputs, e.g. {b}" for n, d, b in failures
    )
