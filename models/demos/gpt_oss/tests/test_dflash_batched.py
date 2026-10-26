# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched DFlash (tt/dflash/batched.py BatchedDFlashRunner): greedy DFlash for 32 / 128 users on the batched decode
model, every user's block of rows in one pass (32 users: 8-row blocks, a 256-row pass; 128 users: 4-row blocks, a
512-row pass). Each user's prompt is the demo's chat-template book prompt (256 tokens, its book cut at offset 997 p, 32
distinct prompts; user u reads prompt u % 32's K / V blocks). Checks the DFlash tokens against the target's own greedy
decode, teacher-forced (the batch-32 decode fed DFlash's tokens, its argmax at every position compared with DFlash's
next token; the first and last 32-user groups), and the mean accepted tokens per round; logs the tokens / s / user.

    pytest models/demos/gpt_oss/tests/test_dflash_batched.py
"""

import json
import time

import pytest
import torch
from loguru import logger

from models.demos.gpt_oss.demo.dflash_demo import (
    PROMPT_DATE,
    PROMPT_FILE,
    _no_book,
    fetch_books,
    parametrize_dflash_mesh,
)
from models.tt_transformers.demo.simple_text_demo import load_inputs

ISL = 256  # prompt tokens per user
NEW_TOKENS = 256  # decoded tokens per user
PROMPTS = 32  # distinct prompts (user u reads prompt u % 32)
N_CHECK = 32  # teacher-forced positions per user
# (measured on QuietBox 2, 2026-10-23: agreement 0.96-0.98 over 16 positions; 4.47 / 3.13 tokens per round, 234 / 106
# tok/s/user)
MIN_AGREEMENT = 0.9
MIN_ACCEPT = {32: 3.5, 128: 2.5}


def book_prompts(tokenizer, n, isl):
    """n prompts of isl token ids: a chat-template user turn ```book cut``` + the prompt file's question, prompt p's
    book cut at token offset 997 p."""
    (question,) = [u["prompt"] for u in json.load(open(PROMPT_FILE))]
    (text,), _ = load_inputs(str(PROMPT_FILE), 1, instruct=True)
    book = text[len("```") : text.rindex("```")]
    if not book:
        _no_book("the prompt's book could not be downloaded (gutenberg.org)")
    mark = "XQXSENTINELXQX"
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "```" + mark + "```\n\n" + question}],
        add_generation_prompt=True,
        tokenize=False,
        strftime_now=lambda fmt: PROMPT_DATE.strftime(fmt),
    )
    pre, post = (tokenizer.encode(t, add_special_tokens=False) for t in chat.split(mark))
    book_ids = tokenizer.encode(book, add_special_tokens=False)
    n_book = isl - len(pre) - len(post)
    return [pre + book_ids[(997 * p) % (len(book_ids) - n_book) :][:n_book] + post for p in range(n)]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("users", [32, 128], ids=["users_32", "users_128"])
@parametrize_dflash_mesh()
def test_dflash_batched(mesh_device, device_params, users, monkeypatch):
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.dflash.batched import BatchedDFlashRunner
    from models.tt_transformers.tt.generator import Generator

    monkeypatch.setenv("GPT_OSS_KV_FILE_CACHE", "0")  # (a fresh K / V pool: no file-cached cache)
    fetch_books(ISL)
    U = users
    sb = ISL // 64  # a prompt's 64-position blocks
    own_n = (
        -(-(NEW_TOKENS + 64 + 8) // 64) + 1
    )  # a user's own blocks: the decoded tokens + a block + the compile passes
    W = sb + own_n  # page table width
    base_own = PROMPTS * sb
    base_dummy = base_own + U * own_n  # (the prefills' padding past their prompts)
    pool = -(-max(base_dummy + 64, W + 1) // 32) * 32
    table = torch.empty(U, W, dtype=torch.int32)
    for u in range(U):  # prompt u % 32's blocks, then the user's own
        table[u, :sb] = (u % PROMPTS) * sb + torch.arange(sb)
        table[u, sb:] = base_own + u * own_n + torch.arange(own_n)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=1))
    model_args, model, _, kv_cache, tokenizer, processor, _ = text_demo.prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=32,
        optimizations=None,
        max_seq_len=W * 64,
        page_params={"page_block_size": 64, "page_max_num_blocks_per_dp": pool},
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=None,
    )
    if not model[0].layers[0].batched_decode:
        pytest.skip("batched DFlash needs the batched decode model (gpt-oss-20b, batch 32)")
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    V = model[0].vocab_size
    prompts = book_prompts(tokenizer, PROMPTS, ISL)
    runner = BatchedDFlashRunner(generator, model[0], table, kv_cache, users=U)  # (the default block / rows per pass)

    # the prompts: prompt p into its blocks (the prefill's page table row: its blocks, then padding); the drafter's
    # context for every user reading it
    firsts_p = []
    for p in range(PROMPTS):
        row = torch.empty(1, W, dtype=torch.int32)
        row[0, :sb] = p * sb + torch.arange(sb)
        row[0, sb:] = base_dummy + torch.arange(W - sb) % 64
        logits = generator.prefill_forward_text(
            torch.tensor([prompts[p]]), page_table=row, kv_cache=kv_cache, prompt_lens=[ISL], empty_slots=[p % 32],
            enable_trace=False, warmup_prefill=False,
        )  # fmt: skip
        firsts_p.append(int(logits.reshape(-1)[:V].float().argmax()))
        features = runner.capture.prefill_features_device()
        for u in range(p, U, PROMPTS):
            runner.draft.add_prompt_context(features, user=u, n=ISL)
        features.deallocate(True)
    firsts = [firsts_p[u % PROMPTS] for u in range(U)]

    # traced DFlash rounds (the acceptance on device)
    lens = [ISL] * U
    runner._set_inputs(lens, firsts)
    runner.capture_traces()
    try:
        t = time.perf_counter()
        toks, accept, times = runner.generate([[0] * ISL] * U, NEW_TOKENS, firsts)
        wall = time.perf_counter() - t
    finally:
        runner.release_traces()
    rounds = len(times)
    mean_accept = sum(sum(a) for a in accept) / sum(len(a) for a in accept)
    round_ms = wall / rounds * 1e3
    logger.info(
        f"batched DFlash: {U} users, {runner.block}-row blocks, {runner.rows}-row passes: {rounds} rounds of "
        f"{round_ms:.1f} ms, {mean_accept:.2f} tokens per round, {mean_accept * 1000 / round_ms:.1f} tok/s/user"
    )

    # teacher-forced against the batch-32 decode (the first and last 32-user groups; positions >= ISL, rewritten in
    # order): toks[u] starts with the prefill's token, dec[u][j] is DFlash's token after position ISL + j (a user's
    # tokens end at its first EOS: its later positions are fed its last token and not compared)
    dec = [t[1:] for t in toks]
    agree = []
    for g in sorted({0, U // 32 - 1}):
        us = range(32 * g, 32 * g + 32)
        a = n = 0
        for j in range(N_CHECK - 1):
            tok_in = [firsts[u] if j == 0 or not dec[u] else dec[u][min(j, len(dec[u])) - 1] for u in us]
            out = generator.decode_forward(
                torch.tensor(tok_in), torch.tensor([ISL + j] * 32), enable_trace=False,
                page_table=table[32 * g : 32 * g + 32], kv_cache=kv_cache, sampling_params=None, reload_inputs=True,
                reload_page_table=True, reload_sampling_params=False, reset_sampling_state=False,
            )  # fmt: skip
            logits = out[0] if isinstance(out, tuple) else out
            arg = logits.reshape(32, -1)[:, :V].float().argmax(-1).tolist()
            a += sum(int(arg[i] == dec[u][j]) for i, u in enumerate(us) if j < len(dec[u]))
            n += sum(j < len(dec[u]) for u in us)
        agree.append(a / n)
    logger.info(f"batched DFlash vs the batch-32 decode, teacher-forced over {N_CHECK - 1} positions: {agree}")
    assert min(agree) >= MIN_AGREEMENT, f"teacher-forced agreement {agree} < {MIN_AGREEMENT}"
    assert mean_accept >= MIN_ACCEPT[U], f"mean accepted tokens per round {mean_accept:.2f} < {MIN_ACCEPT[U]}"
