# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""gpt-oss-20b with DFlash speculative decoding (greedy, batch 1) on a QuietBox 2 (1 x 4 Blackhole mesh).

The z-lab/gpt-oss-20b-DFlash drafter proposes 7 tokens per round from the target's hidden states; the target verifies
the 8-row block [last token, 7 drafts] in one pass and keeps the drafts up to the first mismatch plus its own next
token (tt/dflash/runner.py). Every round (draft, verify and context passes) runs on device as one trace; the host reads
one small result per round.

Prompt: a chat-template user turn with the book of models/ttt_compat/demo/sample_prompts/input_data_long_128k.json
(Frankenstein, Gutenberg #84) and that file's question about it, the book cut to the requested prompt length; past its
end (130,752 tokens) a second book follows (Pride and Prejudice, Gutenberg #1342). Both are downloaded on first use
(without network access the demo skips; under CI it fails). The chat template's "Current date" line is pinned (PROMPT_DATE): the
answer, and with it the drafts accepted per round, would otherwise change from day to day.

    pytest models/demos/gpt_oss/demo/dflash_demo.py -k "isl_1k"   # isl_256 .. isl_128k
"""

import datetime
import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tests.test_factory import L1_SMALL_SIZE
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM
from models.ttt_compat.demo.simple_text_demo import load_and_cache_context, load_inputs

PROMPT_FILE = Path(__file__).parents[3] / "ttt_compat" / "demo" / "sample_prompts" / "input_data_long_128k.json"
SECOND_BOOK = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
PROMPT_CACHE = Path("models/ttt_compat/demo/context_cache")  # load_inputs' download cache
DRAFT_WINDOW = 1024  # the drafter attends to the last 1024 positions (SGLang's --speculative-dflash-draft-window-size)
PROMPT_DATE = datetime.date(2026, 10, 17)  # the chat template's "Current date" (it is part of the prompt)
NEW_TOKENS = 256

# Device: the fused DFlash programs are large; 48 KB more kernel-config ring per worker (taken from the L1 allocator,
# Blackhole default 1461120 B) lets dispatch stage the next program's binaries while the current one runs, and the
# largest fabric packets (15232 B) carry each all-reduce partial in fewer packets.
WORKER_L1_SIZE = 1461120 - 48 * 1024
FABRIC_PAYLOAD = 15232


def dflash_device_params():
    cfg = ttnn.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = FABRIC_PAYLOAD
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "l1_small_size": L1_SMALL_SIZE,
        TRACE_MODEL_KEY_PARAM: "gpt-oss-120b",
        "worker_l1_size": WORKER_L1_SIZE,
        "fabric_router_config": cfg,
    }


def parametrize_dflash_mesh():
    """A QuietBox 2 (the 4 Blackhole chips are the whole system: the fused batch-1 decode DFlash builds on); else
    skipped."""
    supported = is_blackhole() and ttnn.get_num_devices() == 4
    marks = [] if supported else [pytest.mark.skip(reason="DFlash runs on a QuietBox 2 (4 Blackhole chips) only")]
    return pytest.mark.parametrize(
        "mesh_device, device_params",
        [pytest.param((1, 4), dflash_device_params(), id="1x4", marks=marks)],
        indirect=True,
    )


def build_dflash(mesh_device, max_seq_len):
    """(runner, tokenizer) for one user with a KV cache of max_seq_len positions."""
    import models.demos.gpt_oss.demo.text_demo as text_demo
    from models.demos.gpt_oss.tt.dflash.draft import DFlashDraft
    from models.demos.gpt_oss.tt.dflash.runner import DFlashRunner
    from models.ttt_compat.tt.generator import Generator

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=1))
    model_args, model, page_table, kv_cache, tokenizer, processor, _ = text_demo.prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params={"page_block_size": 64, "page_max_num_blocks_per_dp": max_seq_len // 64},
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=None,
    )
    if not model[0].fused_decode:
        pytest.skip("DFlash needs the fused batch-1 decode (gpt-oss-20b, tt/fused_decode/config.py)")
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    draft = DFlashDraft(mesh_device, model[0], max_positions=max_seq_len)
    return DFlashRunner(generator, model[0], page_table, kv_cache, draft), tokenizer


def _no_book(reason):
    """A prompt book could not be downloaded: skipped locally, a failure under CI (where a skip would pass silently)."""
    (pytest.fail if os.getenv("CI") == "true" else pytest.skip)(reason)


def fetch_books(isl):
    """Download (or find in the cache) the books an isl-token prompt needs, before the model is loaded."""
    (text,), _ = load_inputs(str(PROMPT_FILE), 1, instruct=True)
    if not text[len("```") : text.rindex("```")]:
        _no_book("the prompt's book could not be downloaded (gutenberg.org)")
    if isl > 100_000 and not load_and_cache_context(SECOND_BOOK, PROMPT_CACHE):  # (the first book is 101,749 tokens)
        _no_book(f"the second book could not be downloaded ({SECOND_BOOK})")


def book_prompt(tokenizer, isl):
    """isl token ids: a chat-template user turn ```book``` + the prompt file's question, the book cut to fit."""
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
        strftime_now=lambda fmt: PROMPT_DATE.strftime(fmt),  # (a template variable shadows the template's clock)
    )
    pre, post = (tokenizer.encode(t, add_special_tokens=False) for t in chat.split(mark))
    book_ids = tokenizer.encode(book, add_special_tokens=False)
    n_book = isl - len(pre) - len(post)
    if n_book > len(book_ids):
        second = load_and_cache_context(SECOND_BOOK, PROMPT_CACHE)
        if not second:
            _no_book(f"the second book could not be downloaded ({SECOND_BOOK})")
        book_ids += tokenizer.encode(second, add_special_tokens=False)
    assert 0 <= n_book <= len(book_ids), f"ISL {isl}: the prompt holds {len(pre) + len(post)} + {len(book_ids)} tokens"
    return pre + book_ids[:n_book] + post


def run_dflash(runner, prompt_ids, new_tokens, window=DRAFT_WINDOW):
    """Prefill, one eager round batch (compiles the programs), trace capture, then the traced generation. Returns
    (tokens, per-round accepted counts, seconds of the traced rounds)."""
    first = runner.prefill(prompt_ids)
    runner.set_draft_window(window)
    runner.generate(prompt_ids, 16, first=first)
    runner.capture_traces()
    t = time.perf_counter()
    toks, accept, timing = runner.generate(prompt_ids, new_tokens, first=first)
    rounds_s = sum(sum(r) for r in timing["rounds"])
    logger.info(f"DFlash: {len(toks)} tokens in {time.perf_counter() - t:.2f} s")
    assert accept, "no DFlash round ran (the prefill's token ended the generation)"
    return toks, accept, rounds_s


ISLS = {"256": 256, "512": 512, "1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
ISLS |= {"64k": 65536, "128k": 130752}  # (130752 + 256 new tokens fit the 131072-position KV cache)
# perf regression guard: ms per round at most = 1.2 x the measured round time (QuietBox 2, 2026-10-19: 4.842, 4.749,
# 4.967, 4.957, 4.923, 5.074, 5.162, 5.474, 5.921, 6.739 ms at 256 .. 130752), rounded up to 0.1 ms
MAX_ROUND_MS = dict(zip(ISLS.values(), [5.9, 5.7, 6.0, 6.0, 6.0, 6.1, 6.2, 6.6, 7.2, 8.1]))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("isl", list(ISLS.values()), ids=[f"isl_{k}" for k in ISLS])
@parametrize_dflash_mesh()
def test_dflash_demo(mesh_device, device_params, isl):
    fetch_books(isl)
    max_seq_len = max(4096, 1 << (isl + NEW_TOKENS + 15).bit_length())
    runner, tokenizer = build_dflash(mesh_device, max_seq_len)
    prompt_ids = book_prompt(tokenizer, isl)
    try:
        toks, accept, rounds_s = run_dflash(runner, prompt_ids, NEW_TOKENS)
    finally:
        runner.release_traces()
    n = len(toks)
    round_ms = rounds_s / len(accept) * 1e3
    logger.info(f"Output: {tokenizer.decode(toks)!r}")
    logger.info(
        f"DFlash ISL {isl}: {n} tokens, {sum(accept) / len(accept):.2f} tokens per round ({len(accept)} rounds), "
        f"{round_ms:.3f} ms per round, {n / rounds_s:.1f} tok/s/user"
    )
    # perf regression guard: the round time moves little with the answer text (at most 2.4% over 5 prompt dates), the
    # acceptance moves a lot
    assert round_ms <= MAX_ROUND_MS[isl], f"{round_ms:.3f} ms per round > {MAX_ROUND_MS[isl]} ms at ISL {isl}"
