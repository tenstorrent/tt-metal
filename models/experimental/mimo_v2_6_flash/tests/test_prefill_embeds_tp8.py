# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill from host embeddings (multimodal rows) on the full 48-layer model, one TP=8 replica (a 1x8 row: the row a
row lock exports, or row 0 of the Galaxy), eager and traced.

TtMiMoModel.prefill_enqueue(embeds=, embed_rows=) replaces the token embeddings of some rows with host rows (a
multimodal item's embeddings at its placeholder tokens); the merge is an exact copy (TtMiMoModel._merge_embeds, and in
a traced pass the same gather from persistent inputs, TtMiMoModel.enable_trace_embeds). So a prefill whose host rows
are the embedding table's own rows must give exactly what the tokens give:

* one chunk: a slot prefills the prompt from its tokens, another from embeds = embedding[tokens] (every row);
* chunks of 512: a slot prefills a prompt whose rows R hold other tokens (``alt``), another prefills the prompt with
  the placeholder id at R and embeds = embedding[alt] at R only. R covers part of the first chunk and part of the
  last, so the middle chunk has no embedded row and runs the token path.

Phase A runs both pairs eagerly (slots 0 / 2 and 1 / 3) before any trace exists. Then traced first chunks (start 0) of
1536 and 512 tokens and traced later chunks of 512 are set up, warmed and captured with host rows enabled; phase C
runs the same prompts eagerly again (allow_trace=False) on slots 0-3 and phase B replays them on slots 4-7 (traced
token path on 4 / 5, traced embeds on 6 / 7). Every member of {0, 2, 4, 6} and of {1, 3, 5, 7} must agree bit for
bit: the last-token logits (B against C: both read the slot's persistent logits row once traces are captured), the
K / V every layer cached (A, B and C; each chip's copy, through the slot's page tables) and two greedy decode steps
after the prompt, which read that KV. Even and odd slots keep the global layers' chip parity of their pair. Nothing
may compile after the captures (a program compiled then can sit where a replay writes); run with
TT_METAL_TRACE_ALLOC_TRACKING=1 to also check no buffer allocated after a capture is alive at a replay.

``kv_pair_split``: the global layers' 128-position blocks split over the chip pair (the vLLM serving layout).
"""

import math
import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import (
    GALAXY_DEVICE_PARAMS,
    ROW_MESH,
    cache_dir,
    first_row,
    load_config_and_ckpt,
    model_path,
)
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

PROMPT_TOKENS = 1500
MAX_PREFILL_CHUNK = 2048
CHUNK = 512
PLACEHOLDER = 151655  # image_token_id: the rows' token ids must not matter
EMBED_RUNS = ((100, 400), (1100, 1130))  # R: rows of the first and last chunk of 512
DECODE_STEPS = 2
TRACE_REGION = 512 << 20
# Traced first chunks: the one-chunk prompt (1500 tokens: 1536) and the first chunk of 512; traced later chunks of 512
# (starts 512 and 1024: global SDPA chunk 128).
FIRST_TRACED = (1536, CHUNK)


def long_prompt(tok):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "reference", "model.py")) as f:
        text = "Explain what the following Python module does.\n\n" + f.read()
    ids = torch.tensor(tok(text)["input_ids"])
    assert ids.numel() >= PROMPT_TOKENS, ids.numel()
    return ids[:PROMPT_TOKENS]


def slot_kv(model, slot, n):
    """{layer: (entries backed [tp, entries], [K, V] each [tp, entries, block rows, dim])}: the cache blocks ``slot``
    maps for positions [0, n) through its page tables, on every chip (a pair-split global table maps 2 * block_size
    positions per entry, each chip holding its halves); entries a chip maps to a dead block (0 for the SWA layers'
    released blocks, the scratch block of a global table) are left out (zeroed)."""
    out = {}
    for i, layer in enumerate(model.layers):
        key = model._table_key(i)
        table, scratch = model._host_table(key)
        span = model.block_size * (2 if model._pair_key(key) else 1)
        blocks = table[:, slot, : math.ceil(n / span)].to(torch.int64)  # [tp, entries]
        dead = blocks == 0 if layer.is_swa else (blocks == scratch if scratch is not None else blocks < 0)
        caches = []
        for t in model.kv_caches[i]:
            chips = [ttnn.to_torch(c).float() for c in ttnn.get_device_tensors(t)]  # [num_blocks, 1, rows, dim]
            rows = torch.stack([c[blocks[k], 0] for k, c in enumerate(chips)])  # [tp, entries, rows, dim]
            caches.append(torch.where(dead[..., None, None], torch.zeros(()), rows))
        out[i] = (~dead, caches)
    return out


def compare_kv(a, b, what):
    worst = 0.0
    for i in a:
        (keep_a, kv_a), (keep_b, kv_b) = a[i], b[i]
        assert torch.equal(keep_a, keep_b), f"{what}: layer {i}: different backed positions"
        assert int(keep_a.sum()) > 0, f"{what}: layer {i}: no backed position"
        for name, x, y in zip("KV", kv_a, kv_b):
            diff = float((x - y).abs().max())
            worst = max(worst, diff)
            assert diff == 0, f"{what}: layer {i} {name} cache differs (max |diff| {diff})"
    logger.info(f"{what}: K / V of every layer identical over the backed positions (max |diff| {worst})")


def run_prompt(model, tok, slot, r, n, allow_trace=True):
    t1 = time.time()
    handle = model.prefill_enqueue(
        r["tokens"],
        slot,
        chunk_size=r["chunk"],
        embeds=r.get("embeds"),
        embed_rows=r.get("embed_rows"),
        allow_trace=allow_trace,
    )
    t2 = time.time()
    logits = model.prefill_read(handle)[0]
    logger.info(
        f"slot {slot}: prefill of {n} tokens in chunks of {r['chunk']} ({'embeds' if 'embeds' in r else 'tokens'}, "
        f"{'replayed' if handle.get('persistent') and allow_trace else 'eager'}) enqueue {1e3 * (t2 - t1):.0f} ms, "
        f"total {1e3 * (time.time() - t1):.0f} ms, next token {tok.decode([int(logits.argmax())])!r}"
    )
    return logits


@pytest.mark.parametrize("kv_pair_split", [False, True], ids=["owner_kv", "pair_split"])
@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("mesh_device", [ROW_MESH], indirect=True)
def test_prefill_embeds_tp8(mesh_device, kv_pair_split):
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())
    row = first_row(mesh_device)
    t0 = time.time()
    model = TtMiMoModel(
        row,
        cfg,
        ckpt=ckpt,
        cache_dir=cache_dir("model_bfp4experts"),
        max_seq_len=4096,
        max_prefill_chunk=MAX_PREFILL_CHUNK,
        decode_users=8,
        kv_pair_split=kv_pair_split,
    )
    model.setup_kv_cache(8)
    logger.info(f"model ready in {time.time() - t0:.0f}s")

    embedding = ckpt.embedding().to(torch.bfloat16)
    ids = long_prompt(tok)
    n = ids.numel()
    rows = torch.zeros(n, dtype=torch.bool)
    for a, b in EMBED_RUNS:
        rows[a:b] = True
    alt = ids.clone()
    alt[rows] = ids[(torch.arange(n)[rows] + 37) % n]  # other text at R
    placeholder = alt.clone()
    placeholder[rows] = PLACEHOLDER
    assert (alt[rows] != PLACEHOLDER).all()

    runs = {
        0: dict(tokens=ids, chunk=MAX_PREFILL_CHUNK),
        2: dict(tokens=ids, chunk=MAX_PREFILL_CHUNK, embeds=embedding[ids]),
        1: dict(tokens=alt, chunk=CHUNK),
        3: dict(tokens=placeholder, chunk=CHUNK, embeds=embedding[alt], embed_rows=rows),
    }
    groups = ((0, 2, 4, 6), (1, 3, 5, 7))

    # Phase A: eager, before any trace (this also compiles every program the eager passes run).
    first = {slot: run_prompt(model, tok, slot, r, n) for slot, r in runs.items()}
    for a, b, what in ((0, 2, "one chunk"), (1, 3, f"chunks of {CHUNK}")):
        diff = float((first[a] - first[b]).abs().max())
        logger.info(f"{what}: last-token logits max |diff| {diff}")
        assert torch.equal(first[a], first[b]), f"{what}: last-token logits differ (max |diff| {diff})"
    kv_a = {slot: slot_kv(model, slot, n) for slot in runs}
    compare_kv(kv_a[0], kv_a[2], "eager, one chunk")
    compare_kv(kv_a[1], kv_a[3], f"eager, chunks of {CHUNK}")

    # Traced passes that take host rows: first chunks of FIRST_TRACED, later chunks of 512.
    model.enable_trace_embeds()
    t1 = time.time()
    keys = model.setup_prefill_chunk_traces(FIRST_TRACED, first=True)
    keys += model.setup_prefill_chunk_traces([CHUNK], sdpa_chunks=(128,), late=False)
    model.warm_prefill_chunk_traces()
    used0 = model.trace_region_bytes()[0]
    assert model.capture_prefill_chunk_traces(), "the chunk traces did not fit"
    used, total = model.trace_region_bytes()
    logger.info(
        f"traces {keys} set up, warmed and captured in {time.time() - t1:.0f}s: {(used - used0) / 2**20:.1f} MB of the "
        f"trace region ({used / 2**20:.0f} of {total / 2**20:.0f} MB used per device)"
    )
    programs = row.num_program_cache_entries()

    # Phase C: the same prompts eagerly (slots 0-3); phase B: replayed (slots 4-7).
    eager = {slot: run_prompt(model, tok, slot, r, n, allow_trace=False) for slot, r in runs.items()}
    traced = {}
    for slot, r in runs.items():
        traced[slot + 4] = run_prompt(model, tok, slot + 4, r, n)
    assert row.num_program_cache_entries() == programs, "a prefill compiled programs after the captures"
    # Every chunk of slots 4-7 replayed: the one chunk, and the three chunks of 512 (the last one 476 tokens).
    pieces = [(4, 0, n), (5, 0, CHUNK), (5, CHUNK, CHUNK), (5, 2 * CHUNK, n - 2 * CHUNK)]
    assert all(model._chunk_trace(s, start, ids[:count]) is not None for s, start, count in pieces)
    logits = {**eager, **traced}
    for group in groups:
        for s in group[1:]:
            diff = float((logits[group[0]] - logits[s]).abs().max())
            assert torch.equal(
                logits[group[0]], logits[s]
            ), f"slots {group[0]} / {s}: logits differ (max |diff| {diff})"
        logger.info(
            f"slots {group}: last-token logits identical (eager tokens, eager embeds, traced tokens and embeds)"
        )
    for group in groups:
        assert int(logits[group[0]].argmax()) == int(first[group[0]].argmax()), "the traced pass picks another token"

    kv = {slot: slot_kv(model, slot, n) for slot in range(8)}
    for group in groups:
        for s in group:
            compare_kv(kv_a[group[0]], kv[s], f"slots {group[0]} (phase A) / {s}")

    # Decode reads the cached prompt: every slot of a group continues from the token-path slot's greedy token.
    tokens = torch.zeros(model.max_batch, dtype=torch.int32)
    positions = torch.full((model.max_batch,), -1, dtype=torch.int32)
    for group in groups:
        for s in group:
            tokens[s] = int(logits[group[0]].argmax())
            positions[s] = n
    for step in range(DECODE_STEPS):
        out = model.decode(tokens, positions)
        for group in groups:
            for s in group[1:]:
                diff = float((out[group[0]] - out[s]).abs().max())
                assert torch.equal(out[group[0]], out[s]), f"decode step {step}: slots {group[0]} / {s} differ ({diff})"
            for s in group:
                tokens[s] = int(out[group[0]].argmax())
        logger.info(f"decode step {step}: slots {groups[0]} and {groups[1]} identical")
        positions[:8] += 1
