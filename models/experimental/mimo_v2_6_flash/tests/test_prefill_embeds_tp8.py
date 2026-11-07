# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill from host embeddings (multimodal rows) on the full 48-layer model, one TP=8 replica (row 0 of the Galaxy).

TtMiMoModel.prefill_enqueue(embeds=, embed_rows=) replaces the token embeddings of some rows with host rows (a
multimodal item's embeddings at its placeholder tokens); the merge is an exact copy (TtMiMoModel._merge_embeds). So a
prefill whose host rows are the embedding table's own rows must give exactly what the tokens give:

* one chunk: slot 0 prefills the prompt from its tokens, slot 2 from embeds = embedding[tokens] (every row);
* chunks of 512: slot 1 prefills a prompt whose rows R hold other tokens (``alt``), slot 3 prefills the prompt with
  the placeholder id at R and embeds = embedding[alt] at R only. R covers part of the first chunk and part of the
  last, so the middle chunk has no embedded row and runs the token path.

Both pairs are compared bit for bit: the last-token logits, the K / V every layer cached for every prompt position
(each chip's copy, through the slot's page tables), and two greedy decode steps after the prompt, which read that KV.
Slots 0 / 2 and 1 / 3 have the same parity, so the global layers' chip pair keeps them on the same chip.
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
    cache_dir,
    dp_submeshes,
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


def long_prompt(tok):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "reference", "model.py")) as f:
        text = "Explain what the following Python module does.\n\n" + f.read()
    ids = torch.tensor(tok(text)["input_ids"])
    assert ids.numel() >= PROMPT_TOKENS, ids.numel()
    return ids[:PROMPT_TOKENS]


def slot_kv(model, slot, n):
    """{layer: (positions backed, [K, V] each [tp, positions, dim])}: what ``slot`` cached for positions [0, n),
    read through its page tables on every chip; positions a chip maps to a scratch block are left out."""
    out = {}
    bs = model.block_size
    for i, layer in enumerate(model.layers):
        table, scratch = model._host_table(model._table_key(i))
        blocks = table[:, slot, : math.ceil(n / bs)].to(torch.int64)  # [tp, nb]
        dead = blocks == 0 if layer.is_swa else (blocks == scratch if scratch is not None else blocks < 0)
        keep = (~dead).repeat_interleave(bs, dim=1)[:, :n]  # [tp, n]
        caches = []
        for t in model.kv_caches[i]:
            chips = [ttnn.to_torch(c).float() for c in ttnn.get_device_tensors(t)]  # [num_blocks, 1, bs, dim]
            rows = torch.stack([c[blocks[k], 0].reshape(-1, c.shape[-1])[:n] for k, c in enumerate(chips)])
            caches.append(torch.where(keep[..., None], rows, torch.zeros(())))
        out[i] = (keep, caches)
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


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prefill_embeds_tp8(mesh_device):
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())
    row = dp_submeshes(mesh_device)[0]
    t0 = time.time()
    model = TtMiMoModel(
        row,
        cfg,
        ckpt=ckpt,
        cache_dir=cache_dir("model_bfp4experts"),
        max_seq_len=4096,
        max_prefill_chunk=MAX_PREFILL_CHUNK,
        decode_users=4,
    )
    model.setup_kv_cache(4)
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
    first = {}
    for slot, r in runs.items():
        t1 = time.time()
        handle = model.prefill_enqueue(
            r["tokens"], slot, chunk_size=r["chunk"], embeds=r.get("embeds"), embed_rows=r.get("embed_rows")
        )
        first[slot] = model.prefill_read(handle)[0]
        logger.info(
            f"slot {slot}: prefill of {n} tokens in chunks of {r['chunk']} "
            f"({'embeds' if 'embeds' in r else 'tokens'}) {1e3 * (time.time() - t1):.0f} ms, next token "
            f"{tok.decode([int(first[slot].argmax())])!r}"
        )

    for a, b, what in ((0, 2, "one chunk"), (1, 3, f"chunks of {CHUNK}")):
        diff = float((first[a] - first[b]).abs().max())
        logger.info(f"{what}: last-token logits max |diff| {diff}")
        assert torch.equal(first[a], first[b]), f"{what}: last-token logits differ (max |diff| {diff})"

    kv = {slot: slot_kv(model, slot, n) for slot in runs}
    compare_kv(kv[0], kv[2], "one chunk")
    compare_kv(kv[1], kv[3], f"chunks of {CHUNK}")

    # Decode reads the cached prompt: both slots of a pair continue from the token-path slot's greedy token.
    base = {slot: slot % 2 for slot in runs}  # the token-path slot of each pair
    tokens = torch.zeros(model.max_batch, dtype=torch.int32)
    positions = torch.full((model.max_batch,), -1, dtype=torch.int32)
    for slot in runs:
        tokens[slot] = int(first[base[slot]].argmax())
        positions[slot] = n
    for step in range(DECODE_STEPS):
        logits = model.decode(tokens, positions)
        for a, b in ((0, 2), (1, 3)):
            diff = float((logits[a] - logits[b]).abs().max())
            assert torch.equal(logits[a], logits[b]), f"decode step {step}: slots {a} / {b} differ (max |diff| {diff})"
        logger.info(f"decode step {step}: slots 0 / 2 and 1 / 3 identical")
        for slot in runs:
            tokens[slot] = int(logits[base[slot]].argmax())
        positions[list(runs)] += 1
