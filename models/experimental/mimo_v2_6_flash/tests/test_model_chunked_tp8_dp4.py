# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill and prefix-cache hits on the full 48-layer model (TP=8 x DP=4).

The same ~1.5K-token prompt is prefilled three ways on every replica, each into its own decode slot:

* slot 0: one shot;
* slot 1: in chunks (row r uses ``CHUNKS[r]`` tokens per chunk);
* slot 2: a prefix-cache hit: its first ``PREFIX`` tokens are prefilled on their own, then the rest
  continues from ``start_pos = PREFIX``, reading the prefix from the cache.

All three must give near-identical next-token logits (PCC > 0.99), and each must pick a token among the other's
top ``TOP_K``: this prompt's first token is a near-tie (top two within ~0.4 logit), and equivalent computations
move each logit by up to ~0.6 after 48 layers (the one-shot prefill alone moved 0.375 when the MoE prefill
changed from sparse_matmul to expert-major at layer-level PCC 0.99999). Then every slot decodes greedily from the one-shot slot's first token for a few steps, which
reads the KV the chunks wrote; the generated text is logged and the first tokens must agree. Later chunks of the SWA layers read only the window before them from the cache; those of the
global layers read the whole context through chunked MLA prefill on the chip that owns the slot.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    load_config_and_ckpt,
    model_path,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

PROMPT_TOKENS = 1500
PREFIX = 512
CHUNKS = (128, 256, 512, 1024)  # chunk size on replica rows 0..3
MAX_PREFILL_CHUNK = 2048  # one-shot prefill of the whole prompt must fit in one chunk
GREEDY_TOKENS = 8
AGREE_TOKENS = 4
TOP_K = 3


def near_top(logits, token):
    """``token`` is among the TOP_K highest logits."""
    return token in logits.topk(TOP_K).indices.tolist()


def long_prompt(tok):
    """Deterministic natural-ish text: this model's own reference implementation, as a code prompt."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "reference", "model.py")) as f:
        text = "Explain what the following Python module does.\n\n" + f.read()
    ids = torch.tensor(tok(text)["input_ids"])
    assert ids.numel() >= PROMPT_TOKENS, ids.numel()
    return ids[:PROMPT_TOKENS]


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_chunked_tp8_dp4(mesh_device):
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())
    gen = MiMoGalaxyGenerator(
        mesh_device,
        cfg,
        ckpt,
        cache_dir("model_bfp4experts"),
        dp=DP,
        users_per_replica=4,
        max_seq_len=4096,
        max_prefill_chunk=MAX_PREFILL_CHUNK,
    )
    ids = long_prompt(tok)
    n = ids.numel()

    # ---- prefill: three ways per row, all rows enqueued before anything is read
    pending = []
    for r, model in enumerate(gen.models):
        pending.append((r, 0, model.prefill_enqueue(ids, 0, chunk_size=MAX_PREFILL_CHUNK)))
        pending.append((r, 1, model.prefill_enqueue(ids, 1, chunk_size=CHUNKS[r])))
        model.prefill_read(model.prefill_enqueue(ids[:PREFIX], 2))  # the earlier request that filled the prefix
        pending.append((r, 2, model.prefill_enqueue(ids[PREFIX:], 2, start_pos=PREFIX)))
    first = {}
    for r, slot, handle in pending:
        first[(r, slot)] = gen.models[r].prefill_read(handle)[0]

    worst_pcc, agree = 1.0, True
    for r in range(DP):
        base = first[(r, 0)]
        for slot, name in ((1, f"chunks of {CHUNKS[r]}"), (2, f"prefix hit at {PREFIX}")):
            p = pcc(first[(r, slot)], base)
            other = first[(r, slot)]
            same = near_top(base, int(other.argmax())) and near_top(other, int(base.argmax()))
            worst_pcc, agree = min(worst_pcc, p), agree and same
            top = [first[(r, k)].topk(2) for k in (0, slot)]
            logger.info(
                f"row {r} {name}: next-token logits PCC vs one shot {p:.5f}, picks in each other's top-{TOP_K} {same}; top-2 "
                + " vs ".join(
                    f"{tok.decode([int(i)])!r} {float(v):.3f}" for t in top for i, v in zip(t.indices, t.values)
                )
            )

    # ---- greedy decode from all three slots on every row, all from the one-shot slot's first token
    users = {r * 4 + s: s for r in range(DP) for s in range(3)}
    gen_tokens = {u: [int(first[(gen.locate(u)[0], 0)].argmax())] for u in users}
    positions = {u: n for u in users}
    for _ in range(GREEDY_TOKENS - 1):
        out = gen.decode({u: t[-1] for u, t in gen_tokens.items()}, positions)
        for u, lg in out.items():
            gen_tokens[u].append(int(lg.argmax()))
            positions[u] += 1
    agree_decode = True
    for r in range(DP):
        texts = [gen_tokens[r * 4 + s] for s in range(3)]
        logger.info(f"row {r}: " + " | ".join(repr(tok.decode(t)) for t in texts))
        agree_decode = agree_decode and all(t[:AGREE_TOKENS] == texts[0][:AGREE_TOKENS] for t in texts)

    assert agree, "chunked or prefix-hit prefill picked a different next token than one-shot prefill"
    assert worst_pcc > 0.99, worst_pcc
    assert agree_decode, f"greedy decode diverged within {AGREE_TOKENS} tokens"
