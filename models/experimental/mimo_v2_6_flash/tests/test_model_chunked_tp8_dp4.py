# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill and prefix-cache hits on the full 48-layer model (TP=8 x DP=4).

The same ~1.5K-token prompt is prefilled three ways on every replica, each into its own decode slot:

* slot 0: one shot;
* slot 1: in chunks (row r uses ``CHUNKS[r]`` tokens per chunk);
* slot 2: a prefix-cache hit: its first ``PREFIX`` tokens are prefilled on their own, then the rest
  continues from ``start_pos = PREFIX``, reading the prefix from the cache.

All three must give near-identical next-token logits and pick the same next token up to a bf16 near-tie
(a token the other run scores within ``TIE_STEPS`` bf16 steps of its best: this prompt's top two are one
step apart). Then every slot decodes greedily from the one-shot slot's first token for a few steps, which
reads the KV the chunks wrote; the generated text is logged and the first ``AGREE_TOKENS`` tokens must agree
with the one-shot slot's up to their first difference, which passes only as the same kind of near-tie (each
slot's pick near the top of the other's logits there; common.first_divergence). The histories fork there, so
later tokens are not compared: the chunked slots' logits differ from one shot's by bf16 noise (PCC ~0.998), and
the prompt's continuation has decode steps whose top two are under a step apart. Later chunks of the SWA layers
read only the window before them from the cache; those of the global layers read the whole context through
chunked MLA prefill on the chip that owns the slot.

``pair_split``: the global layers' KV pair split (TtMiMoModel(kv_pair_split=True), vLLM's default layout): each chip
of a pair keeps half of every 128-token block, a later chunk runs pair-split chunked MLA prefill on both chips and
merges by LSE, and decode attends each chip's half and merges (tt/pair_split.py).
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
    first_divergence,
    load_config_and_ckpt,
    model_path,
    near_top,
    pcc,
    tie_margin,
)
from models.experimental.mimo_v2_6_flash.tt.attention import PAIR_SPLIT_OPS
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

PROMPT_TOKENS = 1500
PREFIX = 512
CHUNKS = (128, 256, 512, 1024)  # chunk size on replica rows 0..3
MAX_PREFILL_CHUNK = 2048  # one-shot prefill of the whole prompt must fit in one chunk
GREEDY_TOKENS = 8
AGREE_TOKENS = 4


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
@pytest.mark.parametrize("kv_pair_split", [False, True], ids=["users", "pair_split"])
def test_model_chunked_tp8_dp4(mesh_device, kv_pair_split):
    if kv_pair_split and not PAIR_SPLIT_OPS:
        pytest.skip("needs the MLA attention ops with return_lse (the KV pair split)")
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
        kv_pair_split=kv_pair_split,
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
                f"row {r} {name}: next-token logits PCC vs one shot {p:.5f}, same top-1 up to a tie {same}; top-2 "
                + " vs ".join(
                    f"{tok.decode([int(i)])!r} {float(v):.3f}" for t in top for i, v in zip(t.indices, t.values)
                )
            )

    # ---- greedy decode from all three slots on every row, all from the one-shot slot's first token
    users = {r * 4 + s: s for r in range(DP) for s in range(3)}
    gen_tokens = {u: [int(first[(gen.locate(u)[0], 0)].argmax())] for u in users}
    gen_logits = {u: [first[(gen.locate(u)[0], s)]] for u, s in users.items()}  # gen_logits[u][k] picked token k
    positions = {u: n for u in users}
    for k in range(1, GREEDY_TOKENS):
        out = gen.decode({u: t[-1] for u, t in gen_tokens.items()}, positions)
        for u, lg in out.items():
            gen_tokens[u].append(int(lg.argmax()))
            if k < AGREE_TOKENS:
                gen_logits[u].append(lg)
            positions[u] += 1
    agree_decode = True
    for r in range(DP):
        texts = [gen_tokens[r * 4 + s] for s in range(3)]
        logger.info(f"row {r}: " + " | ".join(repr(tok.decode(t)) for t in texts))
        ref = r * 4
        for slot, name in ((1, f"chunks of {CHUNKS[r]}"), (2, f"prefix hit at {PREFIX}")):
            u = ref + slot
            fork = first_divergence(
                gen_tokens[u][:AGREE_TOKENS], gen_logits[u], gen_tokens[ref][:AGREE_TOKENS], gen_logits[ref]
            )
            if fork is None:
                continue
            k, tie = fork
            agree_decode = agree_decode and tie
            picks = [(gen_tokens[v][k], gen_logits[v][k], gen_logits[w][k]) for v, w in ((ref, u), (u, ref))]
            logger.info(
                f"row {r} {name}: greedy decode differs from one shot at step {k}, near-tie {tie}; "
                + " vs ".join(
                    f"{who} picks {tok.decode([t])!r} ({float(own[t]):.4f}), "
                    f"which the other scores {float(other[t]):.4f}, "
                    f"{tie_margin(other, t):.2f} bf16 steps below its best {float(other.max()):.4f}"
                    for who, (t, own, other) in zip(("one shot", name), picks)
                )
            )

    assert agree, "chunked or prefix-hit prefill picked a different next token than one-shot prefill"
    assert worst_pcc > 0.99, worst_pcc
    assert agree_decode, f"greedy decode diverged beyond a bf16 near-tie within {AGREE_TOKENS} tokens"
