# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full 48-layer MiMo-V2.6-Flash at long context (32K and 128K tokens) on the Galaxy, TP=8 x DP=4 (#61416).

One run covers steps 3, 4 and 5 of #61416, with 4 users per row (the Phase 0 sizing point):

* **Step 3: same answer for two cuts, and passkey retrieval.** No fp32 golden exists at this length (the
  host would need hours for 48 layers). Each row prefills its own prompt twice: slot 0 in the baseline cut
  (one shot at 32K; 16K-token chunks at 128K, where one shot does not fit), slot 1 in that row's chunk size
  (``CHUNKS``). Both must predict the same next token with near-equal logits and agree for the first
  ``AGREE_TOKENS`` greedy tokens. The prompt is filler text with a 5-digit pass key hidden at a different
  depth on each row (``DEPTHS``); greedy decode from both slots must say the key, which only works if
  attention reaches back to it.
* **Step 5: DP=4.** Slot 2 holds the same prompt on every row, cut the same way. All four rows must give
  the same logits and the same greedy text.
* **Step 4: memory.** DRAM per chip is logged after the weights, after the KV caches for 4 users, and after
  prefill. Then row 0 runs one prefill chunk of each larger size in slot 3, until one does not fit. The SWA
  pool no longer depends on the chunk size (``kv.swa_blocks_per_user``), so only prefill activations grow.
  The largest that ran is logged.

Each case builds its own generator, because ``max_seq_len`` sizes the KV caches. Prefill wall time is
logged; tuning it is #61418.
"""

import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    heartbeat,
    load_config_and_ckpt,
    model_path,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator
from models.experimental.mimo_v2_6_flash.tt.model import TtMiMoModel

USERS = 4  # decode slots per row: 0 baseline cut, 1 the row's chunk size, 2 the shared prompt, 3 the chunk probe
BASE, CHUNKED, SHARED, PROBE = range(USERS)
DEPTHS = (0.10, 0.35, 0.65, 0.90)  # where the pass key sits on rows 0..3
CHUNKS = (1024, 2048, 4096, 8192)  # slot-1 chunk size on rows 0..3
SHARED_CHUNK = 4096
SHARED_DEPTH = 0.50
GREEDY_TOKENS = 12
AGREE_TOKENS = 4
DECODE_ROOM = 512  # positions after the prompt, for decode


@dataclass(frozen=True)
class Case:
    prompt_len: int
    base_chunk: int  # slot 0; also max_prefill_chunk

    @property
    def max_seq_len(self):
        return self.prompt_len + DECODE_ROOM

    @property
    def probe_chunks(self):
        """Chunk sizes above the baseline, up to the whole prompt, for the step-4 probe."""
        sizes, c = [], self.base_chunk * 2
        while c <= self.prompt_len:
            sizes.append(c)
            c *= 2
        return sizes


CASES = {
    "32k": Case(prompt_len=32768, base_chunk=32768),
    "128k": Case(prompt_len=131072, base_chunk=16384),
}

HEAD = (
    "There is an important piece of information hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about it."
)
FILLER = " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
QUESTION = " What is the pass key? The pass key is"


def passkey_prompt(tok, length, depth, key):
    """Token ids of exactly ``length`` tokens: HEAD, filler with the key at ``depth``, QUESTION at the end."""

    def ids(text):
        return tok(text, add_special_tokens=False)["input_ids"]

    head, question = ids(HEAD), ids(QUESTION)
    needle = ids(f" The pass key is {key}. Remember it. {key} is the pass key.")
    n_fill = length - len(head) - len(needle) - len(question)
    filler = ids(FILLER)
    stream = (filler * (n_fill // len(filler) + 1))[:n_fill]
    at = int(depth * n_fill)
    out = head + stream[:at] + needle + stream[at:] + question
    assert len(out) == length
    return torch.tensor(out, dtype=torch.int32)


def dram_gb(device):
    """(allocated, total) DRAM per chip in GB. Every chip of a row holds the same layout."""
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    gb = mv.num_banks / 1e9
    return mv.total_bytes_allocated_per_bank * gb, mv.total_bytes_per_bank * gb


def is_oom(err):
    text = str(err).lower()
    return "out of memory" in text or "oom" in text or "allocat" in text


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("case_name", list(CASES))
def test_model_long_tp8_dp4(mesh_device, case_name, monkeypatch):
    case = CASES[case_name]
    assert all(c < case.base_chunk for c in CHUNKS) and SHARED_CHUNK < case.base_chunk
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())

    # ---- step 4: DRAM after the weights alone (just before the KV caches) and with them
    setup_kv = TtMiMoModel.setup_kv_cache
    after_weights = {}

    def measured_setup(model, num_users):
        after_weights[id(model)] = dram_gb(model.mesh_device)[0]
        return setup_kv(model, num_users)

    monkeypatch.setattr(TtMiMoModel, "setup_kv_cache", measured_setup)
    gen = MiMoGalaxyGenerator(
        mesh_device,
        cfg,
        ckpt,
        cache_dir("model_bfp4experts"),
        dp=DP,
        users_per_replica=USERS,
        max_seq_len=case.max_seq_len,
        max_prefill_chunk=case.base_chunk,
    )
    monkeypatch.setattr(TtMiMoModel, "setup_kv_cache", setup_kv)
    memory = {}
    for r, model in enumerate(gen.models):
        used, total = dram_gb(model.mesh_device)
        memory[r] = dict(weights=after_weights[id(model)], with_kv=used, total=total)

    keys = [int(k) for k in torch.randint(10000, 100000, (DP + 1,), generator=torch.Generator().manual_seed(61416))]
    prompts = [passkey_prompt(tok, case.prompt_len, DEPTHS[r], keys[r]) for r in range(DP)]
    shared_key = keys[DP]
    shared = passkey_prompt(tok, case.prompt_len, SHARED_DEPTH, shared_key)

    # ---- prefill: three prompts per row, everything enqueued before anything is read
    t0 = time.time()
    with heartbeat(f"{case_name} prefill"):
        pending = []
        for r, model in enumerate(gen.models):
            pending.append((r, BASE, model.prefill_enqueue(prompts[r], BASE, chunk_size=case.base_chunk)))
            pending.append((r, CHUNKED, model.prefill_enqueue(prompts[r], CHUNKED, chunk_size=CHUNKS[r])))
            pending.append((r, SHARED, model.prefill_enqueue(shared, SHARED, chunk_size=SHARED_CHUNK)))
        first = {(r, slot): gen.models[r].prefill_read(handle)[0] for r, slot, handle in pending}
    logger.info(f"{case_name}: prefill of 3 x {case.prompt_len} tokens on each of 4 rows in {time.time() - t0:.0f} s")
    for r, model in enumerate(gen.models):
        memory[r]["after_prefill"] = dram_gb(model.mesh_device)[0]

    # ---- step 3: the two cuts agree
    worst_pcc, same_top1 = 1.0, True
    for r in range(DP):
        p = pcc(first[(r, CHUNKED)], first[(r, BASE)])
        same = int(first[(r, CHUNKED)].argmax()) == int(first[(r, BASE)].argmax())
        worst_pcc, same_top1 = min(worst_pcc, p), same_top1 and same
        logger.info(
            f"row {r}: chunks of {CHUNKS[r]} vs {case.base_chunk}: next-token logits PCC {p:.5f}, same top-1 {same}"
        )

    # ---- step 5: the shared prompt gives the same logits on every row
    shared_pcc = [pcc(first[(r, SHARED)], first[(0, SHARED)]) for r in range(DP)]
    logger.info(f"shared prompt: next-token logits PCC vs row 0 per row {[f'{p:.6f}' for p in shared_pcc]}")

    # ---- greedy decode from slots 0-2 on every row
    users = [r * USERS + s for r in range(DP) for s in (BASE, CHUNKED, SHARED)]
    gen_tokens = {u: [int(first[gen.locate(u)].argmax())] for u in users}
    positions = {u: case.prompt_len for u in users}
    with heartbeat(f"{case_name} decode"):
        for _ in range(GREEDY_TOKENS - 1):
            out = gen.decode({u: t[-1] for u, t in gen_tokens.items()}, positions)
            for u, lg in out.items():
                gen_tokens[u].append(int(lg.argmax()))
                positions[u] += 1

    agree, found = True, {}
    for r in range(DP):
        texts = {s: tok.decode(gen_tokens[r * USERS + s]) for s in (BASE, CHUNKED, SHARED)}
        found[(r, BASE)] = str(keys[r]) in texts[BASE]
        found[(r, CHUNKED)] = str(keys[r]) in texts[CHUNKED]
        found[(r, SHARED)] = str(shared_key) in texts[SHARED]
        agree = agree and gen_tokens[r * USERS + BASE][:AGREE_TOKENS] == gen_tokens[r * USERS + CHUNKED][:AGREE_TOKENS]
        logger.info(
            f"row {r} key {keys[r]} at {DEPTHS[r]:.0%}: base {texts[BASE]!r} (found {found[(r, BASE)]}) | "
            f"chunks of {CHUNKS[r]} {texts[CHUNKED]!r} (found {found[(r, CHUNKED)]}) | "
            f"shared key {shared_key}: {texts[SHARED]!r} (found {found[(r, SHARED)]})"
        )
    shared_texts = [gen_tokens[r * USERS + SHARED] for r in range(DP)]

    for r in range(DP):
        m = memory[r]
        logger.info(
            f"{case_name} row {r} DRAM per chip: weights {m['weights']:.2f} GB, + KV for {USERS} users "
            f"{m['with_kv']:.2f} GB, after prefill {m['after_prefill']:.2f} GB, of {m['total']:.2f} GB "
            f"({m['total'] - m['with_kv']:.2f} GB free for prefill scratch)"
        )

    # ---- step 4: the largest prefill chunk that fits (row 0; every row has the same layout)
    model = gen.models[0]
    largest = case.base_chunk
    for chunk in case.probe_chunks:
        try:
            t0 = time.time()
            with heartbeat(f"{case_name} probe {chunk}"):
                model.prefill(shared[:chunk], PROBE, chunk_size=chunk)
            logger.info(f"{case_name} probe: a {chunk}-token chunk fits, in {time.time() - t0:.0f} s")
            largest = chunk
        except RuntimeError as err:
            if not is_oom(err):
                raise
            logger.info(f"{case_name} probe: a {chunk}-token chunk does not fit: {str(err).splitlines()[0][:200]}")
            break
    logger.info(f"{case_name}: largest prefill chunk that fits with KV for {USERS} users: {largest}")

    assert same_top1, "the two cuts predicted different next tokens"
    assert worst_pcc > 0.99, worst_pcc
    assert agree, f"greedy decode of the two cuts diverged within {AGREE_TOKENS} tokens"
    assert all(found.values()), f"pass key not retrieved: {found}"
    assert min(shared_pcc) > 0.9999, shared_pcc
    assert all(t == shared_texts[0] for t in shared_texts), "rows disagree on the shared prompt"
