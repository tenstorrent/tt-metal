# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch > 1 full-model tests for GLM-4.7-Flash on one Blackhole p150-class chip.

Runs in its own pytest session: the 47-layer weights are 17.4 GiB, so only one
full model fits on the card at a time and this suite builds a batch-B model
instead of the batch-1 one in ``test_full_model.py``.

    GLM47_FM_BATCH=32 GLM47_FM_BATCH_SEQ=8192 pytest \\
        models/autoports/zai_org_glm_4_7_flash/tests/test_full_model_batch.py -x -q -s

Batch and per-user context trade against each other in DRAM: the paged latent
cache costs 612 B per token per layer x 47 layers = 28.8 KiB per token, so
``batch x context`` is bounded by what is left after the 17.4 GiB of weights.
The defaults (32 users x 8192 tokens = 7.5 GiB) sit inside that budget; batch 1
keeps the full advertised 202752 context (see ``doc/context_contract.json``).

At batch > 1 the MoE decode path switches from the batch-1 indexed
``sparse_matmul`` to the union-sparsity walk, exactly as in the optimized
decoder stage - that is the decoder's own contract, not a full-model change.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator

MODEL_DIR = Path(__file__).resolve().parents[1]
BATCH = int(os.environ.get("GLM47_FM_BATCH", "32"))
BATCH_SEQ = int(os.environ.get("GLM47_FM_BATCH_SEQ", "8192"))
#: Printed so the committed log records the geometry *and* whether it came from
#: the environment or the defaults; the two are indistinguishable otherwise.
BATCH_SOURCE = {
    "GLM47_FM_BATCH": os.environ.get("GLM47_FM_BATCH", "<default 32>"),
    "GLM47_FM_BATCH_SEQ": os.environ.get("GLM47_FM_BATCH_SEQ", "<default 8192>"),
}
TRACE_REGION_SIZE = 350_000_000


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=TRACE_REGION_SIZE
    )
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def gen(device):
    generator = build_generator(
        MODEL_DIR,
        device,
        max_batch_size=BATCH,
        max_seq_len=BATCH_SEQ,
        warmup_prefill_lens=(128, 256),
    )
    yield generator
    generator.teardown()


def test_batch_geometry_is_recorded(capsys=None):
    """One line in the log saying where BATCH and BATCH_SEQ came from."""
    print(f"batch geometry: BATCH={BATCH} BATCH_SEQ={BATCH_SEQ} from {BATCH_SOURCE}", flush=True)
    assert BATCH >= 1 and BATCH_SEQ >= 1


def _prompt_ids(gen, seq, salt=0):
    text = (
        f"Prompt {salt}. Tenstorrent builds AI accelerators. "
        "This paragraph exists so the tokenizer produces an ordinary in-distribution prompt "
        "for the batched full-model tests. "
    ) * 60
    ids = gen.tokenizer.encode(text, add_special_tokens=True)
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def test_batch_capacity_contract(gen):
    model = gen.model
    assert model.max_batch_size == BATCH
    weights = model.weight_bytes()["total"]
    cache = model.kv_cache_bytes()
    print(
        f"batch {BATCH} x context {model.max_seq_len}: weights {weights / 2**30:.3f} GiB "
        f"+ cache {cache / 2**30:.3f} GiB = {(weights + cache) / 2**30:.3f} GiB"
    )
    assert weights + cache < int(31.5 * 2**30)


def test_batch_prefill_mixed_lengths_and_decode(gen):
    """Mixed prompt lengths in fixed slots, per-user page-table rows, per-user
    positions, and token feedback across all slots.

    Each user gets its own prompt length (deliberately non-aligned), its own
    page-table row, and its own decode position; the batched decode step must
    keep them independent.
    """
    model = gen.model
    kv_cache = gen._kv_cache
    page_table = gen._page_table_torch
    gen.reset()

    lengths = [33 + 17 * i for i in range(BATCH)]
    max_len = max(lengths)
    tokens = torch.zeros(BATCH, max_len, dtype=torch.int32)
    prompts = []
    for user, plen in enumerate(lengths):
        ids = _prompt_ids(gen, plen, salt=user)
        prompts.append(ids)
        tokens[user, :plen] = torch.tensor(ids, dtype=torch.int32)

    logits = gen.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=lengths)
    assert logits.shape == (BATCH, 1, model.vocab_size)
    assert torch.isfinite(logits).all()
    first = torch.argmax(logits[:, 0], dim=-1)

    # every slot must decode from its own position with its own token
    step_logits = gen.decode_forward(
        first.reshape(BATCH, 1),
        torch.tensor(lengths, dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
    )
    assert step_logits.shape == (BATCH, model.vocab_size)
    assert torch.isfinite(step_logits).all()
    second = torch.argmax(step_logits, dim=-1)
    print("first tokens :", first.tolist())
    print("second tokens:", second.tolist())
    # distinct prompts must not collapse to one answer
    assert len(set(first.tolist())) > 1 or len(set(second.tolist())) > 1


def test_batch_slot_isolation_matches_single_user(gen):
    """A user placed in slot k with the batched path predicts the same first
    token as the same prompt run alone in slot 0."""
    model = gen.model
    kv_cache = gen._kv_cache
    page_table = gen._page_table_torch
    slot = min(BATCH - 1, 7)
    seq = 96
    ids = _prompt_ids(gen, seq, salt=slot)

    gen.reset()
    solo = gen.prefill_forward(
        torch.tensor([ids], dtype=torch.int32), page_table=page_table, kv_cache=kv_cache, prompt_lens=[seq]
    )
    solo_token = int(solo[0, 0].argmax())

    gen.reset()
    lengths = [seq] * BATCH
    tokens = torch.zeros(BATCH, seq, dtype=torch.int32)
    for user in range(BATCH):
        tokens[user] = torch.tensor(_prompt_ids(gen, seq, salt=user), dtype=torch.int32)
    tokens[slot] = torch.tensor(ids, dtype=torch.int32)
    batched = gen.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=lengths)
    assert int(batched[slot, 0].argmax()) == solo_token


def test_batch_inactive_rows(gen):
    """Inactive slots carry position -1 through the *traced* loop:
    ``ttnn.plus_one(skip_negative_entries=True)`` leaves the position alone,
    and because the RoPE index is derived from it on device rather than being
    incremented separately, an inactive slot stays pinned at RoPE index 0
    instead of walking off the end of the cos/sin table."""
    if BATCH < 2:
        pytest.skip("needs batch > 1")
    model = gen.model
    kv_cache = gen._kv_cache
    page_table = gen._page_table_torch
    gen.reset()
    seq = 64
    active = [0, 1]
    inactive = [u for u in range(BATCH) if u not in active]
    tokens = torch.zeros(BATCH, seq, dtype=torch.int32)
    for user in active:
        tokens[user] = torch.tensor(_prompt_ids(gen, seq, salt=user), dtype=torch.int32)
    logits = gen.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=[seq] * BATCH)
    first = torch.argmax(logits[:, 0], dim=-1)

    positions = torch.full((BATCH,), -1, dtype=torch.int32)
    for user in active:
        positions[user] = seq
    gen.set_decode_tokens(first.tolist())
    gen.set_decode_positions(positions)

    for step in range(3):
        rot = ttnn.to_torch(model.decode_rope_indices(gen._pos_dev, BATCH)).reshape(-1).tolist()
        assert all(int(rot[u]) == 0 for u in inactive), (step, rot)
        assert all(int(rot[u]) == seq + step for u in active), (step, rot)
        gen.decode_step_traced()
        gen.read_decode_tokens(BATCH)
    pos_after = ttnn.to_torch(gen._pos_dev).reshape(-1)
    assert all(int(pos_after[u]) == -1 for u in inactive), pos_after.tolist()
    assert all(int(pos_after[u]) == seq + 3 for u in active), pos_after.tolist()
    assert gen.counters["position_refreshes"] >= 1


def test_recapture_mid_decode_leaves_a_deeper_slot_untouched(gen):
    """A trace recapture in the middle of a slot's decode must not disturb it.

    The recapture exists because a prefill can compile programs while the
    traces are live (work log FM-016). Its construction-time sibling warms by
    running one eager decode, which *writes a KV row for every slot*, so a
    recapture that warmed would corrupt any slot sitting at a different
    position: at batch > 1 with mixed prompts that is the normal case, not the
    corner (FM-017). The recapture therefore does not warm, and this is the
    control: identical tokens with and without one injected mid-decode.
    """
    if BATCH < 2:
        pytest.skip("needs batch > 1")
    kv_cache = gen._kv_cache
    seq, steps = 64, 6
    tokens = torch.zeros(BATCH, seq, dtype=torch.int32)
    for user in range(BATCH):
        tokens[user] = torch.tensor(_prompt_ids(gen, seq, salt=user), dtype=torch.int32)

    def run(recapture_after=None):
        gen.reset()
        logits = gen.prefill_forward(
            tokens, page_table=gen._page_table_torch, kv_cache=kv_cache, prompt_lens=[seq] * BATCH
        )
        first = torch.argmax(logits[:, 0], dim=-1).tolist()
        gen.set_decode_tokens(first)
        gen.set_decode_positions([seq] * BATCH)
        out = [first]
        for step in range(steps):
            if step == recapture_after:
                gen.recapture_decode_traces()
            gen.decode_step_traced()
            out.append(gen.read_decode_tokens(BATCH))
        return out

    control = run()
    with_recapture = run(recapture_after=3)
    assert with_recapture == control, (control, with_recapture)


def test_batch_decode_positions_pad_inactive_slots(gen):
    """Fixed slots: a caller may name fewer positions than slots and leave the
    rest inactive, which is only meaningful at batch > 1."""
    if BATCH < 2:
        pytest.skip("needs batch > 1")
    gen.set_decode_positions([7, 9])
    assert gen._host_positions == [7, 9] + [-1] * (BATCH - 2)
    gen.set_decode_positions([5] * BATCH)
    assert gen._host_positions == [5] * BATCH
    gen.reset()
    assert gen._host_positions == [-1] * BATCH


def test_batch_prefill_targets_a_named_slot(gen, expect_error):
    """``prefill_forward(user_ids=...)`` fills the slot the caller names.

    Without it an adapter could only target a slot by handing in a re-rowed
    page table, and a stray `user_id=` kwarg was silently swallowed into
    ``**kwargs`` and prefilled slot 0 (FM-017).
    """
    if BATCH < 3:
        pytest.skip("needs batch > 2")
    kv_cache = gen._kv_cache
    seq = 64
    ids = torch.tensor([_prompt_ids(gen, seq, salt=11)], dtype=torch.int32)
    gen.reset()
    into = BATCH - 1
    logits = gen.prefill_forward(
        ids, page_table=gen._page_table_torch, kv_cache=kv_cache, prompt_lens=[seq], user_ids=[into]
    )
    named = int(torch.argmax(logits[0, 0]))

    # The same prompt in slot `into` of a full batch prefill must agree.
    tokens = torch.zeros(BATCH, seq, dtype=torch.int32)
    for user in range(BATCH):
        tokens[user] = torch.tensor(_prompt_ids(gen, seq, salt=11 if user == into else user), dtype=torch.int32)
    gen.reset()
    full = gen.prefill_forward(tokens, page_table=gen._page_table_torch, kv_cache=kv_cache, prompt_lens=[seq] * BATCH)
    assert named == int(torch.argmax(full[into, 0]))

    with expect_error(ValueError, "user_ids must be in"):
        gen.prefill_forward(
            ids, page_table=gen._page_table_torch, kv_cache=kv_cache, prompt_lens=[seq], user_ids=[BATCH]
        )


def test_batch_traced_decode_feedback(gen):
    """Token feedback and on-device position advance hold for every slot."""
    kv_cache = gen._kv_cache
    gen.reset()
    seq = 64
    tokens = torch.zeros(BATCH, seq, dtype=torch.int32)
    for user in range(BATCH):
        tokens[user] = torch.tensor(_prompt_ids(gen, seq, salt=user), dtype=torch.int32)
    logits = gen.prefill_forward(tokens, page_table=gen._page_table_torch, kv_cache=kv_cache, prompt_lens=[seq] * BATCH)
    first = torch.argmax(logits[:, 0], dim=-1).tolist()
    gen.set_decode_tokens(first)
    gen.set_decode_positions([seq] * BATCH)
    gen.reset_counters()

    expected = first
    for step in range(3):
        got_in = [int(v) for v in ttnn.to_torch(gen._tokens_dev).reshape(-1)[:BATCH].tolist()]
        assert got_in == expected, (step, got_in, expected)
        pos = [int(v) for v in ttnn.to_torch(gen._pos_dev).reshape(-1).tolist()]
        assert pos == [seq + step] * BATCH, (step, pos)
        gen.decode_step_traced()
        expected = gen.read_decode_tokens(BATCH)
    assert gen.counters["token_input_refreshes"] == 0
    assert gen.counters["position_refreshes"] == 0
    assert gen.counters["page_table_refreshes"] == 0
