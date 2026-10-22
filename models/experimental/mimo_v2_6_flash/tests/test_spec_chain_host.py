# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP draft chained behind a verify (reference/spec_chain.py, the integer mirror of kernels/spec_chain.cpp); no
device. The chain step must write exactly the upload the host built after reading the verify (pack_mtp(plan_mtp(...))
from the plugin's committed tokens and positions), for every accepted count, around the MTP ring's wrap and at the end
of the context, and decline where propose_draft_tokens declines; the drafts step must pick the anchor's tokens; the
result page must round-trip, and a ChainedDraft must match exactly the commit it was drafted from. End to end with the
accept kernel's mirror (reference/sample_verify.py) on greedy requests: the committed tokens are the plugin's greedy
walk over the argmax ids. test_spec_inputs_tp8.py runs the kernel against the same mirror on a Galaxy row."""

import random

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.sample_verify import (
    GROUPS,
    NO_TOKEN,
    OUT_WORDS,
    control_words,
    pass_inputs,
    verify_pass,
)
from models.experimental.mimo_v2_6_flash.reference.spec_chain import (
    ACCEPT_AT,
    ACCEPT_WORDS,
    ARGMAX_AT,
    CHAIN_WORDS,
    DRAFTS_AT,
    HEADER_WORDS,
    NO_SLOT,
    RESULT_WORDS,
    STATUS_AT,
    chain_row,
    chain_step,
    chained_draft,
    drafts_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    MAX_DRAFTS,
    SPEC_SLOTS,
    accept_greedy,
    draft_source,
    plan_mtp,
    plan_verify_passes,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import (
    MTP_ROWS,
    derive_verify,
    owned_slot_caps,
    owned_slot_columns,
    pack_mtp,
    pack_verify,
    slot_caps,
)

B = 32
VOCAB = 152576
MAX_SEQ_LEN = 1 << 20
# Bonus positions around the ring's wrap (MTP_RING_TOKENS = 256) and the start of the context.
BONUS = [1, 2, 3, 4, 254, 255, 256, 257, 258, 511, 512, 513, 4096]


def one_row_pass(p, n, rng):
    """A verify pass of one request: its last committed token and n drafts at positions p .. p + n (slots 0 .. n)."""
    tokens = [rng.randrange(VOCAB) for _ in range(n + 1)]
    (vp,) = plan_verify_passes([(5, tokens, list(range(p, p + n + 1)), n)])
    return vp


def upload(vp, chain=None):
    """The verify upload's values (int32 list), as _enqueue_verify_pass packs a pass into the first slots."""
    tok = torch.zeros(B, dtype=torch.int32)
    pos = torch.full((B,), -1, dtype=torch.int32)
    tok[:SPEC_SLOTS] = torch.tensor(vp.tokens, dtype=torch.int32)
    pos[:SPEC_SLOTS] = torch.tensor(vp.positions, dtype=torch.int32)
    return pack_verify(tok, pos, None if chain is None else torch.tensor(chain, dtype=torch.int32)).tolist()


def accept_page(groups):
    """SampleVerify's output page for [(count, committed tokens)] per group (unused groups count 0)."""
    words = []
    for g in range(GROUPS):
        count, tokens = groups[g] if g < len(groups) else (0, [])
        out = [count] + [t & 0xFFFFFFFF for t in tokens] + [NO_TOKEN] * (1 + MAX_DRAFTS - len(tokens))
        words += out + [0] * (OUT_WORDS - len(out))
    return words


def host_upload(source, committed, positions, m):
    """What propose_draft_tokens uploaded for the plugin's commit of m tokens (pack_mtp(plan_mtp(...)))."""
    plan = plan_mtp(source, positions, m)
    slot_tokens = [0] * SPEC_SLOTS
    for c in range(m):
        slot_tokens[source.slots[c]] = committed[c]
    return pack_mtp(B, slot_tokens, plan).tolist(), plan


def committed_positions(vp, m):
    """The plugin's committed positions of a verify row (model_runner._committed_positions): column 0's + 1 onward."""
    return [vp.positions[0] + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)


# ---------------------------------------------------------------------- layouts
def test_chain_row():
    assert chain_row(B) == [0] * B
    vp = one_row_pass(40, 2, random.Random(0))
    row = chain_row(B, draft_source([vp]), 1000)
    assert row[:CHAIN_WORDS] == [1, 3, 1000, 0, 1, 2, -1] and row[CHAIN_WORDS:] == [0] * (B - CHAIN_WORDS)


def test_the_chain_row_leaves_the_verify_inputs_alone():
    g = torch.Generator().manual_seed(0)
    tok = torch.randint(0, VOCAB, (B,), generator=g, dtype=torch.int32)
    pos = torch.full((B,), -1, dtype=torch.int32)
    pos[:4] = torch.arange(700, 704, dtype=torch.int32)
    chain = torch.tensor(chain_row(B, draft_source([one_row_pass(700, 3, random.Random(1))]), 5000))
    caps, own_cap, own_col = slot_caps(B), owned_slot_caps(B, 8, 2), owned_slot_columns(B, 8, 2)
    for chip in range(8):
        a = derive_verify(pack_verify(tok, pos), own_cap[chip], own_col[chip], caps, True)
        b = derive_verify(pack_verify(tok, pos, chain), own_cap[chip], own_col[chip], caps, True)
        for name in a:
            pairs = zip(a[name], b[name]) if name == "serial" else [(a[name], b[name])]
            assert all(torch.equal(x, y) for x, y in pairs), name


# ---------------------------------------------------------------------- the chain step
@pytest.mark.parametrize("bonus", BONUS + [MAX_SEQ_LEN - 1 - MAX_DRAFTS])
def test_chain_writes_the_host_planned_upload(bonus):
    """Every draft count n and accepted count m: the chain step's upload is pack_mtp(plan_mtp(...)) of the plugin's
    commit, the header carries the accept page, the argmax ids and the anchor."""
    rng = random.Random(bonus)
    limit = MAX_SEQ_LEN - MAX_DRAFTS
    for n in range(MAX_DRAFTS + 1):
        for m in range(1, n + 2):
            p = bonus - m
            if p < 0:
                continue
            vp = one_row_pass(p, n, rng)
            source = draft_source([vp])
            committed = [rng.randrange(VOCAB) for _ in range(m)]
            accept = accept_page([(m, committed)])
            argmax = [rng.randrange(VOCAB) for _ in range(B)]
            mtp, header = chain_step(upload(vp, chain_row(B, source, limit)), accept, argmax)
            want, plan = host_upload(source, committed, committed_positions(vp, m), m)
            assert mtp == want, (n, m)
            assert header[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS] == accept
            assert header[ARGMAX_AT : ARGMAX_AT + SPEC_SLOTS] == argmax[:SPEC_SLOTS]
            assert header[STATUS_AT : STATUS_AT + 4] == [1, plan.anchor, bonus, 0]
            assert len(header) == HEADER_WORDS


def test_an_idle_chain_row_writes_an_idle_draft():
    rng = random.Random(3)
    vp = one_row_pass(300, 3, rng)
    accept = accept_page([(2, [7, 8])])
    mtp, header = chain_step(upload(vp), accept, list(range(B)))
    assert mtp == pack_mtp(B).tolist()
    assert header[STATUS_AT : STATUS_AT + 4] == [0, NO_SLOT, 0, 0]
    assert header[ACCEPT_AT : ACCEPT_AT + ACCEPT_WORDS] == accept and header[ARGMAX_AT : ARGMAX_AT + 4] == [0, 1, 2, 3]


@pytest.mark.parametrize("k", range(1, MAX_DRAFTS + 1))
def test_the_chain_declines_where_propose_does(k):
    """A verify of width 1 + k: the chain drafts while bonus + k < max_seq_len (the limit max_seq_len - k), what
    propose_draft_tokens allows for num_drafts k, and every RoPE index it writes stays inside the tables."""
    seq = 4096
    rng = random.Random(k)
    for bonus in range(seq - k - 6, seq - k + 1):
        for m in (1, 2):
            vp = one_row_pass(bonus - m, MAX_DRAFTS, rng)
            accept = accept_page([(m, [11] * m)])
            mtp, header = chain_step(upload(vp, chain_row(B, draft_source([vp]), seq - k)), accept, [0] * B)
            drafts = bonus + k < seq
            assert header[STATUS_AT] == int(drafts), (bonus, m)
            assert mtp == (
                host_upload(draft_source([vp]), [11] * m, committed_positions(vp, m), m)[0]
                if drafts
                else pack_mtp(B).tolist()
            )
            assert max(mtp[4 * B : MTP_ROWS * B]) < seq


def test_a_count_past_the_source_columns_drafts_nothing():
    vp = one_row_pass(50, 1, random.Random(4))
    chain = chain_row(B, draft_source([vp]), 1000)
    for m in (0, 3):
        mtp, header = chain_step(upload(vp, chain), accept_page([(m, [1] * m)]), [0] * B)
        assert mtp == pack_mtp(B).tolist() and header[STATUS_AT] == 0


# ---------------------------------------------------------------------- the drafts step and the result page
def test_drafts_step_reads_the_anchor():
    rng = random.Random(5)
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    for anchor in range(SPEC_SLOTS):
        vp = one_row_pass(100, 3, rng)
        mtp, _ = host_upload(
            draft_source([vp]), [1, 2, 3, 4][: anchor + 1], committed_positions(vp, anchor + 1), anchor + 1
        )
        out = drafts_step(mtp, heads)
        assert out[0] == anchor and out[1 : 1 + MAX_DRAFTS] == [h[anchor] for h in heads]
        assert len(out) == RESULT_WORDS - DRAFTS_AT
    idle = drafts_step(pack_mtp(B).tolist(), heads)
    assert idle[0] == NO_SLOT and idle[1 : 1 + MAX_DRAFTS] == [NO_TOKEN] * MAX_DRAFTS


def chain_pass(vp, accept, argmax, heads, limit=MAX_SEQ_LEN - MAX_DRAFTS, source=True):
    """The result page after the accept trace's chain step and the MTP trace's drafts step."""
    chain = chain_row(B, draft_source([vp]), limit) if source else None
    mtp, header = chain_step(upload(vp, chain), accept, argmax)
    return header + drafts_step(mtp, heads)


def test_the_result_page_round_trips_into_a_chained_draft():
    rng = random.Random(6)
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    vp = one_row_pass(1000, 3, rng)
    committed = [rng.randrange(VOCAB) for _ in range(2)]
    argmax = [rng.randrange(VOCAB) for _ in range(B)]
    result = unpack_result(chain_pass(vp, accept_page([(2, committed)]), argmax, heads))
    assert result.accept[0] == (2, committed, False) and result.argmax == argmax[:SPEC_SLOTS]
    assert result.drafting and result.anchor == result.draft_anchor == 1 and result.bonus == 1002
    d = chained_draft(result, draft_source([vp]), vp.positions)
    assert d.row == 5 and d.tokens == committed and d.positions == [1001, 1002]
    assert d.drafts == [h[1] for h in heads]
    # propose_draft_tokens' row: the committed block padded with -1, the plugin's positions.
    tokens, positions = committed + [-1, -1], committed_positions(vp, 2)
    assert d.matches(tokens, positions, 2)
    assert not d.matches(tokens, positions, 1)
    assert not d.matches([committed[0], committed[1] + 1, -1, -1], positions, 2)
    assert not d.matches(tokens, [p + 1 for p in positions], 2)


def test_no_chained_draft_without_a_source_or_past_the_limit():
    rng = random.Random(7)
    heads = [[0] * B for _ in range(MAX_DRAFTS)]
    vp = one_row_pass(2000, 3, rng)
    accept = accept_page([(4, [1, 2, 3, 4])])
    for words in (
        chain_pass(vp, accept, [0] * B, heads, source=False),
        chain_pass(vp, accept, [0] * B, heads, limit=2004),
    ):
        result = unpack_result(words)
        assert not result.drafting and result.draft_anchor is None
        assert chained_draft(result, draft_source([vp]), vp.positions) is None
    assert unpack_result(chain_pass(vp, accept, [0] * B, heads, limit=2005)).drafting


# ---------------------------------------------------------------------- with the accept kernel's mirror
@pytest.mark.parametrize("seed", range(40))
def test_greedy_accept_then_chain(seed):
    """An argmax_ids step through the device path: SampleVerify's greedy walk over the verify's argmax ids commits
    what the plugin's greedy walk commits (the cells argmax_ids answers), and the chain drafts from that commit as the
    host would have after reading it."""
    rng = random.Random(seed)
    n = rng.randrange(MAX_DRAFTS + 1)
    p = rng.choice([0, 1, 255, 256, 4095, rng.randrange(MAX_SEQ_LEN - 8)])
    vp = one_row_pass(p, n, rng)
    drafts = vp.tokens[1 : n + 1]
    # The target agrees with a prefix of the drafts.
    agree = rng.randrange(n + 1)
    argmax = [rng.randrange(VOCAB) for _ in range(B)]
    for j in range(n):
        argmax[j] = drafts[j] if j < agree else (drafts[j] + 1) % VOCAB
    slots, groups = pass_inputs(vp, None)
    assert slots == {}
    accept = verify_pass(None, None, argmax, control_words(groups), VOCAB // 8)
    want = accept_greedy(argmax, drafts, n)
    m = len(want)
    assert accept[0] == m and accept[1 : 1 + m] == want
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    result = unpack_result(chain_pass(vp, accept, argmax, heads))
    # The argmax_ids answer reads the page's argmax ids; the walk over them is what the device committed.
    ids = [result.argmax[s] for s, _, _ in vp.answers]
    assert accept_greedy(ids, drafts, n) == result.accept[0][1]
    d = chained_draft(result, draft_source([vp]), vp.positions)
    positions = committed_positions(vp, m)
    assert d.matches(want + [-1] * (1 + MAX_DRAFTS - m), positions, m)
    mtp, header = chain_step(upload(vp, chain_row(B, draft_source([vp]), MAX_SEQ_LEN - MAX_DRAFTS)), accept, argmax)
    assert mtp == host_upload(draft_source([vp]), want, positions, m)[0]
    assert d.drafts == [h[m - 1] for h in heads]


def test_a_pass_of_two_requests_chains_nothing():
    (vp,) = plan_verify_passes([(0, [1, 2], [5, 6], 1), (1, [3], [9], 0)])
    assert draft_source([vp]) is None
    argmax = list(range(100, 100 + B))
    slots, groups = pass_inputs(vp, None)
    accept = verify_pass(None, None, argmax, control_words(groups), VOCAB // 8)
    result = unpack_result(chain_pass(vp, accept, argmax, [[0] * B] * MAX_DRAFTS, source=False))
    assert [g[0] for g in result.accept] == [1, 1, 0, 0] and result.argmax == argmax[:SPEC_SLOTS]
    assert not result.drafting
