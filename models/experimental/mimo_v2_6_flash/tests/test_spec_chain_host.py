# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP draft chained behind a verify, and the next verify chained onto it (reference/spec_chain.py, the integer
mirror of kernels/spec_chain.cpp); no device. The chain step must write exactly the upload the host built after
reading the verify (pack_mtp(plan_mtp(...)) from the plugin's committed tokens and positions), for every accepted
count, around the MTP ring's wrap and at the end of the context, and decline where propose_draft_tokens declines; the
drafts step must pick the anchor's tokens; the result page must round-trip, and a ChainedDraft must match exactly the
commit it was drafted from. End to end with the accept kernel's mirror (reference/sample_verify.py) on greedy
requests: the committed tokens are the plugin's greedy walk over the argmax ids. The relay step must write exactly the
verify upload and control page the host would have built after reading the page (the last committed token at the
bonus and the drafts), for every count of the step it chains onto and every number of drafts verified, around the
ring's wrap and at the context limit, from a provisional layout above the real position, leaving every other slot
alone, and idle the pass when the page drafted nothing; with either drafter (one MTP layer, or every layer: the
description, the plan step on the stash, the anchor in the last slot). With DFlash drafting the chain step must write
exactly the DFlash upload the host built from the plugin's commit (generator_vllm._enqueue_dflash), decline where it
declines, and its result page relay and round-trip as the MTP's. SpecChain.warm, at every decode trace capture,
must run its steps on idle uploads, so that the stash a prompt emptied stays empty. test_spec_inputs_tp8.py runs the
kernel against the same mirror on a Galaxy row."""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.mimo_v2_6_flash.reference.mtp import MtpProgramMirror
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
    CHAIN_HSLOTS,
    CHAIN_RELAY,
    CHAIN_WORDS,
    DRAFTS_AT,
    HEADER_WORDS,
    NO_SLOT,
    POSITIONS_AT,
    RESULT_WORDS,
    STATUS_AT,
    DFlashLayout,
    chain_row,
    chain_step,
    chained_draft,
    dflash_chain_step,
    dflash_drafts_step,
    dflash_inputs,
    drafts_step,
    relay_step,
    plan_step,
    stash_page,
    tokens_step,
    unpack_result,
)
from models.experimental.mimo_v2_6_flash.reference.spec_decode import (
    DRAFT_GROUPS,
    MAX_DRAFTS,
    SPEC_SLOTS,
    DraftSource,
    MtpStash,
    accept_greedy,
    draft_source,
    mtp_cache_position,
    mtp_layer_tokens,
    mtp_stash,
    plan_mtp,
    plan_mtp_layers,
    plan_verify_passes,
    serial_positions,
)
from models.experimental.mimo_v2_6_flash.reference.spec_inputs import (
    MTP_ROWS,
    derive_verify,
    owned_slot_caps,
    owned_slot_columns,
    pack_mtp,
    pack_mtp_layers,
    pack_verify,
    slot_caps,
)
from models.experimental.mimo_v2_6_flash.tests.seeded_rng import SeededRng
from models.experimental.mimo_v2_6_flash.tests.test_spec_decode_host import (
    SCHEDULE,
    TinyMiMo,
    drive,
    mtp_drafter,
    tiny_config,
)
from models.experimental.mimo_v2_6_flash.reference import spec_decode as spec_layout

# The MTP drafter's 4-slot verify (the draft groups: test_spec_groups_host.py, test_spec_groups_adapter_host.py).
pytestmark = pytest.mark.skipif(
    (spec_layout.SPEC_SLOTS, spec_layout.DRAFT_GROUPS) != (4, 1),
    reason="MTP's 4-slot layouts: run without MIMO_SPEC_SLOTS / MIMO_SPEC_GROUPS",
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
    vp = one_row_pass(40, 2, SeededRng(0))
    row = chain_row(B, draft_source([vp]), 1000)
    assert row[:CHAIN_WORDS] == [1, 3, 1000, 0, 1, 2, -1, 0] and row[CHAIN_WORDS:] == [0] * (B - CHAIN_WORDS)
    relayed = chain_row(B, draft_source([vp]), 1000, relay=True)
    assert relayed[:CHAIN_WORDS] == [1, 3, 1000, 0, 1, 2, -1, 1]
    # The MTP history's request slots follow the relay word (kernels/mtp_history.cpp reads them there).
    both = chain_row(B, draft_source([vp]), 1000, relay=True, hslots=[2, 2, 2, None])
    assert CHAIN_HSLOTS == CHAIN_RELAY + 1 == CHAIN_WORDS
    assert both[:CHAIN_WORDS] == relayed[:CHAIN_WORDS] and both[CHAIN_WORDS : CHAIN_WORDS + SPEC_SLOTS] == [3, 3, 3, 0]
    assert both[CHAIN_WORDS + SPEC_SLOTS :] == [0] * (B - CHAIN_WORDS - SPEC_SLOTS)
    assert chain_row(B, hslots=[0, 1, None, 3])[:CHAIN_WORDS] == [0] * CHAIN_WORDS


def test_the_chain_row_leaves_the_verify_inputs_alone():
    g = torch.Generator().manual_seed(0)
    tok = torch.randint(0, VOCAB, (B,), generator=g, dtype=torch.int32)
    pos = torch.full((B,), -1, dtype=torch.int32)
    pos[:4] = torch.arange(700, 704, dtype=torch.int32)
    chain = torch.tensor(chain_row(B, draft_source([one_row_pass(700, 3, SeededRng(1))]), 5000))
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
    rng = SeededRng(bonus)
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
    rng = SeededRng(3)
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
    rng = SeededRng(k)
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
    vp = one_row_pass(50, 1, SeededRng(4))
    chain = chain_row(B, draft_source([vp]), 1000)
    for m in (0, 3):
        mtp, header = chain_step(upload(vp, chain), accept_page([(m, [1] * m)]), [0] * B)
        assert mtp == pack_mtp(B).tolist() and header[STATUS_AT] == 0


# ---------------------------------------------------------------------- the drafts step and the result page
def test_drafts_step_reads_the_anchor():
    rng = SeededRng(5)
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


def chain_pass(vp, accept, argmax, heads, limit=MAX_SEQ_LEN - MAX_DRAFTS, source=True, layers=1):
    """The result page after the accept trace's chain step and the MTP trace's drafts step (``layers``: the
    drafter's MTP layers; with every layer the heads' anchor is the last slot)."""
    chain = chain_row(B, draft_source([vp]), limit) if source else None
    mtp, header = chain_step(upload(vp, chain), accept, argmax, layers)
    return header + drafts_step(mtp, heads, layers)


def test_the_result_page_round_trips_into_a_chained_draft():
    rng = SeededRng(6)
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
    rng = SeededRng(7)
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


# ---------------------------------------------------------------------- with DFlash drafting
DFLASH = DFlashLayout(ring=1024, block=8, heads=8, last=MAX_SEQ_LEN - 1, groups=DRAFT_GROUPS)


def dflash_host_upload(source, committed, positions, m):
    """What generator_vllm._enqueue_dflash uploads for the plugin's commit of m tokens: the committed columns' slots as
    context rows at their committed position - 1, the last committed token at its position from slot slots[m - 1]."""
    context = [(source.slots[c], positions[c] - 1) for c in range(m)]
    return dflash_inputs(DFLASH, [(source.group, context, committed[m - 1], positions[m - 1], source.slots[m - 1])])


@pytest.mark.parametrize("bonus", [1, 2, 1023, 1024, 1025, 4096, MAX_SEQ_LEN - 1 - MAX_DRAFTS])
def test_dflash_chain_writes_the_host_planned_upload(bonus):
    """Every draft count n and accepted count m: the DFlash chain step's upload is the host's of the plugin's commit,
    its header the MTP chain's, and the result page with the drafts step's section round-trips into the chained
    draft of that commit (the anchor the last committed column's slot, as one MTP layer's)."""
    rng = SeededRng(bonus)
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
            verify = upload(vp, chain_row(B, source, limit))
            packed, header = dflash_chain_step(verify, accept, argmax, DFLASH)
            positions = committed_positions(vp, m)
            assert packed == dflash_host_upload(source, committed, positions, m), (n, m)
            mtp_header = chain_step(verify, accept, argmax)[1]
            assert header == mtp_header and header[STATUS_AT : STATUS_AT + 4] == [1, source.slots[m - 1], bonus, 0]
            head = [rng.randrange(VOCAB) for _ in range(B)]
            result = unpack_result(header + dflash_drafts_step(packed, head, DFLASH))
            d = chained_draft(result, source, vp.positions)
            assert d.tokens == committed and d.positions == positions[:m] and d.drafts == head[:MAX_DRAFTS]
            assert d.matches(committed + [-1] * (1 + MAX_DRAFTS - m), positions, m)


@pytest.mark.parametrize("k", [1, 3, MAX_DRAFTS])
def test_dflash_chain_declines_where_propose_does(k):
    """The DFlash chain drafts while bonus + k < max_seq_len, and an idle or declined chain writes the idle upload
    (dflash_inputs with no drafts: no context row written, no drafts reported)."""
    seq = 4096
    rng = SeededRng(k)
    idle = dflash_inputs(DFLASH)
    for bonus in range(seq - k - 4, seq - k + 1):
        vp = one_row_pass(bonus - 1, MAX_DRAFTS, rng)
        accept = accept_page([(1, [11])])
        packed, header = dflash_chain_step(
            upload(vp, chain_row(B, draft_source([vp]), seq - k)), accept, [0] * B, DFLASH
        )
        drafts = bonus + k < seq
        assert header[STATUS_AT] == int(drafts)
        assert (packed == idle) != drafts
    packed, header = dflash_chain_step(upload(vp), accept, [0] * B, DFLASH)
    assert packed == idle and header[STATUS_AT : STATUS_AT + 4] == [0, NO_SLOT, 0, 0]
    assert dflash_drafts_step(idle, list(range(B)), DFLASH)[: 1 + MAX_DRAFTS] == [NO_SLOT] + [NO_TOKEN] * MAX_DRAFTS
    assert idle[DFLASH.RING_ROWS : DFLASH.words] == [DFLASH.ring] * 32


def test_dflash_pages_relay_as_the_mtp_pages():
    """A verify chained onto a DFlash pass relays its page as an MTP pass's: the last committed token at the bonus,
    then the DFlash drafts."""
    rng = SeededRng(9)
    for m in range(1, MAX_DRAFTS + 2):
        vp = one_row_pass(5000 - m, MAX_DRAFTS, rng)
        source = draft_source([vp])
        committed = [rng.randrange(VOCAB) for _ in range(m)]
        packed, header = dflash_chain_step(
            upload(vp, chain_row(B, source, MAX_SEQ_LEN)), accept_page([(m, committed)]), [0] * B, DFLASH
        )
        head = [rng.randrange(VOCAB) for _ in range(B)]
        page = header + dflash_drafts_step(packed, head, DFLASH)
        nxt = one_row_pass(9000, MAX_DRAFTS, rng)  # provisional positions above the real ones
        verify, ctrl = relay_step(
            upload(nxt, chain_row(B, draft_source([nxt]), MAX_SEQ_LEN, relay=True)),
            page,
            control_words(pass_inputs(nxt, None)[1]),
        )
        assert verify[: 1 + MAX_DRAFTS] == [committed[-1]] + head[:MAX_DRAFTS]
        assert verify[B : B + 1 + MAX_DRAFTS] == list(range(5000, 5000 + 1 + MAX_DRAFTS))
        assert ctrl[4 : 4 + MAX_DRAFTS] == head[:MAX_DRAFTS]


# ---------------------------------------------------------------------- with the accept kernel's mirror
@pytest.mark.parametrize("seed", range(40))
def test_greedy_accept_then_chain(seed):
    """An argmax_ids step through the device path: SampleVerify's greedy walk over the verify's argmax ids commits
    what the plugin's greedy walk commits (the cells argmax_ids answers), and the chain drafts from that commit as the
    host would have after reading it."""
    rng = SeededRng(seed)
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


# ---------------------------------------------------------------------- the relay step (a chained verify)
def pending_page(p, m, rng, limit=MAX_SEQ_LEN - MAX_DRAFTS, layers=1):
    """The result page of a deferred verify of one request at p with MAX_DRAFTS drafts that committed m tokens, and
    its committed tokens and drafts' head rows (``layers``: the drafter's MTP layers)."""
    vp = one_row_pass(p, MAX_DRAFTS, rng)
    committed = [rng.randrange(VOCAB) for _ in range(m)]
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    page = chain_pass(vp, accept_page([(m, committed)]), list(range(B)), heads, limit=limit, layers=layers)
    return page, committed, heads


def host_block(last, bonus, drafts, limit=MAX_SEQ_LEN - MAX_DRAFTS):
    """The upload and control page the host would build for the next verify, having read the page."""
    n = len(drafts)
    (vp,) = plan_verify_passes([(5, [last] + drafts, list(range(bonus, bonus + n + 1)), n)])
    return upload(vp, chain_row(B, draft_source([vp]), limit)), control_words(pass_inputs(vp, None)[1]), vp


def chained_block(bound, n, limit=MAX_SEQ_LEN - MAX_DRAFTS):
    """What the adapter uploads for a chain row: placeholder tokens at the provisional positions, relayed."""
    (vp,) = plan_verify_passes([(5, [0] * (n + 1), list(range(bound, bound + n + 1)), n)])
    return upload(vp, chain_row(B, draft_source([vp]), limit, relay=True)), control_words(pass_inputs(vp, None)[1])


@pytest.mark.parametrize("layers", [1, MAX_DRAFTS], ids=["mtp1", "mtp3"])
@pytest.mark.parametrize("bonus", BONUS + [MAX_SEQ_LEN - 1 - MAX_DRAFTS])
def test_the_relay_writes_the_block_the_host_would_build(bonus, layers):
    """With either drafter: the relay reads the page the step chained onto left (with every layer the drafts are the
    last slot's, wherever the commit ended)."""
    rng = SeededRng(bonus)
    for m in range(1, MAX_DRAFTS + 2):
        p = bonus - m
        if p < 0:
            continue
        page, committed, heads = pending_page(p, m, rng, layers=layers)
        result = unpack_result(page)
        assert result.drafting and result.bonus == bonus
        anchor = result.anchor
        assert anchor == (SPEC_SLOTS - 1 if layers > 1 else m - 1)
        for n in range(1, MAX_DRAFTS + 1):
            drafts = [heads[k][anchor] for k in range(n)]
            want_upload, want_ctrl, vp = host_block(committed[-1], bonus, drafts)
            # The plugin's provisional column 0: every in-flight step counted at its full width, an upper bound.
            for bound in (bonus, bonus + 1, p + 2 * (1 + MAX_DRAFTS)):
                overlay, ctrl = chained_block(bound, n)
                got, got_ctrl = relay_step(overlay, page, ctrl)
                assert got[: 2 * B] == want_upload[: 2 * B], (bonus, m, n, bound)
                chain = got[2 * B :]
                assert chain[: CHAIN_WORDS - 1] == want_upload[2 * B : 2 * B + CHAIN_WORDS - 1]
                assert got_ctrl == want_ctrl
                # The verify then runs as the host-built one: the same derived inputs and chain step.
                caps, own_cap, own_col = slot_caps(B), owned_slot_caps(B, 8, 2), owned_slot_columns(B, 8, 2)
                a = derive_verify(torch.tensor(got, dtype=torch.int32), own_cap[3], own_col[3], caps, True)
                b = derive_verify(torch.tensor(want_upload, dtype=torch.int32), own_cap[3], own_col[3], caps, True)
                for name in a:
                    pairs = zip(a[name], b[name]) if name == "serial" else [(a[name], b[name])]
                    assert all(torch.equal(x, y) for x, y in pairs), name
                argmax = [rng.randrange(VOCAB) for _ in range(B)]
                accept = verify_pass(None, None, argmax, got_ctrl, VOCAB // 8)
                assert accept == verify_pass(None, None, argmax, want_ctrl, VOCAB // 8)
                assert chain_step(got, accept, argmax, layers) == chain_step(want_upload, accept, argmax, layers)
                header = chain_step(got, accept, argmax, layers)[1]
                assert header[POSITIONS_AT : POSITIONS_AT + SPEC_SLOTS] == [
                    (q if q >= 0 else q & 0xFFFFFFFF) for q in vp.positions
                ]


def test_the_relay_leaves_every_other_slot_and_unrelayed_uploads_alone():
    rng = SeededRng(11)
    page, committed, heads = pending_page(500, 2, rng)
    # One drafted column relayed into slots 0 .. 1; the host's own content in slots 2 .. 3 and past the pass.
    overlay, ctrl = chained_block(510, 1)
    overlay[2], overlay[3], overlay[B + 2], overlay[B + 3] = 77, 78, 900, 901
    overlay[20], overlay[B + 20] = 79, 902
    ctrl[16:20] = [1, 2, 3, 4]  # group 1's words
    got, got_ctrl = relay_step(overlay, page, ctrl)
    keep = [i for i in range(2 * B) if i not in (0, 1, B, B + 1)]
    assert [got[i] for i in keep] == [overlay[i] for i in keep]
    assert got_ctrl[16:] == ctrl[16:] and got_ctrl[:4] == ctrl[:4] and got_ctrl[5:16] == ctrl[5:16]
    assert got[:2] == [committed[-1], heads[0][1]] and got[B : B + 2] == [502, 503]
    # Without the relay word nothing changes, whatever the page holds.
    plain, plain_ctrl = host_block(5, 502, [6, 7])[:2]
    assert relay_step(plain, page, plain_ctrl) == (plain, plain_ctrl)


@pytest.mark.parametrize("layers", [1, MAX_DRAFTS], ids=["mtp1", "mtp3"])
def test_a_page_that_drafted_nothing_idles_the_relayed_pass(layers):
    """The step chained onto declined its draft at the context limit, or its drafts are not the chain's (a host-planned
    draft ran after it): the relayed columns go idle (no K / V written), the pass drafts nothing, and its page reports
    position -1, which the adapter raises on."""
    rng = SeededRng(12)
    limit = 2000
    declined, _, _ = pending_page(limit - 2, 2, rng, limit=limit, layers=layers)  # the bonus reaches the limit
    assert not unpack_result(declined).drafting
    other, _, _ = pending_page(700, 1, rng, layers=layers)
    other[DRAFTS_AT] = 2  # the drafts step's anchor is not the chain's
    for page in (declined, other):
        overlay, ctrl = chained_block(limit + 4, MAX_DRAFTS, limit=MAX_SEQ_LEN - MAX_DRAFTS)
        got, got_ctrl = relay_step(overlay, page, ctrl)
        assert got[: 1 + MAX_DRAFTS] == [0] * (1 + MAX_DRAFTS) and got[B : B + 1 + MAX_DRAFTS] == [-1] * (
            1 + MAX_DRAFTS
        )
        assert got_ctrl[4 : 4 + MAX_DRAFTS] == [0] * MAX_DRAFTS and got[2 * B] == 0, "the source is cleared"
        mtp, header = chain_step(got, accept_page([(1, [5])]), [0] * B, layers)
        assert mtp == (pack_mtp(B) if layers == 1 else pack_mtp_layers(B)).tolist() and header[STATUS_AT] == 0
        result = unpack_result(header + drafts_step(mtp, [[0] * B] * MAX_DRAFTS, layers))
        assert result.positions == [-1] * SPEC_SLOTS and result.draft_anchor is None


@pytest.mark.parametrize("layers", [1, MAX_DRAFTS], ids=["mtp1", "mtp3"])
def test_a_chain_of_relays_follows_the_host_built_steps(layers):
    """Steps chained one onto the other on the mirrors (relay, the oracle's argmax, the accept walk, the chain and
    drafts steps; with every layer the draft trace's plan step on its stash and the tokens steps) commit exactly what
    host-built steps over the read pages commit, and with every layer plan the same rows (stash hits included)."""
    rng = SeededRng(13)
    seq = [rng.randrange(VOCAB) for _ in range(600)]

    def wrong(bonus, k):
        """A drafter wrong at every third bonus position on its second draft."""
        return seq[bonus + 1 + k] if bonus % 3 or k != 1 else (seq[bonus + 1 + k] + 1) % VOCAB

    def run(upload_words, ctrl, flow):
        argmax = [seq[q + 1] if q >= 0 else 0 for q in upload_words[B : 2 * B]]
        accept = verify_pass(None, None, argmax, ctrl, VOCAB // 8)
        mtp, header = chain_step(upload_words, accept, argmax, layers)
        heads = [[0] * B for _ in range(MAX_DRAFTS)]
        if layers == 1:
            for s in range(SPEC_SLOTS):
                if mtp[2 * B + s] >= 0:
                    bonus = mtp[5 * B + s]
                    for k in range(MAX_DRAFTS):
                        heads[k][s] = wrong(bonus, k)
        else:
            inputs, flow["stash"] = plan_step(mtp, flow["stash"], B)
            rows = [j for j in range(SPEC_SLOTS) if inputs["pos"][j] >= 0]
            assert all(inputs["tokens0"][j] == seq[inputs["rope"][j] + 1] for j in rows), "committed tokens"
            flow["plans"].append([(inputs["rope"][j], inputs["hidden"][j]) for j in rows])
            anchor = SPEC_SLOTS - 1
            if rows:
                bonus = inputs["rope"][anchor] + 1
                for k in range(MAX_DRAFTS):
                    heads[k][anchor] = wrong(bonus, k)
                for layer in range(1, MAX_DRAFTS):
                    assert tokens_step(inputs["tokens0"], heads, layer)[anchor] == heads[layer - 1][anchor]
        return header + drafts_step(mtp, heads, layers)

    chained_flow = {"stash": stash_page(B), "plans": []}
    host_flow = {"stash": stash_page(B), "plans": []}
    first, ctrl, _ = host_block(seq[100], 100, seq[101:104])
    page = run(first, ctrl, chained_flow)
    host_flow["stash"], host_flow["plans"] = list(chained_flow["stash"]), list(chained_flow["plans"])
    chained_seq = seq[:101] + unpack_result(page).accept[0][1]
    host_seq = list(chained_seq)
    host_page = page
    while len(chained_seq) < 400:
        # Chained: the relay builds the block from the page; host-built: from the page read on the host.
        overlay, ctrl = chained_block(len(chained_seq) + 2 * MAX_DRAFTS, MAX_DRAFTS)
        page = run(*relay_step(overlay, page, ctrl), chained_flow)
        r = unpack_result(page)
        assert r.positions[0] == len(chained_seq) - 1
        chained_seq += r.accept[0][1]
        h = unpack_result(host_page)
        upload_words, host_ctrl, _ = host_block(host_seq[-1], len(host_seq) - 1, h.drafts)
        host_page = run(upload_words, host_ctrl, host_flow)
        host_seq += unpack_result(host_page).accept[0][1]
        assert chained_seq == host_seq
    assert chained_seq == seq[: len(chained_seq)]
    assert chained_flow["plans"] == host_flow["plans"] and chained_flow["stash"] == host_flow["stash"]
    if layers > 1:
        # After the first two drafts every draft re-runs the rows layers 1 and 2 drafted: 3 or 4 rows.
        assert all(len(rows) in (MAX_DRAFTS, SPEC_SLOTS) for rows in chained_flow["plans"][2:])


# ---------------------------------------------------------------------- every MTP layer (MIMO_MTP_LAYERS=3)
def host_description(source, committed, positions, m):
    """What propose_draft_tokens uploads for the plugin's commit with every layer (pack_mtp_layers)."""
    return pack_mtp_layers(B, source, committed, positions, m).tolist()


@pytest.mark.parametrize("bonus", BONUS + [MAX_SEQ_LEN - 1 - MAX_DRAFTS])
def test_layers_chain_writes_the_commits_description(bonus):
    """Every draft count n and accepted count m: the chain step's upload is the description propose_draft_tokens
    builds from the plugin's commit, the anchor the last slot; it declines past the limit as with one layer."""
    rng = SeededRng(100 + bonus)
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
            for lim, drafting in ((limit, True), (bonus, False)):
                mtp, header = chain_step(upload(vp, chain_row(B, source, lim)), accept, [0] * B, layers=MAX_DRAFTS)
                want = host_description(source, committed, committed_positions(vp, m), m) if drafting else None
                assert mtp == (want if drafting else pack_mtp_layers(B).tolist()), (n, m, drafting)
                assert header[STATUS_AT : STATUS_AT + 4] == (
                    [1, SPEC_SLOTS - 1, bonus, 0] if drafting else [0, NO_SLOT, 0, 0]
                )
    # An idle chain row, and the one-layer idle upload (an idle description too: SpecInputs.reset writes it).
    vp = one_row_pass(300, 3, rng)
    mtp, _ = chain_step(upload(vp), accept_page([(2, [7, 8])]), [0] * B, layers=MAX_DRAFTS)
    assert mtp == pack_mtp_layers(B).tolist() and pack_mtp(B)[:B].tolist() == pack_mtp_layers(B)[:B].tolist()


def stash_cases(a):
    """Stash pages before a draft whose committed rows start at ``a`` (A): the last draft's rows in every layout the
    flows leave (anchor at A - 1 with 1 .. 4 rows), unrelated positions, a hole at A - 2, and empty."""
    rng = SeededRng(a)
    cases = []
    for rows in range(1, SPEC_SLOTS + 1):
        positions = [-1] * (SPEC_SLOTS - rows) + [a - rows + j for j in range(rows)]
        cases.append(positions)
    cases += [[a - 40, a - 39, a - 38, a - 37], [-1, a - 3, -1, a - 1], [-1] * SPEC_SLOTS]
    out = []
    for positions in cases:
        positions = [p if p >= 0 else -1 for p in positions]
        out.append((positions, [rng.randrange(VOCAB) if p >= 0 else 0 for p in positions]))
    return out


@pytest.mark.parametrize("bonus", BONUS + [MAX_SEQ_LEN - 1])
def test_plan_step_is_plan_mtp_layers(bonus):
    """The plan step on the description of every commit (count m, a verify's slots or a plain step's one slot) and
    every stash layout writes plan_mtp_layers' passes: ring cache positions, serial positions, RoPE indices, the
    hidden table rows of its sources (step slot s: s; stash slot k: T + k) and layer-0 tokens; and the new stash is
    mtp_stash(plan) with those tokens. Around the ring's wrap and at the end of the context."""
    rng = SeededRng(bonus)
    sources = [(DraftSource(row=1, slots=[0, 1, 2, 3]), m) for m in range(1, SPEC_SLOTS + 1)]
    sources += [(DraftSource(row=1, slots=[s]), 1) for s in range(SPEC_SLOTS)]
    for source, m in sources:
        if bonus - m < 0:
            continue
        positions = [bonus - m + 1 + c for c in range(m)] + [-1] * (1 + MAX_DRAFTS - m)
        committed = [rng.randrange(VOCAB) for _ in range(m)]
        mtp = host_description(source, committed, positions, m)
        a = positions[0] - 1
        for stash_positions, stash_tokens in stash_cases(a):
            page = stash_page(B, stash_positions, stash_tokens)
            got, new = plan_step(mtp, page, B)
            plan = plan_mtp_layers(source, positions, m, MtpStash(stash_positions))
            want_hidden, want_tokens = [B + j for j in range(B)], [0] * B
            for j in plan.rows:
                kind, k = plan.sources[j]
                want_hidden[j] = k if kind == "step" else B + k
                want_tokens[j] = committed[source.slots.index(k)] if kind == "step" else stash_tokens[k]
            assert got["pos"] == [mtp_cache_position(p) for p in plan.positions] + [-1] * (B - SPEC_SLOTS)
            assert got["rope"] == [max(p, 0) for p in plan.positions] + [0] * (B - SPEC_SLOTS)
            assert got["serial"] == [serial_positions(got["pos"], SPEC_SLOTS)[j] for j in range(SPEC_SLOTS)]
            assert got["hidden"] == want_hidden and got["tokens0"] == want_tokens
            assert new == stash_page(B, mtp_stash(plan).positions, want_tokens[:SPEC_SLOTS])
            assert max(got["rope"]) < MAX_SEQ_LEN


def test_an_idle_description_leaves_the_stash():
    page = stash_page(B, [-1, 5, 6, 7], [0, 1, 2, 3])
    for mtp in (pack_mtp_layers(B).tolist(), pack_mtp(B).tolist()):
        got, new = plan_step(mtp, page, B)
        assert new == page and got["pos"] == [-1] * B and got["hidden"] == [B + j for j in range(B)]
        assert got["tokens0"] == [0] * B and got["rope"] == [0] * B


def test_warm_leaves_the_stash_a_prompt_emptied():
    """tt/spec_chain.SpecChain.warm with its steps on the mirrors: a recapture after a prompt (the prompt emptied the
    stash, TtMiMoMTP.reset) finds the last served chained verify and its accept page in the uploads; warm idles them
    (SpecInputs.reset) before its steps, so the plan step leaves the stash empty instead of re-planning that commit."""
    from models.experimental.mimo_v2_6_flash.tt.spec_chain import SpecChain

    rng = SeededRng(9)
    vp = one_row_pass(300, MAX_DRAFTS, rng)
    accept = accept_page([(2, [rng.randrange(VOCAB) for _ in range(2)])])
    live = SimpleNamespace(verify=upload(vp, chain_row(B, draft_source([vp]), MAX_SEQ_LEN - MAX_DRAFTS)), mtp=None)
    live.stash = stash_page(B)
    mtp, _ = chain_step(live.verify, accept, [0] * B, layers=MAX_DRAFTS)
    assert plan_step(mtp, live.stash, B)[1] != live.stash, "the live uploads' commit fills the stash"

    def reset():
        live.verify = pack_verify(torch.zeros(B, dtype=torch.int32), torch.full((B,), -1, dtype=torch.int32)).tolist()
        live.mtp = pack_mtp(B).tolist()

    chain = SpecChain.__new__(SpecChain)
    chain.spec_inputs, chain.layer_inputs, chain.layers = SimpleNamespace(reset=reset), object(), MAX_DRAFTS
    chain.relay = lambda ctrl: setattr(live, "verify", relay_step(live.verify, [0] * RESULT_WORDS, ctrl)[0])
    chain.chain = lambda: setattr(live, "mtp", chain_step(live.verify, accept, [0] * B, layers=MAX_DRAFTS)[0])
    chain.drafts = chain.tokens = lambda *args: None
    chain.plan_layers = lambda: setattr(live, "stash", plan_step(live.mtp, live.stash, B)[1])
    chain.warm(control_words([]))
    assert live.stash == stash_page(B) and live.mtp == pack_mtp_layers(B).tolist()


@pytest.mark.parametrize("rows", range(1, SPEC_SLOTS + 1))
def test_tokens_step_is_mtp_layer_tokens(rows):
    rng = SeededRng(rows)
    tokens0 = [0] * (SPEC_SLOTS - rows) + [rng.randrange(VOCAB) for _ in range(rows)] + [0] * (B - SPEC_SLOTS)
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    drafts = [h[SPEC_SLOTS - 1] for h in heads]
    used = tokens0[SPEC_SLOTS - rows : SPEC_SLOTS]
    for layer in range(1, MAX_DRAFTS):
        out = tokens_step(tokens0, heads, layer)
        assert out[SPEC_SLOTS - rows : SPEC_SLOTS] == mtp_layer_tokens(used, drafts, layer)
        assert out[SPEC_SLOTS:] == [0] * (B - SPEC_SLOTS)


def test_layers_drafts_step_and_chained_draft():
    rng = SeededRng(9)
    heads = [[rng.randrange(VOCAB) for _ in range(B)] for _ in range(MAX_DRAFTS)]
    vp = one_row_pass(1000, 3, rng)
    committed = [rng.randrange(VOCAB) for _ in range(2)]
    chain = chain_row(B, draft_source([vp]), MAX_SEQ_LEN - MAX_DRAFTS)
    mtp, header = chain_step(upload(vp, chain), accept_page([(2, committed)]), [0] * B, layers=MAX_DRAFTS)
    result = unpack_result(header + drafts_step(mtp, heads, layers=MAX_DRAFTS))
    assert result.drafting and result.anchor == result.draft_anchor == SPEC_SLOTS - 1 and result.bonus == 1002
    d = chained_draft(result, draft_source([vp]), vp.positions, layers=MAX_DRAFTS)
    assert d.tokens == committed and d.positions == [1001, 1002] and d.drafts == [h[SPEC_SLOTS - 1] for h in heads]
    idle = drafts_step(pack_mtp_layers(B).tolist(), heads, layers=MAX_DRAFTS)
    assert idle[0] == NO_SLOT and idle[1 : 1 + MAX_DRAFTS] == [NO_TOKEN] * MAX_DRAFTS
    # A host description of a plain step's commit drafts at the last slot too.
    host = host_description(DraftSource(row=0, slots=[2]), [9], [701, -1, -1, -1], 1)
    assert drafts_step(host, heads, layers=MAX_DRAFTS)[0] == SPEC_SLOTS - 1


# ---------------------------------------------------------------------- the device program's mirror, end to end
class ProgramDrafter:
    """The device program (reference.mtp.MtpProgramMirror) behind ReferenceMTPDrafter's interface, so that the host
    tests' driver serves it: plan() keeps the commit, draft() uploads its description and replays the program on the
    step's final norm rows (the slots' hidden, zeros elsewhere)."""

    def __init__(self, reference):
        self.program = MtpProgramMirror(reference.mtps, reference.embedding, reference.lm_head, table_rows=B)
        self.mtps, self.outputs, self.stash = reference.mtps, None, None

    def plan(self, source, committed_positions, accepted):
        mtp = pack_mtp_layers(B, source, [0] * SPEC_SLOTS, committed_positions, accepted).tolist()
        inputs, _ = plan_step(mtp, self.program.stash, B)
        positions = [inputs["rope"][j] if inputs["pos"][j] >= 0 else -1 for j in range(SPEC_SLOTS)]
        return SimpleNamespace(source=source, committed=committed_positions, m=accepted, positions=positions)

    def draft(self, plan, slot_tokens, slot_hidden):
        tokens = [slot_tokens[plan.source.slots[c]] for c in range(plan.m)]
        mtp = pack_mtp_layers(B, plan.source, tokens, plan.committed, plan.m).tolist()
        step = torch.zeros(B, next(iter(slot_hidden.values())).shape[-1])
        for s, h in slot_hidden.items():
            step[s] = h
        out = self.program.run(mtp, step)
        self.outputs = self.program.outputs
        assert out[0] == SPEC_SLOTS - 1
        return out[1 : 1 + MAX_DRAFTS]


def test_the_device_program_drafts_what_the_reference_drafter_drafts():
    """Every accepted count after every other, and plain steps: the device program's mirror (plan step on the device
    stash, gather, tokens step, three passes) drafts exactly the reference drafter's drafts, which equal a
    recomputation of every layer over the whole sequence."""
    g = torch.Generator().manual_seed(1)
    cfg = tiny_config()
    target = TinyMiMo(cfg, g)
    reference = mtp_drafter(cfg, target, g, MAX_DRAFTS)
    prompt = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38]
    want = [
        (drafts, [o.clone() for o in outputs]) for _, drafts, outputs, _ in drive(target, reference, prompt, SCHEDULE)
    ]
    reference.reset()
    program = ProgramDrafter(reference)
    got = list(drive(target, program, prompt, SCHEDULE))
    assert len(got) == len(want)
    for (_, drafts, outputs, (re_drafts, re_outputs)), (want_drafts, want_outputs) in zip(got, want):
        assert drafts == want_drafts == re_drafts
        for a, b in zip(outputs, want_outputs):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
