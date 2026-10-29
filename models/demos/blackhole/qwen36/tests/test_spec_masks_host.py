# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of tt/spec_masks.py and of the mask paths of tt/spec_fused_sampled.py and tt/spec_engine.py (phase 3 P1, no device).

H2 bit layout, H6 tie guard and greedy tie rule, H10 masked host_redo_walk against the float64 full-vocabulary reference,
the masked fused_sampled_reference against the masked host walk, H4-lite (the composed mask path commits what the plugin's
accept_sampled_drafts commits with the same grammar bitmask rows and uniforms), and the engine plumbing on a stub engine.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_masks_host.py"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_masks as SM
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine
from models.demos.blackhole.qwen36.tt.spec_fused import fused_accept_aux_host
from models.demos.blackhole.qwen36.tt.spec_fused_sampled import (
    OVF_TIE,
    _row_distribution_full,
    fused_sampled_reference,
    host_redo_walk,
    merge_redo,
    predraw_uniforms,
)

K, T, TP, M = 3, 4, 8, 64
SCRATCH, NB, CAP = 5000, 6, 4096


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _aux(B, nv, g):
    return fused_accept_aux_host(
        nv,
        torch.randint(1, 60, (B,), generator=g),
        torch.ones(B, dtype=torch.bool),
        torch.zeros(B, dtype=torch.bool),
        CAP,
        K,
        SCRATCH,
        0,
        4096,
    )


def _union(x, per):
    """The topk trace readback of full logits x [R, TP*per]: per-shard top-M, (value desc, local id asc), bf16 values."""
    vs, ls = [], []
    for s in range(TP):
        sh = x[:, s * per : (s + 1) * per].float()
        o = torch.sort(sh, dim=1, descending=True, stable=True).indices[:, :M]
        ls.append(o)
        vs.append(sh.gather(1, o))
    return torch.cat(vs, 1).to(torch.bfloat16), torch.cat(ls, 1)


def _device_mask(x, allowed):
    """What the mask trace writes into the shard logits: x where allowed, MASK_VALUE (in x's dtype) elsewhere."""
    return torch.where(allowed, x, torch.full_like(x, SM.MASK_NEG))


def _rand_allowed(g, x, kind):
    """bool [V] allowed set of one row of logits x [V] for a mask kind."""
    V = int(x.numel())
    if kind == "dense":
        a = torch.rand(V, generator=g) < 0.7
    elif kind == "structure":  # 2-8 tokens, anywhere (usually outside the top-64 of the row)
        a = torch.zeros(V, dtype=torch.bool)
        a[torch.randperm(V, generator=g)[: int(torch.randint(2, 9, (), generator=g))]] = True
    elif kind == "top":  # a few of the top-8 tokens
        a = torch.zeros(V, dtype=torch.bool)
        a[
            torch.topk(x.float(), 8).indices[
                torch.randperm(8, generator=g)[: int(torch.randint(1, 5, (), generator=g))]
            ]
        ] = True
    elif kind == "low":  # only tokens far below the top-64
        a = torch.zeros(V, dtype=torch.bool)
        low = torch.sort(x.float(), descending=True).indices[200:]
        a[low[torch.randperm(int(low.numel()), generator=g)[:5]]] = True
    else:
        raise AssertionError(kind)
    if not bool(a.any()):
        a[int(torch.randint(0, V, (), generator=g))] = True
    return a


def _walk_ref64(masked_logits, tok, params, uni, B, nv):
    """The float64 full-vocabulary reference walk on logits already carrying -inf: _row_distribution_full on every row (a
    full sort), the plugin's accept rule. Independent of host_redo_walk's fast paths and its compact-row shortcut."""
    committed = torch.full((B, T), -1, dtype=torch.int32)
    counts = torch.ones(B, dtype=torch.int32)
    for u in range(B):
        n = int(nv[u])

        def amax(r):
            row = masked_logits[r].float()
            return int((row == row.max()).nonzero()[0])

        if bool(params["greedy"][u]):
            m = 0
            while m < n and amax(u * T + m) == int(tok[u, m + 1]):
                m += 1
            for j in range(m + 1):
                committed[u, j] = amax(u * T + j)
            counts[u] = m + 1
            continue
        m, rejected, dist = 0, False, None
        while m < n:
            order, prob = _row_distribution_full(
                masked_logits[u * T + m],
                float(params["temperature"][u]),
                int(params["top_k"][u]),
                float(params["top_p"][u]),
            )
            hit = (order == int(tok[u, m + 1])).nonzero()
            p_d = float(prob[int(hit[0])]) if hit.numel() else 0.0
            if p_d > 0 and p_d >= float(uni[u, 1 + m]):
                committed[u, m] = int(tok[u, m + 1])
                m += 1
                continue
            rejected, dist = True, (order, prob)
            break
        if dist is None:
            dist = _row_distribution_full(
                masked_logits[u * T + m],
                float(params["temperature"][u]),
                int(params["top_k"][u]),
                float(params["top_p"][u]),
            )
        order, prob = dist
        w = prob.clone()
        if rejected:
            w[order == int(tok[u, m + 1])] = 0.0
        cdf = w.cumsum(0)
        uc = float(uni[u, K + 1] if rejected else uni[u, 0])
        pick = min(int((cdf <= uc * float(cdf[-1])).sum()), int((w > 0).nonzero().max()))
        committed[u, m] = int(order[pick])
        counts[u] = m + 1
    return committed, counts


def _case(g, B, V, kinds, bf16=True):
    """Random masked step: logits [R, V], allowed [R, V] bool, packed mask [B, T, V/32], tokens [B, T], num_valid, params, uniforms."""
    R = B * T
    scale = torch.rand(R, 1, generator=g) * 3 + 1.0
    x = torch.randn(R, V, generator=g) * scale
    if bf16:
        x = x.to(torch.bfloat16)
    allowed = torch.zeros(R, V, dtype=torch.bool)
    for r in range(R):
        allowed[r] = _rand_allowed(g, x[r], kinds[int(torch.randint(0, len(kinds), (), generator=g))])
    nv = torch.randint(0, K + 1, (B,), generator=g)
    tok = torch.zeros(B, T, dtype=torch.int64)
    for u in range(B):
        for j in range(K):
            if int(torch.randint(0, 4, (), generator=g)) == 0:  # a draft the mask may forbid
                tok[u, j + 1] = int(torch.randint(0, V, (), generator=g))
            else:
                ids = allowed[u * T + j].nonzero().reshape(-1)
                tok[u, j + 1] = int(ids[int(torch.randint(0, int(ids.numel()), (), generator=g))])
    temp = torch.rand(B, generator=g) * 1.2 + 0.3
    top_k = torch.randint(1, 64, (B,), generator=g)
    top_p = torch.tensor([1.0, 0.95, 0.8, 0.5])[torch.randint(0, 4, (B,), generator=g)]
    return dict(
        x=x,
        allowed=allowed,
        mask=SM.pack_allowed(allowed).reshape(B, T, -1),
        tok=tok,
        nv=nv,
        temp=temp,
        top_k=top_k,
        top_p=top_p,
    )


# --------------------------------------------------------------------------- #
# H2: bit layout
# --------------------------------------------------------------------------- #
def test_constants_and_shard_split(expect_error):
    assert SM.VOCAB == 8 * 31040 == 8 * 970 * 32 == 7760 * 32
    assert (SM.MASK_WORDS, SM.SHARD_WORDS, SM.SHARDS, SM.MASK_ROWS, SM.RING_DEPTH) == (7760, 970, 8, 32, 2)
    mask = torch.arange(5 * SM.MASK_WORDS, dtype=torch.int32).reshape(5, SM.MASK_WORDS)
    sh = SM.shard_split(mask)
    assert tuple(sh.shape) == (5, 8, 970)
    for d in range(8):
        assert torch.equal(sh[:, d], mask[:, d * 970 : (d + 1) * 970])
    with expect_error(ValueError, ".*"):
        SM.shard_split(torch.zeros(2, 100, dtype=torch.int32))


@pytest.mark.parametrize("v", [0, 1, 31, 32, 33, 31039, 31040, 31041, 124159, 248288, 248319])
def test_bit_v_mod_32_of_word_v_div_32(v):
    allowed = torch.zeros(2, SM.VOCAB, dtype=torch.bool)
    allowed[1, v] = True
    mask = SM.pack_allowed(allowed)
    assert tuple(mask.shape) == (2, SM.MASK_WORDS) and mask.dtype == torch.int32
    assert int(mask[0].abs().sum()) == 0
    nz = mask[1].nonzero().reshape(-1)
    assert nz.tolist() == [v // 32]
    word = int(mask[1, v // 32]) & 0xFFFFFFFF  # the bit pattern, whatever the sign
    assert word == 1 << (v % 32)
    # the shard that owns the token holds the word, and only that shard
    d = v // SM.SHARD_VOCAB
    sh = SM.shard_split(mask)[1]
    assert [i for i in range(8) if int(sh[i].abs().sum())] == [d]
    assert int(sh[d, (v - d * SM.SHARD_VOCAB) // 32]) & 0xFFFFFFFF == 1 << (v % 32)
    assert SM.unpack_allowed(mask).nonzero().tolist() == [[1, v]]
    assert bool(SM.token_allowed(mask[1:2], torch.tensor([[v, 0, 248319]]))[0, 0])


def test_roundtrip_all_ones_padding_and_empty_rows(expect_error):
    g = torch.Generator().manual_seed(1)
    allowed = torch.rand(3, SM.VOCAB, generator=g) < 0.5
    mask = SM.pack_allowed(allowed)
    assert torch.equal(SM.unpack_allowed(mask), allowed)
    ones = SM.all_ones(2, 3)
    assert tuple(ones.shape) == (2, 3, 7760) and bool(SM.is_unconstrained(ones).all())
    assert bool(SM.unpack_allowed(ones).all())
    assert not bool(SM.is_unconstrained(mask).any())
    one_zero = ones.clone()
    one_zero[1, 2, 100] ^= 1 << 5
    assert SM.is_unconstrained(one_zero).tolist() == [[True, True, True], [True, True, False]]
    # vocabulary padding: ids past the tokenizer are cleared; a narrow mask has a disallowed tail
    cleared = SM.clear_padding(SM.all_ones(2), 248077)
    a = SM.unpack_allowed(cleared)
    assert bool(a[:, :248077].all()) and not bool(a[:, 248077:].any())
    assert int(cleared[0, 248077 // 32]) & 0xFFFFFFFF == (1 << (248077 % 32)) - 1
    narrow = SM.normalise_width(torch.full((1, 100), -1, dtype=torch.int32))
    assert tuple(narrow.shape) == (1, 7760)
    assert int(SM.unpack_allowed(narrow).sum()) == 3200
    with expect_error(ValueError, ".*"):
        SM.normalise_width(torch.zeros(1, 7761, dtype=torch.int32))
    # empty rows count only ids below the vocabulary
    only_pad = torch.zeros(2, SM.VOCAB, dtype=torch.bool)
    only_pad[0, 248200] = True
    only_pad[1, 5] = True
    pm = SM.pack_allowed(only_pad)
    assert SM.empty_rows(pm).tolist() == [False, False]
    assert SM.empty_rows(pm, 248077).tolist() == [True, False]
    assert SM.empty_rows(torch.zeros(1, 7760, dtype=torch.int32)).tolist() == [True]
    # compose = AND, equal to the AND of the unpacked rows
    a, b = (torch.rand(2, SM.VOCAB, generator=g) < 0.8 for _ in range(2))
    assert torch.equal(SM.unpack_allowed(SM.compose_masks(SM.pack_allowed(a), SM.pack_allowed(b))), a & b)
    # token_allowed equals the unpacked lookup
    ids = torch.randint(0, SM.VOCAB, (3, 50), generator=g)
    assert torch.equal(SM.token_allowed(mask, ids), allowed.gather(1, ids))


def test_apply_to_logits_does_not_touch_the_input():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, SM.VOCAB, generator=g)
    allowed = torch.rand(2, SM.VOCAB, generator=g) < 0.5
    y = SM.apply_to_logits(x, SM.pack_allowed(allowed))
    assert torch.equal(y[allowed], x[allowed]) and bool(torch.isneginf(y[~allowed]).all())
    z = SM.apply_to_logits(x, SM.pack_allowed(allowed), fill=SM.MASK_NEG)
    assert bool((z[~allowed] == SM.MASK_NEG).all())
    assert not bool(torch.isneginf(x).any())


def test_slot_to_engine_row_mapping_follows_to_rows(expect_error):
    B = 4
    g = torch.Generator().manual_seed(3)
    mask = SM.pack_allowed(torch.rand(3, T, SM.VOCAB, generator=g) < 0.5)  # 3 live users of B = 4
    phys = [2, 0, 3, 1]  # runner slot s -> engine row phys[s]
    perm = (torch.tensor(phys), torch.tensor([phys.index(r) for r in range(B)]))
    stub = SimpleNamespace(_phys_perm=perm)
    full = SM.mask_to_engine_rows(mask, None, B, T)  # identity map: runner order, the padding user all ones
    assert tuple(full.shape) == (B, T, 7760) and bool(SM.is_unconstrained(full[3]).all())
    assert torch.equal(full[:3], mask)
    eng = SM.mask_to_engine_rows(mask, perm[1], B, T)
    ref = MTPSpecEngine._to_rows(stub, full)[0]
    assert torch.equal(eng, ref)
    for s in range(B):
        assert torch.equal(eng[phys[s]], full[s])  # slot s lands on engine row phys[s]
    assert bool(SM.is_unconstrained(eng[phys[3]]).all())  # the padding slot's row stays all ones
    assert SM.mask_to_engine_rows(None, perm[1], B, T) is None
    with expect_error(ValueError, ".*"):
        SM.mask_to_engine_rows(mask[:, :3], None, B, T)
    # an unconstrained step stays unconstrained
    assert bool(SM.is_unconstrained(SM.mask_to_engine_rows(SM.all_ones(2, T), perm[1], B, T)).all())


def test_staging_ring_alternates_two_buffers(expect_error):
    ring = SM.MaskStagingRing()
    assert [tuple(b.shape) for b in ring.bufs] == [(32, 7760)] * 2 and all(b.dtype == torch.int32 for b in ring.bufs)
    g = torch.Generator().manual_seed(4)
    m1 = SM.pack_allowed(torch.rand(16, SM.VOCAB, generator=g) < 0.5)
    m2 = SM.pack_allowed(torch.rand(8, SM.VOCAB, generator=g) < 0.5)
    b1 = ring.stage(m1)
    b2 = ring.stage(m2)
    assert b1.data_ptr() != b2.data_ptr()
    assert torch.equal(b1[:16], m1) and bool((b1[16:] == -1).all())  # step N's buffer is intact while N+1 is staged
    assert torch.equal(b2[:8], m2) and bool((b2[8:] == -1).all())
    b3 = ring.stage(m2)
    assert b3.data_ptr() == b1.data_ptr() and torch.equal(b3[:8], m2) and bool((b3[8:] == -1).all())
    with expect_error(ValueError, ".*"):
        ring.stage(torch.zeros(33, 7760, dtype=torch.int32))


# --------------------------------------------------------------------------- #
# H6: tie guard and greedy tie rule
# --------------------------------------------------------------------------- #
def _small_ref(x, allowed, *, B, per, tok, nv, greedy, temp, top_k, top_p, uni, g, mask=True):
    uv, ul = _union(_device_mask(x, allowed), per)
    R = B * T
    return fused_sampled_reference(
        uv,
        ul,
        tok.reshape(-1),
        torch.zeros(R, dtype=torch.int64),
        _aux(B, nv, g),
        torch.randint(1, 4000, (B, NB), generator=g, dtype=torch.int32),
        B=B,
        K=K,
        per_shard=per,
        M=M,
        greedy=greedy,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        uniforms=uni,
        num_valid=nv,
        mask=SM.pack_allowed(allowed).reshape(B, T, -1) if mask else None,
    )


def test_guard_allowed_set_smaller_than_top_k_raises_no_tie_overflow():
    per, B = 512, 2
    V = TP * per
    g = torch.Generator().manual_seed(11)
    x = (torch.randn(B * T, V, generator=g) * 2).to(torch.bfloat16)
    allowed = torch.zeros(B * T, V, dtype=torch.bool)
    for r in range(B * T):  # 2-8 allowed tokens, top_k up to 63 > the allowed count
        allowed[r, torch.randperm(V, generator=g)[: 2 + r]] = True
    nv = torch.full((B,), K)
    tok = torch.randint(0, V, (B, T), generator=g)
    uni = torch.rand(B, K + 2, generator=g)
    for top_k in (5, 20, 63):
        out = _small_ref(
            x,
            allowed,
            B=B,
            per=per,
            tok=tok,
            nv=nv,
            greedy=torch.zeros(B, dtype=torch.bool),
            temp=torch.tensor([1.0, 0.7]),
            top_k=torch.full((B,), top_k),
            top_p=torch.ones(B),
            uni=uni,
            g=g,
        )
        assert int(out.overflow.sum()) == 0, (top_k, out.overflow)
        # the committed token is an allowed one
        for u in range(B):
            for j in range(int(out.counts[u])):
                assert bool(allowed[u * T + j, int(out.committed[u, j])]), (u, j)
    # a masked k-th value is what the guard ignores: the same step with the old rule (kth > -inf only) WOULD flag a tie
    from models.demos.blackhole.qwen36.tt.spec_fused_sampled import union_topk_stable

    uv, ul = _union(_device_mask(x, allowed), per)
    vals, _ = union_topk_stable(uv, ul, per, M)
    kth = vals[:, 62:63] / 1.0
    assert bool((kth < SM.MASK_KTH_GUARD).all()) and bool((vals[:, M - 1 :] >= kth).all())


def test_a_real_tie_at_the_kth_value_still_raises_the_overflow_bit():
    per, B = 512, 1
    V = TP * per
    g = torch.Generator().manual_seed(12)
    x = torch.ones(B * T, V, dtype=torch.bfloat16)  # one big tie group: it reaches the 64th candidate
    x[:, :5] = 2.0
    allowed = torch.ones(B * T, V, dtype=torch.bool)
    out = _small_ref(
        x,
        allowed,
        B=B,
        per=per,
        tok=torch.zeros(B, T, dtype=torch.int64),
        nv=torch.zeros(B, dtype=torch.int64),
        greedy=torch.zeros(B, dtype=torch.bool),
        temp=torch.tensor([1.0]),
        top_k=torch.tensor([10]),
        top_p=torch.ones(1),
        uni=torch.rand(B, K + 2, generator=g),
        g=g,
        mask=False,
    )
    assert int(out.overflow[0]) & OVF_TIE


def test_greedy_as_top_k1_top_p0_picks_the_lowest_id_among_ties():
    per, B = 512, 2
    V = TP * per
    g = torch.Generator().manual_seed(13)
    x = torch.zeros(B * T, V, dtype=torch.bfloat16)
    allowed = torch.zeros(B * T, V, dtype=torch.bool)
    tie = [[700, 90, 2600, 3000], [3500, 4000, 77, 1023]]  # a tie group over several shards and ids
    for r in range(B * T):
        ids = tie[r // T] + [11 + r]
        allowed[r, ids] = True
        x[r, ids[:-1]] = 3.0
        x[r, ids[-1]] = 1.0
    nv = torch.zeros(B, dtype=torch.int64)
    uni = torch.rand(B, K + 2, generator=g)
    out = _small_ref(
        x,
        allowed,
        B=B,
        per=per,
        tok=torch.zeros(B, T, dtype=torch.int64),
        nv=nv,
        greedy=torch.zeros(B, dtype=torch.bool),
        temp=torch.ones(B),
        top_k=torch.ones(B, dtype=torch.int64),
        top_p=torch.zeros(B),
        uni=uni,
        g=g,
    )
    assert int(out.overflow.sum()) == 0
    assert out.committed[:, 0].tolist() == [90, 77]  # lowest id of each tie group
    redo_c, _ = host_redo_walk(
        x,
        torch.zeros(B, T, dtype=torch.int64),
        {
            "temperature": torch.ones(B),
            "top_k": torch.ones(B, dtype=torch.int64),
            "top_p": torch.zeros(B),
            "greedy": torch.ones(B, dtype=torch.bool),
        },
        uni,
        B=B,
        K=K,
        num_valid=nv,
        mask=SM.pack_allowed(allowed).reshape(B, T, -1),
    )
    assert redo_c[:, 0].tolist() == [90, 77]
    # the same rows as a sampled row with top_k 1 / top_p 0 on the host walk give the same id
    redo_s, _ = host_redo_walk(
        x,
        torch.zeros(B, T, dtype=torch.int64),
        {
            "temperature": torch.ones(B),
            "top_k": torch.ones(B, dtype=torch.int64),
            "top_p": torch.zeros(B),
            "greedy": torch.zeros(B, dtype=torch.bool),
        },
        uni,
        B=B,
        K=K,
        num_valid=nv,
        mask=SM.pack_allowed(allowed).reshape(B, T, -1),
    )
    assert redo_s[:, 0].tolist() == [90, 77]


# --------------------------------------------------------------------------- #
# H10: masked host_redo_walk against the float64 full-vocabulary reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kinds,seed",
    [
        (["dense"], 21),
        (["structure"], 22),  # 2-8 allowed tokens, mostly outside the top-64
        (["top"], 23),
        (["low"], 24),  # allowed tokens far outside the top-64 union
        (["dense", "structure", "top", "low"], 25),
    ],
)
def test_masked_host_redo_walk_equals_the_float64_reference(kinds, seed):
    V, B = 4096, 4
    g = torch.Generator().manual_seed(seed)
    bad = n = 0
    for case in range(60):
        c = _case(g, B, V, kinds)
        if case % 3 == 0:  # flat rows: a large temperature and a narrow range
            c["x"] = (c["x"].float() * 0.05).to(torch.bfloat16)
        if case % 5 == 0:  # ties: bf16-coarse values
            c["x"] = (c["x"].float() * 0.2).round().to(torch.bfloat16)
        greedy = torch.rand(B, generator=g) < 0.25
        params = {"temperature": c["temp"], "top_k": c["top_k"], "top_p": c["top_p"], "greedy": greedy}
        torch.manual_seed(case)
        uni = predraw_uniforms({}, B, K, c["nv"])
        got = host_redo_walk(c["x"], c["tok"], params, uni, B=B, K=K, num_valid=c["nv"], mask=c["mask"])
        masked = c["x"].masked_fill(~c["allowed"], float("-inf"))
        want = _walk_ref64(masked, c["tok"], params, uni, B, c["nv"])
        n += 1
        bad += int(not (torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])))
    assert bad == 0, f"{bad} of {n} cases differ"


def test_masked_host_redo_walk_on_the_real_vocabulary():
    V, B = SM.VOCAB, 2
    g = torch.Generator().manual_seed(31)
    for kinds in (["structure"], ["low"], ["dense", "top"]):
        c = _case(g, B, V, kinds)
        params = {
            "temperature": c["temp"],
            "top_k": c["top_k"],
            "top_p": c["top_p"],
            "greedy": torch.tensor([False, True]),
        }
        torch.manual_seed(5)
        uni = predraw_uniforms({}, B, K, c["nv"])
        got = host_redo_walk(c["x"], c["tok"], params, uni, B=B, K=K, num_valid=c["nv"], mask=c["mask"])
        want = _walk_ref64(c["x"].masked_fill(~c["allowed"], float("-inf")), c["tok"], params, uni, B, c["nv"])
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), kinds
        for u in range(B):  # every committed token is allowed
            for j in range(int(got[1][u])):
                assert bool(c["allowed"][u * T + j, int(got[0][u, j])])


def test_masked_host_redo_walk_rejects_an_empty_row_and_keeps_unconstrained_rows(expect_error):
    V, B = 1024, 2
    g = torch.Generator().manual_seed(32)
    x = torch.randn(B * T, V, generator=g)
    mask = SM.all_ones(B, T, words=V // 32)
    params = {
        "temperature": torch.ones(B),
        "top_k": torch.full((B,), 5),
        "top_p": torch.ones(B),
        "greedy": torch.zeros(B, dtype=torch.bool),
    }
    tok = torch.zeros(B, T, dtype=torch.int64)
    nv = torch.zeros(B, dtype=torch.int64)
    uni = torch.rand(B, K + 2, generator=g)
    a = host_redo_walk(x, tok, params, uni, B=B, K=K, num_valid=nv)
    b = host_redo_walk(x, tok, params, uni, B=B, K=K, num_valid=nv, mask=mask)  # all ones = no mask
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    mask[1, 0] = 0
    with expect_error(ValueError, "no token"):
        host_redo_walk(x, tok, params, uni, B=B, K=K, num_valid=nv, mask=mask)
    with expect_error(ValueError, "rows"):
        host_redo_walk(x, tok, params, uni, B=B, K=K, num_valid=nv, mask=mask[:1])


def test_subset_redo_merge_with_a_mask():
    V, B = 2048, 4
    g = torch.Generator().manual_seed(41)
    c = _case(g, B, V, ["structure", "dense", "low"])
    c["nv"] = torch.full((B,), K)
    params = {
        "temperature": c["temp"],
        "top_k": c["top_k"],
        "top_p": c["top_p"],
        "greedy": torch.tensor([False, False, True, False]),
    }
    torch.manual_seed(7)
    uni = predraw_uniforms({}, B, K, c["nv"])
    kw = dict(B=B, K=K, num_valid=c["nv"], mask=c["mask"])
    full_c, full_n = host_redo_walk(c["x"], c["tok"], params, uni, **kw)
    sub_c, sub_n = host_redo_walk(c["x"], c["tok"], params, uni, users=[1, 3], **kw)
    for u in (1, 3):  # the subset walk equals the full walk for its users
        assert torch.equal(sub_c[u], full_c[u]) and int(sub_n[u]) == int(full_n[u])
    dev_c = torch.full((B, T), 7, dtype=torch.int32)
    dev_n = torch.full((B,), 2, dtype=torch.int32)
    mc, mn = merge_redo(dev_c, dev_n, sub_c, sub_n, [1, 3])
    for u in range(B):
        if u in (1, 3):
            assert torch.equal(mc[u], full_c[u]) and int(mn[u]) == int(full_n[u])
        else:
            assert torch.equal(mc[u], dev_c[u]) and int(mn[u]) == 2


# --------------------------------------------------------------------------- #
# masked fused_sampled_reference == masked host_redo_walk on pass rows
# --------------------------------------------------------------------------- #
def test_masked_reference_equals_the_masked_host_walk_on_pass_rows():
    per, B = 512, 4
    V = TP * per
    g = torch.Generator().manual_seed(51)
    n_users = n_over = n_diff = 0
    for case in range(120):
        c = _case(g, B, V, ["dense", "structure", "top", "low"])
        x, allowed = c["x"], c["allowed"]
        greedy_masked = torch.rand(B, generator=g) < 0.25  # sent as sampled top_k 1 / top_p 0 / T 1 on the device
        top_k = torch.where(greedy_masked, torch.ones_like(c["top_k"]), c["top_k"])
        top_p = torch.where(greedy_masked, torch.zeros(B), c["top_p"])
        temp = torch.where(greedy_masked, torch.ones(B), c["temp"])
        torch.manual_seed(case)
        uni = predraw_uniforms({}, B, K, c["nv"])
        ref = _small_ref(
            x,
            allowed,
            B=B,
            per=per,
            tok=c["tok"],
            nv=c["nv"],
            greedy=torch.zeros(B, dtype=torch.bool),
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            uni=uni,
            g=g,
        )
        # the same union without the mask= argument (the device edit already happened): identical result
        ref0 = _small_ref(
            x, allowed, B=B, per=per, tok=c["tok"], nv=c["nv"], greedy=torch.zeros(B, dtype=torch.bool), temp=temp,
            top_k=top_k, top_p=top_p, uni=uni, g=g, mask=False,
        )  # fmt: skip
        assert torch.equal(ref.committed, ref0.committed) and torch.equal(ref.counts, ref0.counts)
        assert torch.equal(ref.overflow, ref0.overflow)
        params = {"temperature": temp, "top_k": top_k, "top_p": top_p, "greedy": greedy_masked}
        redo_c, redo_n = host_redo_walk(x, c["tok"], params, uni, B=B, K=K, num_valid=c["nv"], mask=c["mask"])
        for u in range(B):
            if int(ref.overflow[u]):
                n_over += 1
                continue
            n_users += 1
            same = torch.equal(ref.committed[u], redo_c[u]) and int(ref.counts[u]) == int(redo_n[u])
            n_diff += int(not same)
    print(f"\n[masked ref] users={n_users} overflow-skipped={n_over} differing={n_diff}")
    assert n_users > 300
    assert n_diff <= 2, "only float32 boundary classes (|p - u| < 1e-6) may differ"


def test_masked_reference_applies_the_mask_to_an_unmasked_union_that_contains_the_allowed_tokens():
    """A union read back BEFORE any edit, whose candidates include every allowed token: mask= edits the candidates the same way."""
    per, B = 512, 2
    V = TP * per
    g = torch.Generator().manual_seed(52)
    for _ in range(20):
        x = (torch.randn(B * T, V, generator=g) * 2).to(torch.bfloat16)
        allowed = torch.zeros(B * T, V, dtype=torch.bool)
        for r in range(B * T):  # allowed tokens among the top-4 of their shard, so they are union candidates
            for s in range(TP):
                if int(torch.randint(0, 3, (), generator=g)) == 0:
                    allowed[r, s * per + int(torch.topk(x[r, s * per : (s + 1) * per].float(), 4).indices[0])] = True
            if not bool(allowed[r].any()):
                allowed[r, int(torch.topk(x[r].float(), 1).indices[0])] = True
        nv = torch.full((B,), K)
        tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), torch.randint(0, V, (B, K), generator=g)], 1)
        kw = dict(
            B=B,
            K=K,
            per_shard=per,
            M=M,
            greedy=torch.zeros(B, dtype=torch.bool),
            temperature=torch.ones(B),
            top_k=torch.full((B,), 20),
            top_p=torch.ones(B),
            uniforms=torch.rand(B, K + 2, generator=g),
            num_valid=nv,
        )
        aux, pt = _aux(B, nv, g), torch.randint(1, 4000, (B, NB), generator=g, dtype=torch.int32)
        ids = torch.zeros(B * T, dtype=torch.int64)
        raw_v, raw_l = _union(x, per)
        a = fused_sampled_reference(
            raw_v, raw_l, tok.reshape(-1), ids, aux, pt, mask=SM.pack_allowed(allowed).reshape(B, T, -1), **kw
        )
        mv, ml = _union(_device_mask(x, allowed), per)
        b = fused_sampled_reference(mv, ml, tok.reshape(-1), ids, aux, pt, **kw)
        assert torch.equal(a.committed, b.committed) and torch.equal(a.counts, b.counts)
        assert torch.equal(a.cand_ids[:, :1], b.cand_ids[:, :1])


# --------------------------------------------------------------------------- #
# H4-lite: the composed mask path equals the plugin accept_sampled_drafts
# --------------------------------------------------------------------------- #
try:
    from vllm_tt_plugin.spec_accept import SpecSamplingInputs, SpecTokenFilters, accept_sampled_drafts

    _HAVE_PLUGIN = True
except ImportError:  # pragma: no cover
    _HAVE_PLUGIN = False


@pytest.mark.skipif(not _HAVE_PLUGIN, reason="vllm_tt_plugin not importable")
def test_composed_mask_path_equals_the_plugin_accept_sampled_drafts():
    """Plugin: grammar bitmask [B, 1+K, W] + allowed_token_ids + min_tokens as filters, full logits, its own uniforms.
    Here: ONE composed packed mask (the AND), host_redo_walk(mask=) and the masked fused_sampled_reference on the union of
    the device-masked logits, with the same uniforms (predraw_uniforms from the same seed). Greedy rows and sampled rows.
    """
    per, B = 512, 4
    V = TP * per
    g = torch.Generator().manual_seed(61)
    n_cmp = bad_redo = bad_ref = n_ref = 0
    for case in range(150):
        c = _case(g, B, V, ["dense", "structure", "top", "low"], bf16=True)
        x = c["x"]
        grammar = c[
            "mask"
        ].clone()  # grammar rows; some rows and users unconstrained (scheduler padding / reasoning phase)
        free = torch.rand(B, T, generator=g) < 0.3
        grammar[free] = -1
        # allowed_token_ids (per user, every column) and min_tokens (stop ids, columns j < remaining)
        not_allowed = torch.zeros(B, V, dtype=torch.bool)
        has_allowed = torch.rand(B, generator=g) < 0.3
        for u in range(B):
            if bool(has_allowed[u]):
                keep = torch.randperm(V, generator=g)[: V // 2]
                not_allowed[u] = True
                not_allowed[u, keep] = False
        stops = {
            u: (int(torch.randint(0, 4, (), generator=g)), [int(s) for s in torch.randint(0, V, (3,), generator=g)])
            for u in range(B)
            if int(torch.randint(0, 2, (), generator=g))
        }
        composed = SM.unpack_allowed(grammar, V) & ~not_allowed[:, None, :]
        for u, (rem, ids) in stops.items():
            composed[u, : min(rem, T), ids] = False
        # a composed row with no allowed token goes to logits mode (the engine never sees it): such a user keeps only its grammar
        # row, in the plugin inputs and in the composed mask alike
        dead = ~composed.any(-1)
        plug_not_allowed, plug_stops = not_allowed.clone(), dict(stops)
        for u in range(B):
            if bool(dead[u].any()):
                plug_not_allowed[u] = False
                plug_stops.pop(u, None)
                composed[u] = SM.unpack_allowed(grammar, V)[u]
        grammar_eff = SM.pack_allowed(composed)
        greedy = torch.rand(B, generator=g) < 0.3
        temp = torch.where(greedy, torch.zeros(B), c["temp"])
        filters = SpecTokenFilters(
            grammar_bitmask=grammar,
            allowed_token_ids_mask=plug_not_allowed if bool(plug_not_allowed.any()) else None,
            min_tokens=plug_stops,
        )
        # the plugin and the composed mask must agree on the rows
        sampling = SpecSamplingInputs(
            vocab_size=V, temperature=temp, top_k=c["top_k"].to(torch.int64), top_p=c["top_p"], filters=filters
        )
        drafts = c["tok"][:, 1:].to(torch.int32)
        torch.manual_seed(case)
        plug = accept_sampled_drafts(x.float().reshape(B, T, V), drafts, c["nv"].to(torch.int32), sampling)
        torch.manual_seed(case)
        uni = predraw_uniforms({}, B, K, c["nv"])
        params = {"temperature": temp, "top_k": c["top_k"], "top_p": c["top_p"], "greedy": greedy}
        got_c, got_n = host_redo_walk(
            x, c["tok"], params, uni, B=B, K=K, num_valid=c["nv"], mask=grammar_eff.reshape(B, T, -1)
        )
        n_cmp += 1
        same = torch.equal(plug.accepted_counts.to(torch.int64), got_n.to(torch.int64)) and all(
            torch.equal(
                plug.committed_token_ids[u, : int(got_n[u])].to(torch.int64), got_c[u, : int(got_n[u])].to(torch.int64)
            )
            for u in range(B)
        )
        bad_redo += int(not same)
        # the masked union reference: greedy rows as sampled top_k 1 / top_p 0 / T 1
        top_k = torch.where(greedy, torch.ones_like(c["top_k"]), c["top_k"])
        top_p = torch.where(greedy, torch.zeros(B), c["top_p"])
        t1 = torch.where(greedy, torch.ones(B), c["temp"])
        allowed_eff = SM.unpack_allowed(grammar_eff, V).reshape(B * T, V)
        ref = _small_ref(
            x,
            allowed_eff,
            B=B,
            per=per,
            tok=c["tok"],
            nv=c["nv"],
            greedy=torch.zeros(B, dtype=torch.bool),
            temp=t1,
            top_k=top_k,
            top_p=top_p,
            uni=uni,
            g=g,
        )
        for u in range(B):
            if int(ref.overflow[u]):
                continue
            n_ref += 1
            ok = torch.equal(plug.accepted_counts[u].to(torch.int64), ref.counts[u].to(torch.int64)) and torch.equal(
                plug.committed_token_ids[u, : int(ref.counts[u])].to(torch.int64),
                ref.committed[u, : int(ref.counts[u])].to(torch.int64),
            )
            bad_ref += int(not ok)
    print(f"\n[H4-lite] cases={n_cmp} redo-mismatch={bad_redo} union-ref users={n_ref} mismatch={bad_ref}")
    assert bad_redo <= 2 and bad_ref <= 3, "only float32 / uniform rounding boundaries may differ"


# --------------------------------------------------------------------------- #
# engine plumbing on a stub engine (no device)
# --------------------------------------------------------------------------- #
def _stub_engine(B, V, per=512):
    eng = object.__new__(MTPSpecEngine)
    eng.__dict__.update(B=B, K=K, T=T, vocab=V, _fs=SimpleNamespace(per_shard=per, M=M, TP=TP), _phys_perm=None)
    eng._fused_note = lambda reason: None
    return eng


def test_engine_host_walk_step_and_fallback_apply_the_step_mask():
    V, B = 2048, 3
    g = torch.Generator().manual_seed(71)
    c = _case(g, B, V, ["structure", "dense", "low"])
    c["nv"] = torch.full((B,), K)
    greedy = torch.tensor([False, True, False])
    sampling = {"temperature": c["temp"], "top_k": c["top_k"], "top_p": c["top_p"], "greedy": greedy}
    torch.manual_seed(3)
    sampling["uniforms"] = predraw_uniforms({}, B, K, c["nv"])
    eng = _stub_engine(B, V)
    want = host_redo_walk(c["x"], c["tok"], sampling, sampling["uniforms"], B=B, K=K, num_valid=c["nv"], mask=c["mask"])
    unmasked = host_redo_walk(c["x"], c["tok"], sampling, sampling["uniforms"], B=B, K=K, num_valid=c["nv"])
    assert not (
        torch.equal(want[0], unmasked[0]) and torch.equal(want[1], unmasked[1])
    )  # the mask matters in this case
    # explicit argument
    fz = eng._host_walk_step(c["x"], c["tok"], c["nv"], sampling, mask=c["mask"])
    assert torch.equal(fz.committed, want[0]) and torch.equal(fz.counts, want[1])
    # the mask rides in the sampling dict (engine rows)
    fz = eng._host_walk_step(c["x"], c["tok"], c["nv"], dict(sampling, mask=c["mask"]))
    assert torch.equal(fz.committed, want[0]) and torch.equal(fz.counts, want[1])
    # no mask: unchanged behaviour
    fz = eng._host_walk_step(c["x"], c["tok"], c["nv"], sampling)
    assert torch.equal(fz.committed, unmasked[0]) and torch.equal(fz.counts, unmasked[1])
    # the fallback result passes it on
    ids = torch.zeros(B, T, dtype=torch.int32)
    _, drafts, fz = eng._fused_fallback_result(
        ids, c["tok"], c["nv"], "top_k_range", logits=c["x"], sampling=sampling, mask=c["mask"]
    )
    assert drafts is None and torch.equal(fz.committed, want[0]) and torch.equal(fz.counts, want[1])
    # subset redo with the mask merges into the device result
    dev = (torch.full((B, T), 9, dtype=torch.int32), torch.full((B,), 1, dtype=torch.int32))
    fz = eng._host_walk_step(c["x"], c["tok"], c["nv"], sampling, users=[2], dev=dev, mask=c["mask"])
    assert torch.equal(fz.committed[2], want[0][2]) and int(fz.counts[2]) == int(want[1][2])
    assert torch.equal(fz.committed[0], dev[0][0]) and int(fz.counts[0]) == 1


def test_engine_audit_reference_accepts_the_mask_and_the_guard(expect_error):
    per, B = 512, 2
    V = TP * per
    g = torch.Generator().manual_seed(72)
    x = (torch.randn(B * T, V, generator=g) * 2).to(torch.bfloat16)
    allowed = torch.zeros(B * T, V, dtype=torch.bool)
    for r in range(B * T):  # fewer allowed tokens than top_k: the guarded tie rule
        allowed[r, torch.randperm(V, generator=g)[:4]] = True
    nv = torch.full((B,), K)
    tok = torch.cat([torch.zeros(B, 1, dtype=torch.int64), torch.randint(0, V, (B, K), generator=g)], 1)
    sampling = {
        "temperature": torch.ones(B),
        "top_k": torch.full((B,), 30),
        "top_p": torch.ones(B),
        "greedy": torch.zeros(B, dtype=torch.bool),
        "uniforms": torch.rand(B, K + 2, generator=g),
        "mask": SM.pack_allowed(allowed).reshape(B, T, -1),
    }
    aux, pt = _aux(B, nv, g), torch.randint(1, 4000, (B, NB), generator=g, dtype=torch.int32)
    uv, ul = _union(_device_mask(x, allowed), per)
    ref = fused_sampled_reference(
        uv,
        ul,
        tok.reshape(-1),
        torch.zeros(B * T, dtype=torch.int64),
        aux,
        pt,
        B=B,
        K=K,
        per_shard=per,
        M=M,
        greedy=sampling["greedy"],
        temperature=sampling["temperature"],
        top_k=sampling["top_k"],
        top_p=sampling["top_p"],
        uniforms=sampling["uniforms"],
        num_valid=nv,
    )
    assert int(ref.overflow.sum()) == 0
    eng = _stub_engine(B, V, per)
    n = eng._audit_sampled_core(
        uv,
        ul,
        torch.zeros(B * T, dtype=torch.int64),
        ref.committed.to(torch.int64),
        ref.counts.to(torch.int64),
        tok,
        nv,
        pt,
        aux,
        sampling,
    )
    assert n == 0
    wrong = ref.committed.to(torch.int64).clone()
    wrong[0, 0] = (wrong[0, 0] + 1) % V
    with expect_error(RuntimeError, "fused sampled accept"):
        eng._audit_sampled_core(
            uv, ul, torch.zeros(B * T, dtype=torch.int64), wrong, ref.counts.to(torch.int64), tok, nv, pt, aux, sampling
        )
