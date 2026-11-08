# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the mask-hole cap (spec_masks.mask_hole_cap, MTPSpecEngine._mask_cap_nv): a masked step never walks into an
all-ones mask column it reaches through a constrained one.

The bug it fixes (yyz2 d30u20, JSONSchemaBench, QWEN36_SPEC_DEVICE_MASK=1: about 3.6% of structured requests terminated by vLLM
with "grammar rejected tokens"): vLLM fills a structured request's scheduler rows 1..K from the drafts only on its deferred
sampling path (``take_draft_token_ids``, when a structured request has output placeholders in flight). Otherwise the bitmask is
computed at once from the ``-1`` placeholders: row 0 is the grammar, rows 1..K are all ones. The plugin still builds a fused
masked step that verifies the real drafts, so the device accepted draft 1 under row 0 and then took columns 1..K from all-ones
rows: a token the grammar forbids, committed (device audit: every committed token was allowed by its own mask row; the
failing rows had con = [1, 0, 0, 0] with 3 drafts). Fake device rigs of the mask tests, no device.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_mask_hole_host.py
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import (
    S_V,
    K,
    T,
    _block,
    _pt,
    _sampling,
    _smooth_logits,
)
from models.demos.blackhole.qwen36.tests.test_spec_mask_chain_engine_host import _rig as _chain_rig
from models.demos.blackhole.qwen36.tests.test_spec_mask_chain_engine_host import _SplitPipe
from models.demos.blackhole.qwen36.tests.test_spec_mask_engine_host import (
    ONES,
    _begin,
    _mask,
    _MaskDefer,
    _MaskSync,
)
from models.demos.blackhole.qwen36.tt import spec_masks as SM
from models.demos.blackhole.qwen36.tt.spec_masks import REASONING_END_IDS, mask_hole_cap, reasoning_boundary_cap

END = REASONING_END_IDS[0]  # "</think>"


# --------------------------------------------------------------------------- #
# a. the pure rule
# --------------------------------------------------------------------------- #
def _m(B, rows):
    """[B, T, W] engine-row mask: all ones except ``rows`` {(u, j): allowed ids}."""
    m = SM.all_ones(B, T)
    for (u, j), ids in rows.items():
        a = torch.zeros(SM.VOCAB, dtype=torch.bool)
        a[list(ids)] = True
        m[u, j] = SM.pack_allowed(a)
    return m


TOK = torch.tensor([[1, 10, 11, 12]])  # column 0 the committed token, drafts 10, 11, 12


@pytest.mark.parametrize(
    "rows, nv, want",
    [
        ({(0, 0): [10, 99]}, 3, 0),  # placeholder rows 1..3: d1 allowed by row 0, row 1 all ones -> no draft verified
        ({(0, 0): [99]}, 3, 3),  # d1 refused by row 0: the walk stops at column 0 anyway
        ({(0, 0): [10], (0, 1): [11]}, 3, 1),  # two real rows, a hole at column 2 after an allowed d2
        ({(0, 0): [10], (0, 1): [11], (0, 2): [12], (0, 3): [5]}, 3, 3),  # every column real
        ({(0, 0): [10], (0, 1): [7]}, 3, 3),  # d2 refused at column 1: no hole is reached
        ({(0, 2): [12], (0, 3): [5]}, 3, 3),  # open reasoning phase in columns 0..1, then the grammar: untouched
        ({(0, 1): [11]}, 3, 1),  # reasoning ends at d1, grammar row 1 allows d2, row 2 a hole
        ({}, 3, 3),  # no constraint
        ({(0, 0): [10, 99]}, 0, 0),  # no draft: nothing to cap
        ({(0, 0): [10], (0, 1): [11]}, 1, 1),  # the hole lies past the verified drafts
    ],
)
def test_mask_hole_cap_rule(rows, nv, want):
    got = mask_hole_cap(_m(1, rows), TOK, torch.tensor([nv]))
    assert got.tolist() == [want]


def test_mask_hole_cap_is_per_row_and_never_raises_nv():
    B = 3
    tok = torch.tensor([[1, 10, 11, 12], [2, 20, 21, 22], [3, 30, 31, 32]])
    m = _m(B, {(0, 0): [10], (2, 0): [30], (2, 1): [31], (2, 2): [32]})
    got = mask_hole_cap(m, tok, torch.tensor([3, 2, 3]))
    assert got.tolist() == [0, 2, 2]
    again = mask_hole_cap(m, tok, got)
    assert torch.equal(again, got)  # idempotent


@pytest.mark.parametrize(
    "tok, rows, nv, want",
    [
        ([1, 10, END, 12], {}, 3, 1),  # "</think>" drafted at column 2, column 2 all ones: commit up to the end id
        ([1, END, 11, 12], {}, 3, 0),
        ([1, 10, 11, END], {}, 3, 2),  # the end id is the last draft: the bonus after it would be unconstrained
        ([1, 10, END, 12], {(0, 2): [12]}, 3, 3),  # the scheduler saw the drafts: column 2 is the grammar's start state
        ([1, 10, END, 12], {}, 1, 1),  # the end id is past the verified drafts
        ([1, 10, 11, 12], {}, 3, 3),  # no end id
        ([END, 10, 11, 12], {}, 3, 3),  # column 0 (the committed token) is not a draft
    ],
)
def test_reasoning_boundary_cap_rule(tok, rows, nv, want):
    mask = _m(1, rows) if rows else None
    got = reasoning_boundary_cap(mask, torch.tensor([tok]), torch.tensor([nv]))
    assert got.tolist() == [want]


# --------------------------------------------------------------------------- #
# b. the engine: the placeholder case on every masked step path
# --------------------------------------------------------------------------- #
def _greedy_case(rig, monkeypatch, B=3, seed=5):
    """All-greedy rows whose drafts are the argmax of each column (the device accepts every one of them unmasked). User 0 has
    the placeholder mask of vLLM's non-deferred bitmask: column 0 allows d1 (and a decoy), columns 1..K are all ones."""
    r = rig(monkeypatch, B)
    logits = _smooth_logits(B * T, seed)
    tok, pos, nvd = _block(B, [3, 3, 3])
    am = logits.float().argmax(dim=-1).reshape(B, T)
    tok[:, 1:] = am[:, :K]
    ss = _sampling(B, nvd, [0.0] * B, topk=3, seed=seed)
    mask = _mask(B, {(0, 0): [int(am[0, 0]), (int(am[0, 0]) + 1) % S_V]})
    return r, logits, tok, pos, nvd, ss, mask


def _check(res_ids, res_counts, tok, mask):
    """User 0 commits exactly one token, allowed by its column-0 mask row; users 1 and 2 (unconstrained) keep their walk."""
    assert int(res_counts[0]) == 1, (res_ids, res_counts)
    allowed = SM.unpack_allowed(mask[0, 0:1, :2], S_V)[0]
    assert bool(allowed[int(res_ids[0, 0])])
    assert int(res_counts[1]) == 4 and int(res_counts[2]) == 4  # drafts = argmax: every draft accepted + the bonus
    assert torch.equal(res_ids[1:, :3], tok[1:, 1:])


def test_deferred_split_host_staged_step_caps_a_placeholder_row(monkeypatch):
    """The prod path (async plugin, a masked step that is not chained): decode_forward_begin(defer_read=True) + finish."""
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _greedy_case(_MaskDefer, monkeypatch, B)
    r.queue = [("logits", logits)]
    h = _begin(r, tok, pos, nvd, ss, defer=True).handle
    n_aux = sum(1 for x in r.log if x[:2] == ("h2d", "fa_aux"))
    r.e.decode_forward_finish(h, mask)
    ids, counts, _ = h.wait()
    _check(ids, counts, tok, mask)
    # the aux words were staged again with the capped num_valid, before the accept trace
    assert sum(1 for x in r.log if x[:2] == ("h2d", "fa_aux")) == n_aux + 1
    assert h.nvv.tolist() == [0, 3, 3] and r.e.mask_hole_caps == 1


def test_sync_split_step_caps_a_placeholder_row(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _greedy_case(_MaskSync, monkeypatch, B)
    r.set_logits(logits)
    h = _begin(r, tok, pos, nvd, ss).handle
    out = r.e.decode_forward_finish(h, mask)
    _check(out.accepted_token_ids, out.accepted_counts, tok, mask)
    assert r.e.mask_hole_caps == 1


def test_one_call_masked_step_caps_a_placeholder_row(monkeypatch):
    B = 3
    r, logits, tok, pos, nvd, ss, mask = _greedy_case(_MaskSync, monkeypatch, B)
    r.set_logits(logits)
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, "mask": mask})
    _check(res.accepted_token_ids, res.accepted_counts, tok, mask)
    assert r.e.mask_hole_caps == 1


def test_a_full_mask_is_not_capped(monkeypatch):
    """Every verified column real (the deferred-sampling path with the drafts handed over): no cap, every draft accepted."""
    B = 3
    r, logits, tok, pos, nvd, ss, _ = _greedy_case(_MaskSync, monkeypatch, B)
    r.set_logits(logits)
    mask = _mask(B, {(0, j): [int(tok[0, j + 1]) if j < K else 3, 1] for j in range(T)})
    res = r.step(tok, pos, nvd, ONES(B), _pt(B), {**ss, "mask": mask})
    assert int(res.accepted_counts[0]) == 4 and r.e.mask_hole_caps == 0


def test_chained_split_step_counts_holes_without_restaging(monkeypatch):
    """A chained split step's aux is built on the device by the chain trace: the cap is not applied, the hole is counted (the
    plugin hands a chained grammar row's drafts over, so its scheduler rows follow them)."""
    B = 3
    r = _chain_rig(monkeypatch, B, 2)
    p = _SplitPipe(r, B, [3, 2, 1])
    h0 = p.enq(None, sampling=_sampling(B, torch.full((B,), K), [0.8, 0.0, 1.2], topk=3, seed=1))
    h1 = p.begin(h0, sampling=_sampling(B, torch.full((B,), K), [0.8, 0.0, 1.2], topk=3, seed=2))
    del r.log[:]
    p.finish(h1, _mask(B, {(0, 0): range(1, S_V)}))  # column 0 allows all but token 0: columns 1..K all ones
    assert h1.cmask is not None  # set when the second half is enqueued
    assert ("h2d", "fa_aux") not in [x[:2] for x in r.log]
    assert r.e.mask_hole_caps == 0 and r.e.mask_hole_chained >= 1


def test_deferred_split_step_caps_a_reasoning_boundary_row(monkeypatch):
    """The run's last rejection: a thinking row drafts "</think>" then a token the grammar's start state refuses; vLLM computed
    its rows from placeholders, so every row is all ones (reasoning "still open"). The split step commits at most up to the end
    id. Row 1 (no end id) keeps its drafts."""
    B = 3
    r, logits, tok, pos, nvd, ss, _ = _greedy_case(_MaskDefer, monkeypatch, B)
    r.queue = [("logits", logits)]
    tok[0, 2] = END
    h = _begin(r, tok, pos, nvd, ss, defer=True).handle
    r.e.decode_forward_finish(h, _mask(B, {(2, 0): [int(tok[2, 1]), 1]}))  # user 2 constrained: a masked step
    ids, counts, _ = h.wait()
    assert h.nvv.tolist()[:2] == [1, 3] and int(counts[0]) <= 2
    assert END not in ids[0, : int(counts[0]) - 1].tolist()  # the end id can only be the step's last token


# --------------------------------------------------------------------------- #
# c. the staging ring's repack cache (the masked step's host cost)
# --------------------------------------------------------------------------- #
def test_perm_staging_ring_cache_equals_the_repack():
    g = torch.Generator().manual_seed(7)
    base = torch.randint(-(2**31), 2**31 - 1, (6, SM.MASK_WORDS), generator=g, dtype=torch.int64).to(torch.int32)
    ring = SM.PermMaskStagingRing(cache_rows=4)
    for step in range(40):
        rows = base[torch.randint(0, 6, (12,), generator=g)]  # repeated rows, more distinct rows than the cache holds
        idx = sorted(set(torch.randint(0, 12, (7,), generator=g).tolist()))
        buf = ring.stage(rows, idx).clone()
        want = torch.full((SM.SHARDS, SM.MASK_ROWS, SM.SHARD_WORDS), -1, dtype=torch.int32)
        want[:, idx] = SM.repack_to_permuted(rows[idx]).permute(1, 0, 2)
        assert torch.equal(buf, want), step
    assert ring.hits > 0 and ring.misses > 0 and len(ring.cache) <= 4
