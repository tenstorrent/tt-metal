# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the speculative-decode logic in ``tt/spec_decode.py`` (no device, torch only): the greedy
acceptance, the candidate-major row layout, the commit masks / dead ring rows, the draft-length policy, and the
``GreedySpecLoop`` against a deterministic fake target + drafter, which must reproduce the plain greedy sequence
for ANY drafter accuracy (lossless) and respect the state discipline (a verify never changes the committed
context; only ``commit`` does).

    pytest models/demos/glm53_flash/tests/test_spec_decode_host.py -q
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from models.demos.glm53_flash.tt.spec_decode import (
    MAX_VERIFY_ROWS,
    GreedySpecLoop,
    accept_greedy,
    candidate_rows,
    dead_ring_row,
    kda_commit_masks,
    ring_slots_masked,
    select_onehot,
    select_rows,
    spec_draft_len,
    verify_ids_user_major,
    verify_row_positions,
    verify_tokens_row_major,
)

VOCAB = 97


# ------------------------------------------------------------------------------------------- fake model
def fake_next(ctx: list[int]) -> int:
    """A deterministic 'language model': the next token is a hash of the last three tokens and the length."""
    s = sum((i + 1) * t for i, t in enumerate(ctx[-3:])) + 7 * len(ctx)
    return (s * 31 + 11) % VOCAB


def plain_greedy(prompt: list[int], n: int) -> list[int]:
    ctx = list(prompt)
    out = []
    for _ in range(n):
        t = fake_next(ctx)
        out.append(t)
        ctx.append(t)
    return out


class FakeTarget:
    """The three callables of ``GreedySpecLoop`` over per-user committed contexts. ``verify`` computes every
    candidate row from the committed context + the candidate prefix (and the drafter's level-1 guess after the
    row's greedy id); ``commit`` is the ONLY call that extends the context; ``draft_chain`` chains the drafter
    from the committed context + the bonus token + the level-1 draft. The drafter guesses the true next token
    with probability ``acc`` (else a wrong one)."""

    def __init__(self, prompts: list[list[int]], K: int, acc: float, seed: int = 0):
        self.ctx = [list(p) for p in prompts]
        self.K, self.acc = K, acc
        self.rng = random.Random(seed)
        self.pending: torch.Tensor | None = None
        self.verifies = self.commits = 0

    def draft(self, ctx: list[int]) -> int:
        true = fake_next(ctx)
        if self.rng.random() < self.acc:
            return true
        return (true + 1 + self.rng.randrange(VOCAB - 1)) % VOCAB  # any other token

    def verify(self, tokens: torch.Tensor, positions: list[int]):
        assert self.pending is None, "verify called twice without a commit in between"
        N, P = tokens.shape
        assert P == self.K + 1
        ids = torch.zeros(N, P, dtype=torch.long)
        d1 = torch.zeros(N, P, dtype=torch.long)
        for u in range(N):
            assert positions[u] == len(self.ctx[u]), "the anchor position must equal the committed context length"
            for j in range(P):
                row_ctx = self.ctx[u] + tokens[u, : j + 1].tolist()
                ids[u, j] = fake_next(row_ctx)
                d1[u, j] = self.draft(row_ctx + [int(ids[u, j])])
        self.pending = tokens.clone()
        self.verifies += 1
        return ids, d1

    def commit(self, m: torch.Tensor, positions: list[int]):
        assert self.pending is not None, "commit without a verify"
        for u in range(len(m)):
            assert positions[u] == len(self.ctx[u])
            self.ctx[u] += self.pending[u, : int(m[u]) + 1].tolist()  # the anchor + the accepted drafts
        self.pending = None
        self.commits += 1

    def draft_chain(self, m: torch.Tensor, d1: torch.Tensor, positions: list[int]) -> torch.Tensor:
        N = len(m)
        out = torch.zeros(N, self.K - 1, dtype=torch.long)
        for u in range(N):
            assert positions[u] + int(m[u]) + 1 == len(self.ctx[u]), "the chain follows the commit"
            bonus = fake_next(self.ctx[u])  # = ids[u, m_u] of the verify
            chain = self.ctx[u] + [bonus, int(d1[u])]
            for k in range(2, self.K + 1):
                t = self.draft(chain)
                out[u, k - 2] = t
                chain.append(t)
        return out


def _run(prompts, K, acc, n_tokens, seed=0, eos=None):
    target = FakeTarget(prompts, K, acc, seed)
    anchors = [fake_next(p) for p in prompts]
    loop = GreedySpecLoop(
        len(prompts), K, verify=target.verify, commit=target.commit, draft_chain=target.draft_chain if K > 1 else None
    )
    outs, stats = loop.run(anchors, [len(p) for p in prompts], n_tokens, eos=eos)
    return outs, stats, target


# ------------------------------------------------------------------------------------------------ tests
def test_accept_greedy_cases():
    drafts = torch.tensor([[5, 6, 7], [5, 9, 7], [1, 2, 3], [5, 6, 7]])
    verify = torch.tensor([[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 0, 8]])
    m, committed, bonus = accept_greedy(drafts, verify)
    assert m.tolist() == [3, 1, 0, 2]
    assert committed == [[5, 6, 7, 8], [5, 6], [5], [5, 6, 0]]
    assert bonus.tolist() == [8, 6, 5, 0]
    # placeholder drafts (first iteration): nothing accepted, the bonus is row 0's id
    m0, c0, b0 = accept_greedy(drafts, verify, valid=0)
    assert m0.tolist() == [0, 0, 0, 0] and c0 == [[5], [5], [5], [5]] and b0.tolist() == [5, 5, 5, 5]
    # K = 0: a plain step
    m1, c1, b1 = accept_greedy(torch.zeros(2, 0, dtype=torch.long), torch.tensor([[3], [4]]))
    assert m1.tolist() == [0, 0] and c1 == [[3], [4]] and b1.tolist() == [3, 4]


def test_serving_draft_len_cap_mirrors_spec_draft_len():
    """``tt/vllm_spec.py`` (host-only, no ttnn) carries its own copy of the policy for the adapter's ``spec_plan``."""
    from models.demos.glm53_flash.tt.vllm_spec import MAX_VERIFY_ROWS as SERVING_MAX_ROWS
    from models.demos.glm53_flash.tt.vllm_spec import draft_len_cap

    assert SERVING_MAX_ROWS == MAX_VERIFY_ROWS
    for B in (1, 2, 3, 4, 5, 8, 10, 16, 17, 32, 33):
        for k in (1, 3, 7, 31):
            assert draft_len_cap(B, k) == spec_draft_len(B, k), (B, k)


def test_draft_len_policy():
    assert spec_draft_len(1, 31) == 31 and spec_draft_len(1, 3) == 3
    assert spec_draft_len(4, 7) == 7 and spec_draft_len(4, 3) == 3
    assert spec_draft_len(8, 5) == 3
    assert spec_draft_len(16, 5) == 1
    assert spec_draft_len(32, 5) == 0  # plain decode at 32 users per row
    for B in (1, 2, 4, 8, 16, 32):
        K = spec_draft_len(B, 64)
        assert candidate_rows(B, K) <= MAX_VERIFY_ROWS
        if K < 64:
            with pytest.raises(ValueError):  # allow-pytest.raises: a pure host check, no device error text
                candidate_rows(B, K + 1)


def test_row_layout_round_trip():
    B, K, rows = 2, 3, 2
    N = B * rows
    tokens = torch.arange(N * (K + 1)).reshape(N, K + 1)
    flat = verify_tokens_row_major(tokens, B, rows)
    # mesh row 0: candidate 0 of users 0, 1 then candidate 1 ... ; mesh row 1 the same for users 2, 3
    assert flat[: B * (K + 1)].tolist() == [0, 4, 1, 5, 2, 6, 3, 7]
    assert torch.equal(verify_ids_user_major(flat, B, K, rows), tokens)
    pos = verify_row_positions([10, 20, 30, 40], B, K, rows)
    assert pos == [10, 20, 11, 21, 12, 22, 13, 23, 30, 40, 31, 41, 32, 42, 33, 43]
    m = torch.tensor([0, 3, 1, 2])
    assert select_rows(m, B) == [0, 7, 2, 5]
    sel = select_onehot(m, B, K, rows)
    assert tuple(sel.shape) == (rows, 1, B, B * (K + 1))
    assert sel.sum().item() == N
    assert sel[0, 0, 0, 0] == 1 and sel[0, 0, 1, 7] == 1 and sel[1, 0, 0, 2] == 1 and sel[1, 0, 1, 5] == 1


def test_commit_masks_and_ring_rows():
    K, P = 3, 4
    m = torch.tensor([0, 3, 1, 2])
    take = kda_commit_masks(m, K)
    assert tuple(take.shape) == (K + 1, 4)
    assert take.sum(0).tolist() == [1, 1, 1, 1]  # every user is taken after exactly one launch
    assert [int(take[:, u].argmax()) for u in range(4)] == m.tolist()
    positions = [5, 6, 7, 8]
    for j in range(K + 1):
        slots = ring_slots_masked(positions, m, j, P)
        for u, p in enumerate(positions):
            if j <= int(m[u]):
                assert slots[u] == (p + j) % P
            else:
                assert P <= slots[u] < 32, "a rejected candidate's ring row must be a dead row of the block"
    assert all(P <= dead_ring_row(j, P) < 32 for j in range(32))


@pytest.mark.parametrize("acc", [1.0, 0.0, 0.6])
@pytest.mark.parametrize("K", [1, 2, 3])
def test_loop_is_lossless(acc, K):
    prompts = [[1, 2, 3, 4, 5], [9, 8, 7], [4, 4, 4, 4, 4, 4, 4]]
    n = 40
    outs, stats, target = _run(prompts, K, acc, n, seed=K)
    for u, p in enumerate(prompts):
        ref = plain_greedy(p, n + 1)  # [anchor, continuation...]
        assert outs[u] == ref[1 : n + 1], f"user {u} acc {acc} K {K}: {outs[u]} vs {ref[1:n + 1]}"
    assert target.commits == target.verifies == stats.iterations
    if acc == 1.0:  # the first iteration commits 1 token, every later one K + 1
        assert stats.iterations == 1 + math.ceil((n - 1) / (K + 1))
    if acc == 0.0:  # one token per iteration (the bonus), m = 0 always
        assert stats.iterations == n and all(a == 0 for a in stats.accepted)
    if acc == 0.6:
        assert 1 + math.ceil((n - 1) / (K + 1)) < stats.iterations < n


def test_loop_plain_decode_k0():
    prompts = [[3, 1, 4, 1, 5]]
    outs, stats, _ = _run(prompts, 0, 0.0, 12)
    assert outs[0] == plain_greedy(prompts[0], 13)[1:13]
    assert stats.iterations == 12


def test_loop_eos_and_uneven_users():
    prompts = [[1, 2, 3], [3, 2, 1]]
    ref0 = plain_greedy(prompts[0], 30)
    eos = ref0[9]  # user 0 stops at its 9th continuation token (index 8 of the output)
    outs, stats, _ = _run(prompts, 3, 0.8, 24, seed=5, eos=eos)
    assert outs[0] == ref0[1:10] and outs[0][-1] == eos
    ref1 = plain_greedy(prompts[1], 25)
    assert outs[1] == ref1[1:25]
    assert stats.tokens_per_iteration > 1.0
