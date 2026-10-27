# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only: the pup's draft sources (``pup_spec_drafters``) against a brute-force prompt-lookup reference."""

import random

import pytest

from models.tt_dit.pipelines.flux3_klein.pup_spec_drafters import MtpDrafter, NGramDrafter, NoDrafter, make_drafter


def _brute(ctx, k, ns, exclude=frozenset(), prompt_len=None):
    """Longest n first; most recent earlier occurrence whose continuation exists (i <= L-2); n-grams / continuations
    with excluded ids never match; the proposal stops at the first excluded id."""
    L = len(ctx)
    for n in ns:
        if L < n:
            continue
        suffix = ctx[L - n :]
        for i in range(L - 2, n - 2, -1):
            if prompt_len is not None and i > prompt_len - 2:
                continue
            key = ctx[i - n + 1 : i + 1]
            if key != suffix or ctx[i + 1] in exclude or any(t in exclude for t in key):
                continue
            out = []
            for t in ctx[i + 1 : i + 1 + k]:
                if t in exclude:
                    break
                out.append(t)
            return out
    return []


@pytest.mark.parametrize("n_max, n_min", [(4, 2), (4, 1), (3, 3)])
@pytest.mark.parametrize("source", ["all", "prompt"])
def test_ngram_matches_brute_force(n_max, n_min, source):
    rnd = random.Random(n_max * 10 + n_min + (source == "prompt"))
    exclude = {98, 99}
    for case in range(200):
        k = rnd.choice((1, 3, 7))
        alphabet = rnd.choice((4, 8, 30, 100))
        prompt = [rnd.randrange(alphabet) if rnd.random() > 0.05 else 98 for _ in range(rnd.randrange(1, 80))]
        d = NGramDrafter(k, n_max=n_max, n_min=n_min, exclude_ids=exclude, source=source)
        d.start(prompt)
        ctx = list(prompt)
        for _ in range(30):
            ns = tuple(range(n_max, n_min - 1, -1))
            want = _brute(ctx, k, ns, exclude, len(prompt) if source == "prompt" else None)
            assert d.propose(ctx, [ctx[-1]], len(ctx)) == want, (case, ctx)
            ctx += [rnd.randrange(alphabet) for _ in range(rnd.randrange(1, k + 2))]


def test_ngram_prefers_longest_and_most_recent():
    d = NGramDrafter(3)
    d.start([1, 2, 3, 9, 9, 2, 3, 7, 7, 1, 2, 3, 5, 6])
    # suffix ... 2 3: the 3-gram (1,2,3) last occurred at index 9..11 (continuation 5 6), earlier at 0..2 (9 9)
    assert d.propose([1, 2, 3, 9, 9, 2, 3, 7, 7, 1, 2, 3, 5, 6, 1, 2, 3]) == [5, 6, 1]
    # no unigram fallback by default: a context ending in an unseen pair proposes nothing
    d.start([1, 2, 3])
    assert d.propose([1, 2, 3, 4, 3]) == []
    # with n_min=1 the unigram (4 -> ... 4 3 ... no; 3 -> 4 at index 2) matches
    d = NGramDrafter(2, n_min=1)
    d.start([1, 2, 3, 4])
    assert d.propose([1, 2, 3, 4, 7, 3]) == [4, 7]


def test_ngram_rejects_a_foreign_context(expect_error):
    d = NGramDrafter(2)
    d.start([1, 2, 3])
    with expect_error(ValueError, "not an extension of the indexed one"):
        d.propose([5, 6])


def test_mtp_and_none_drafters(expect_error):
    class Eng:
        K = 3

        def __init__(self):
            self.calls = []

        def propose(self, committed, first_pos):
            self.calls.append((list(committed), first_pos))
            return [4, 5, 6]

    eng = Eng()
    d = make_drafter("mtp", k=3, engine=eng)
    assert isinstance(d, MtpDrafter) and d.propose([0, 1], [7, 8], 42) == [4, 5, 6]
    assert eng.calls == [([7, 8], 42)]
    assert isinstance(make_drafter("none", k=3), NoDrafter) and make_drafter("none", k=3).propose([1], [1], 1) == []
    with expect_error(ValueError, "the mtp drafter needs the spec engine"):
        make_drafter("mtp", k=3)
    with expect_error(ValueError, "unknown drafter"):
        make_drafter("eagle", k=3)
