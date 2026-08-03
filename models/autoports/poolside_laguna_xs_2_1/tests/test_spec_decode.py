# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Laguna ngram speculative decoding.

Two tiers:
  * HOST (no device, always run): validate the ngram proposer + greedy/sampling
    accept + generate loop against a deterministic Markov-map oracle stub. The
    stub's ``verify_forward`` returns one-hot logits at ``M[input_token]``, so the
    ground-truth greedy sequence is the M-orbit of the prompt; the test asserts
    spec-decode output == plain-greedy output token-for-token (the correctness
    contract) and that acceptance is non-trivial (ngram hits on the periodic orbit).
  * DEVICE (skipped unless LAGUNA_SPEC_DEVICE=1): drives the real Laguna generator
    (see the standalone driver in doc/vllm_integration/scripts/spec_decode_driver.py).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tt.spec_decode import NgramProposer, SpeculativeDecoder, plain_greedy_via_verify  # noqa: E402


# ── ngram proposer ────────────────────────────────────────────────────────
def test_ngram_basic_suffix_match():
    p = NgramProposer(min_n=1, max_n=3)
    # "a b c ... a b" -> after suffix [a,b] the earlier match predicts [c, ...]
    ctx = [5, 1, 2, 3, 9, 1, 2]
    assert p.propose(ctx, 3) == [3, 9, 1]  # earlier [1,2] was followed by 3,9,1


def test_ngram_prefers_longest_suffix():
    p = NgramProposer(min_n=1, max_n=3)
    # suffix [7,1,2] occurs earlier -> follow it; not the shorter [1,2]
    ctx = [7, 1, 2, 4, 0, 7, 1, 2]
    assert p.propose(ctx, 2) == [4, 0]


def test_ngram_no_match_returns_empty():
    p = NgramProposer(min_n=2, max_n=3)
    assert p.propose([1, 2, 3], 3) == []  # no earlier repeat of a >=2 suffix


def test_ngram_truncates_to_k():
    p = NgramProposer(min_n=1, max_n=2)
    ctx = [1, 2, 3, 4, 5, 1, 2]
    assert p.propose(ctx, 2) == [3, 4]


# ── accept logic ──────────────────────────────────────────────────────────
def _onehot(tok, vocab):
    v = torch.full((vocab,), -30.0)
    v[tok] = 10.0
    return v


def test_accept_greedy_all_match():
    drafts = [3, 4, 5]
    vl = [_onehot(t, 8) for t in [3, 4, 5, 6]]  # target greedy = drafts + bonus 6
    m, committed = SpeculativeDecoder._accept_greedy(drafts, vl)
    assert m == 3 and committed == [3, 4, 5, 6]


def test_accept_greedy_reject_at_1():
    drafts = [3, 9, 5]
    vl = [_onehot(t, 12) for t in [3, 4, 5, 6]]  # slot1: target=4 != draft 9 -> reject
    m, committed = SpeculativeDecoder._accept_greedy(drafts, vl)
    assert m == 1 and committed == [3, 4]  # accept d0=3, correct with target greedy 4


def test_accept_greedy_reject_immediately():
    drafts = [99, 1, 2]
    vl = [_onehot(t, 100) for t in [3, 4, 5, 6]]
    m, committed = SpeculativeDecoder._accept_greedy(drafts, vl)
    assert m == 0 and committed == [3]


# ── end-to-end loop vs plain greedy (Markov oracle stub) ────────────────────
class _MarkovOracleGen:
    """Stub generator: verify_forward returns one-hot logits at M[input_token].

    Row j predicts M[verify_tokens[j]] — the deterministic greedy next token. So
    plain-greedy and spec-decode both trace the M-orbit of the prompt and MUST
    agree token-for-token regardless of what ngram proposes."""

    def __init__(self, vocab, a=7, b=3):
        self.vocab = vocab
        self.M = [(a * i + b) % vocab for i in range(vocab)]

    def verify_forward(self, tokens, start_pos, logit_rows=None, **kwargs):
        toks = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        rows = range(len(toks)) if logit_rows is None else logit_rows
        # row r predicts M[input token at r] (deterministic greedy oracle)
        return torch.stack([_onehot(self.M[int(toks[r])], self.vocab) for r in rows], dim=0)

    def verify_forward_decode(self, tokens, positions, **kwargs):
        # batched-decode verify (logits): row j predicts M[tokens[j]] (same oracle, candidates in batch dim)
        toks = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        return torch.stack([_onehot(self.M[int(t)], self.vocab) for t in toks], dim=0)

    def verify_greedy_decode(self, tokens, positions, traced=True, **kwargs):
        # greedy batched-decode verify (ids): row j = argmax = M[tokens[j]]
        toks = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        return torch.tensor([self.M[int(t)] for t in toks], dtype=torch.int32)


@pytest.mark.parametrize("traced", [False, True])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_spec_matches_plain_greedy(K, mode, traced):
    if traced and mode != "decode":
        pytest.skip("traced only applies to decode verify")
    vocab = 53  # prime -> M is a permutation; orbit is periodic -> ngram hits
    gen = _MarkovOracleGen(vocab)
    prompt = [1, 2, 3]
    n = 60

    ref, _ = plain_greedy_via_verify(gen, prompt, n, verify_mode=mode)
    spec = SpeculativeDecoder(gen, draft_len=K, verify_mode=mode, traced=traced)
    got, accepts = spec.generate(prompt, n)

    assert got == ref, f"K={K}: spec diverged from greedy\n ref={ref}\n got={got}"
    assert len(got) == n
    # Sanity: the M-orbit is periodic, so once ngram has one period of history it
    # should accept multiple drafts per iter for K>1 (speedup exists).
    if K > 1 and len(accepts) > 5:
        assert max(accepts) >= 1, f"K={K}: ngram never accepted a draft (accepts={accepts})"


def test_spec_matches_greedy_with_stop_token():
    vocab = 53
    gen = _MarkovOracleGen(vocab)
    prompt = [1, 2, 3]
    # compute the orbit to find a real stop token partway through
    ref, _ = plain_greedy_via_verify(gen, prompt, 40)
    stop_tok = ref[15]
    ref_stop, _ = plain_greedy_via_verify(gen, prompt, 40, stop_tokens={stop_tok})
    spec = SpeculativeDecoder(gen, draft_len=4, stop_tokens={stop_tok})
    got, _ = spec.generate(prompt, 40)
    assert got == ref_stop
    assert got[-1] == stop_tok  # halted exactly at the stop token


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
