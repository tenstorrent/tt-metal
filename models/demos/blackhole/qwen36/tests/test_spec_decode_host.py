# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for the spec-decode pure helpers (no device, no checkpoint)."""

import pytest

from models.demos.blackhole.qwen36.tt.spec_decode import (
    BLOCK,
    block_aligned_prefill_len,
    commit_advance,
    greedy_accept,
)


class TestGreedyAccept:
    def test_all_accepted(self):
        m, committed = greedy_accept([5, 6, 7], [5, 6, 7, 8])
        assert m == 3
        assert committed == [5, 6, 7, 8]  # all drafts + bonus

    def test_first_rejected(self):
        m, committed = greedy_accept([5, 6, 7], [9, 6, 7, 8])
        assert m == 0
        assert committed == [9]  # correction only

    def test_middle_rejected(self):
        m, committed = greedy_accept([5, 6, 7], [5, 2, 7, 8])
        assert m == 1
        assert committed == [5, 2]  # accepted prefix + correction

    def test_always_commits_at_least_one(self):
        for targets in ([1, 1, 1, 1], [0, 0, 0, 0]):
            _, committed = greedy_accept([9, 9, 9], targets)
            assert len(committed) >= 1

    def test_target_len_mismatch_asserts(self):
        with pytest.raises(AssertionError):
            greedy_accept([1, 2], [1, 2])


class TestBlockAlignedPrefillLen:
    @pytest.mark.parametrize(
        "prompt_len,expected",
        [
            (1, 0),
            (2, 0),
            (BLOCK, 0),  # strictly below: block-aligned prompt leaves a full-block tail
            (BLOCK + 1, BLOCK),
            (2 * BLOCK, BLOCK),
            (1000, BLOCK * ((1000 - 1) // BLOCK)),
        ],
    )
    def test_values(self, prompt_len, expected):
        a0 = block_aligned_prefill_len(prompt_len)
        assert a0 == expected
        assert a0 % BLOCK == 0
        # The first verify chunk (the prompt tail) is non-empty and <= one block.
        assert 1 <= prompt_len - a0 <= BLOCK


class TestCommitAdvance:
    @pytest.mark.parametrize(
        "pending_len,expected",
        [
            (1, 0),
            (BLOCK - 1, 0),
            (BLOCK, 0),  # committing all 64 would leave no anchor row — must wait
            (BLOCK + 1, BLOCK),
            (2 * BLOCK, BLOCK),  # leave-one rule: 128 pending commits 64, not 128
            (2 * BLOCK + 1, 2 * BLOCK),
        ],
    )
    def test_values(self, pending_len, expected):
        k = commit_advance(pending_len)
        assert k == expected
        assert k % BLOCK == 0
        assert pending_len - k >= 1  # at least one committed token stays uncommitted


def _simulate_loop(prompt_len, draft_len, accepts_seq, max_new, bucket=128):
    """Replay generate()'s position bookkeeping with forced per-iteration accepts.

    Mirrors Qwen36SpeculativeDecoder.generate exactly (first draft-less verify,
    per-iteration verify chunk, commit_advance) and asserts the invariants the
    device path relies on: non-negative accept row, block-aligned anchor that
    never catches the committed head, and chunk fitting the verify bucket.
    """
    a = block_aligned_prefill_len(prompt_len)
    committed = prompt_len
    # First (draft-less) verify over the prompt tail.
    tail = committed - a
    assert 1 <= tail <= BLOCK
    committed += 1
    out = 1
    a += commit_advance(committed - a)
    for m in accepts_seq:
        if out >= max_new:
            break
        assert 0 <= m <= draft_len
        c = committed - 1
        row_start = c - a
        chunk_len = (c + 1 - a) + draft_len
        assert row_start >= 0, f"anchor overtook the committed head: c={c} a={a}"
        assert chunk_len <= bucket, f"verify chunk {chunk_len} exceeds bucket {bucket}"
        assert row_start + draft_len + 1 <= chunk_len
        committed += m + 1
        out += m + 1
        a += commit_advance(committed - a)
        assert a % BLOCK == 0
        assert a <= committed - 1, f"anchor {a} passed last committed token {committed - 1}"
    return out


class TestLoopInvariants:
    def test_all_rejected_crosses_commit_boundary(self):
        """Silicon regression: T=100, K=3, m=0 every iteration. The 28th token
        lands exactly on the first 64-token commit boundary; the old commit rule
        advanced the anchor past the last committed token there (accept row -1,
        garbage logits -> token id 0)."""
        _simulate_loop(prompt_len=100, draft_len=3, accepts_seq=[0] * 40, max_new=40)

    @pytest.mark.parametrize("prompt_len", [2, 63, 64, 65, 100, 127, 128, 129, 200])
    @pytest.mark.parametrize("pattern", [[0], [3], [0, 1, 2, 3], [1], [2, 0]])
    def test_accept_patterns(self, prompt_len, pattern):
        n_iters = 300 // max(1, sum(m + 1 for m in pattern))
        _simulate_loop(prompt_len, draft_len=3, accepts_seq=pattern * n_iters, max_new=300)

    def test_max_draft_len_worst_case(self):
        _simulate_loop(prompt_len=65, draft_len=31, accepts_seq=[31, 0] * 50, max_new=500)


# --------------------------------------------------------------------------- #
# CPU emulation of the REAL generate() loop: device ops are faked, the
# bookkeeping (pending pairs, accept rows, anchor commits) is the actual code.
# --------------------------------------------------------------------------- #
_V = 97  # toy vocab


def _tok(i):
    """True token at position i of the toy deterministic sequence."""
    return (i * 7 + 3) % _V


def _hid(i):
    """Target post-norm hidden marker for position i (tag 1.0 = catch-up)."""
    import torch

    return torch.tensor([1.0, float(i)])


class _OracleDrafter:
    """Drafter stub that asserts the (hidden[p], token[p+1]) @ position p
    convention on every call and predicts token[p+2] (or a wrong token)."""

    def __init__(self, correct=True):
        import torch

        self._torch = torch
        self.correct = correct
        self.steps = 0

    def step(self, token, hidden, pos):
        torch = self._torch
        self.steps += 1
        if float(hidden[0]) == 1.0:
            # Catch-up pair: true target hidden of `pos`, input token t_{pos+1}.
            assert float(hidden[1]) == pos, f"hidden of position {hidden[1]} fed at drafter pos {pos}"
            assert token == _tok(pos + 1), f"token {token} at drafter pos {pos}, expected t_{pos+1}={_tok(pos+1)}"
        else:
            # Chained step: the drafter's own hidden from the previous position.
            assert float(hidden[0]) == 2.0 and float(hidden[1]) == pos - 1, (hidden, pos)
            if self.correct:
                assert token == _tok(pos + 1)
        pred = _tok(pos + 2) if self.correct else (_tok(pos + 2) + 1) % _V
        logits = torch.zeros(_V)
        logits[pred] = 1.0
        return logits, torch.tensor([2.0, float(pos)])


def _make_fake_spec(draft_len, prompt_len, monkeypatch):
    """Real Qwen36SpeculativeDecoder with device ops faked (no GDN layers, toy
    target whose argmax at any position is the true next token)."""
    from types import SimpleNamespace

    import torch

    import models.demos.blackhole.qwen36.tt.spec_decode as sd_mod

    monkeypatch.setattr(sd_mod, "ttnn", SimpleNamespace(deallocate=lambda *_a, **_k: None))

    class _FakeSpec(sd_mod.Qwen36SpeculativeDecoder):
        def __init__(self, mtp):
            super().__init__(SimpleNamespace(num_devices=1, layers=[]), mtp, None, draft_len=draft_len)
            self._last_chunk = None

        def seed(self):
            """Stand-in for prefill(): state bookkeeping + drafter seeding only."""
            self.committed = [_tok(i) for i in range(prompt_len)]
            self.a = sd_mod.block_aligned_prefill_len(prompt_len)
            self._pending = []
            self.accepts = []
            for i in range(max(0, self.a - self.seed_window), self.a):
                self.mtp.step(self.committed[i + 1], _hid(i), i)

        def _chunk_forward(self, tokens, chunk_start, valid_len):
            assert chunk_start % sd_mod.BLOCK == 0
            assert chunk_start == self.a
            n_committed = len(self.committed) - chunk_start
            assert list(tokens[:n_committed]) == self.committed[chunk_start:], "committed prefix mismatch"
            self._last_chunk = (chunk_start, valid_len)
            return None, 128 * ((len(tokens) + 127) // 128)

        def _extract_rows(self, hidden, bucket, row_start, n, want_logits=True):
            chunk_start, valid_len = self._last_chunk
            assert 0 <= row_start and row_start + n <= valid_len, f"rows [{row_start},{row_start+n}) not processed"
            logits = torch.zeros(n, _V)
            hids = torch.zeros(n, 2)
            for j in range(n):
                pos = chunk_start + row_start + j
                logits[j, _tok(pos + 1)] = 1.0
                hids[j] = _hid(pos)
            return logits, hids

        def _maybe_commit(self):
            self.a += sd_mod.commit_advance(len(self.committed) - self.a)

    return _FakeSpec


@pytest.mark.parametrize("prompt_len", [10, 100, 128])
def test_loop_emulation_oracle_drafter_full_accept(prompt_len, monkeypatch):
    """A drafter that predicts the target exactly must be fully accepted —
    proves the accept comparison and the (hidden, token, position) pairing."""
    K = 3
    fake_cls = _make_fake_spec(K, prompt_len, monkeypatch)
    spec = fake_cls(_OracleDrafter(correct=True))
    spec.seed()
    out, stats = spec.generate(50)
    assert out == [_tok(prompt_len + i) for i in range(len(out))]
    assert len(out) >= 50
    assert all(m == K for m in spec.accepts), spec.accepts


def test_loop_emulation_wrong_drafter_still_matches_target(monkeypatch):
    """Silicon regression at the real-code level: an always-wrong drafter over
    40 tokens (crossing the 64-token commit boundary at token 28 for a
    100-token prompt) must still emit exactly the target sequence."""
    fake_cls = _make_fake_spec(3, 100, monkeypatch)
    spec = fake_cls(_OracleDrafter(correct=False))
    spec.seed()
    out, stats = spec.generate(40)
    assert out == [_tok(100 + i) for i in range(len(out))]
    assert len(out) >= 40
    assert all(m == 0 for m in spec.accepts)
