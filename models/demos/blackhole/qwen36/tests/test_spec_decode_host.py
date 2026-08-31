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


class TestAdaptiveDraftLen:
    def test_clamps_to_range(self):
        from models.demos.blackhole.qwen36.tt.spec_decode import adaptive_draft_len

        assert adaptive_draft_len(0.0, 4) == 1
        assert adaptive_draft_len(10.0, 4) == 4
        assert adaptive_draft_len(0.0, 1) == 1

    @pytest.mark.parametrize("ema,expected", [(0.4, 1), (0.6, 2), (1.4, 2), (1.6, 3), (2.5, 4), (3.0, 4)])
    def test_one_more_than_expected_accepts(self, ema, expected):
        from models.demos.blackhole.qwen36.tt.spec_decode import adaptive_draft_len

        assert adaptive_draft_len(ema, 4) == expected

    def test_emulation_with_adaptive_k(self, monkeypatch):
        """The real generate() under adaptive K still emits exactly the target
        sequence (correctness is K-independent) with a shrunken draft budget."""
        monkeypatch.setenv("TT_SPEC_ADAPTIVE_K", "1")
        fake_cls = _make_fake_spec(4, 100, monkeypatch)
        spec = fake_cls(_OracleDrafter(correct=False))
        spec.seed()
        out, stats = spec.generate(40)
        assert out == [_tok(100 + i) for i in range(len(out))]
        # All-rejected: the EMA collapses and K shrinks to 1.
        assert stats["k_used"][-1] == 1
        assert sum(stats["k_used"]) < 4 * stats["iterations"]


# --------------------------------------------------------------------------- #
# Batched (c8) desync bookkeeping: the REAL Qwen36BatchedSpeculativeDecoder
# generate()/_first_verify() loop with fake device ops and per-user toy targets
# whose drafters accept at DIFFERENT rates, so anchors/commits desync.
# --------------------------------------------------------------------------- #
def _tok_u(u, i):
    """User u's true token at position i (per-user deterministic sequence)."""
    return (i * 7 + 3 + 11 * u) % _V


def _hid_u(u, i):
    import torch

    return torch.tensor([1.0, float(i), float(u)])


class _BatchedOracleDrafter:
    """Per-user drafter stub: user u drafts correctly iff (pos % (u + 1)) == 0,
    so accept rates differ per user and the slots desync. Asserts the
    (hidden[p], token[p+1]) @ position p convention per user."""

    def __init__(self, users):
        import torch

        self._torch = torch
        self.users = users

    def step(self, token, hidden, pos, user=0):
        torch = self._torch
        if float(hidden[0]) == 1.0:
            assert float(hidden[2]) == user, f"user {user} fed user {hidden[2]}'s hidden"
            assert float(hidden[1]) == pos
            assert token == _tok_u(user, pos + 1)
        correct = (pos % (user + 1)) == 0
        pred = _tok_u(user, pos + 2) if correct else (_tok_u(user, pos + 2) + 1) % _V
        logits = torch.zeros(_V)
        logits[pred] = 1.0
        return logits, torch.tensor([2.0, float(pos), float(user)])


def _make_fake_batched_spec(draft_len, prompt_lens, monkeypatch):
    from types import SimpleNamespace

    import torch

    import models.demos.blackhole.qwen36.tt.spec_decode_batched as sb_mod

    monkeypatch.setattr(sb_mod, "ttnn", SimpleNamespace(deallocate=lambda *_a, **_k: None))
    monkeypatch.setenv("TT_SPEC_TRACE", "0")  # the emulation drives the eager loop
    B = len(prompt_lens)

    class _FakeBatched(sb_mod.Qwen36BatchedSpeculativeDecoder):
        def __init__(self, mtp):
            model_stub = SimpleNamespace(num_devices=8, layers=[])
            page_table = torch.arange(B * 4, dtype=torch.int32).reshape(B, 4)
            super().__init__(model_stub, mtp, page_table, draft_len=draft_len)
            self.commit_events = []

        def seed(self):
            for u, T in enumerate(prompt_lens):
                slot = sb_mod.SpecSlot(
                    committed=[_tok_u(u, i) for i in range(T)],
                    a=sb_mod.block_aligned_prefill_len(T),
                )
                slot.k_ema = float(self.draft_len)
                self.slots.append(slot)

        def _snapshot_from_live(self):
            pass

        def _restore_from_snapshot(self):
            pass

        def _batched_verify(self, chunks):
            results = []
            for u, (tokens, a_u, n_u, hid_start, score_start) in enumerate(chunks):
                slot = self.slots[u]
                assert a_u % sb_mod.BLOCK == 0 and a_u == slot.a
                assert n_u == len(tokens) and n_u <= sb_mod._VERIFY_BUCKET
                assert 0 <= hid_start <= score_start < n_u
                n_committed = len(slot.committed) - a_u
                assert list(tokens[:n_committed]) == slot.committed[a_u:], f"user {u} committed prefix mismatch"
                n_score = min(sb_mod._VERIFY_ROWS, n_u - score_start)
                target_ids = [_tok_u(u, a_u + score_start + j + 1) for j in range(n_score)]
                hid = torch.stack([_hid_u(u, a_u + hid_start + j) for j in range(n_u - hid_start)])
                results.append((target_ids, hid))
            return results

        def _commit_user(self, u):
            slot = self.slots[u]
            k = sb_mod.commit_advance(len(slot.committed) - slot.a)
            if k:
                slot.a += k
                self.commit_events.append((u, len(slot.accepts), k))

    return _FakeBatched


def test_batched_loop_desync(monkeypatch):
    """Per-user accept rates differ -> anchors, accepts, and commits desync, and
    every user still emits exactly ITS OWN target sequence."""
    prompt_lens = [100, 70, 130, 65]
    fake_cls = _make_fake_batched_spec(3, prompt_lens, monkeypatch)
    spec = fake_cls(_BatchedOracleDrafter(len(prompt_lens)))
    spec.seed()
    rows, stats = spec.generate(60)
    for u, (row, T) in enumerate(zip(rows, prompt_lens)):
        assert row == [_tok_u(u, T + i) for i in range(len(row))], f"user {u} diverged from its target"
        assert len(row) >= 60
        slot = spec.slots[u]
        assert slot.a % 64 == 0 and slot.a <= len(slot.committed) - 1
    # Desync evidence: differing accept traces and anchors across users.
    assert len({tuple(s.accepts[:10]) for s in spec.slots}) > 1, "accept traces did not desync"
    assert len({s.a for s in spec.slots}) > 1, "anchors did not desync"
    assert spec.commit_events, "no commits fired"


def test_batched_loop_uniform_prompts_stay_identical(monkeypatch):
    """Identical prompts + identical drafter behavior -> identical outputs (the
    on-device batched demo asserts the same row-equality)."""
    prompt_lens = [100, 100, 100]

    class _UniformDrafter(_BatchedOracleDrafter):
        def step(self, token, hidden, pos, user=0):
            # Same accept pattern for every user (cycle of user 0).
            import torch

            if float(hidden[0]) == 1.0:
                assert float(hidden[2]) == user and float(hidden[1]) == pos
            correct = (pos % 2) == 0
            pred = _tok_u(user, pos + 2) if correct else (_tok_u(user, pos + 2) + 1) % _V
            logits = torch.zeros(_V)
            logits[pred] = 1.0
            return logits, torch.tensor([2.0, float(pos), float(user)])

    fake_cls = _make_fake_batched_spec(3, prompt_lens, monkeypatch)
    spec = fake_cls(_UniformDrafter(len(prompt_lens)))
    spec.seed()
    rows, _stats = spec.generate(40)
    base = [t % _V for t in rows[0]]
    for u in range(1, len(prompt_lens)):
        # Same positions + same accept pattern: per-user sequences differ by the
        # 11*u offset but accept/anchor traces must be identical.
        assert spec.slots[u].accepts == spec.slots[0].accepts
        assert spec.slots[u].a == spec.slots[0].a
        assert len(rows[u]) == len(rows[0])
    assert len(base) >= 40


def test_batched_first_verify_arms_pendings(monkeypatch):
    """After the draft-less first verify, every user holds catch-up pairs for its
    whole tail plus the newly sampled token, positioned per user."""
    prompt_lens = [66, 130]
    fake_cls = _make_fake_batched_spec(2, prompt_lens, monkeypatch)
    spec = fake_cls(_BatchedOracleDrafter(len(prompt_lens)))
    spec.seed()
    spec._first_verify()
    for u, slot in enumerate(spec.slots):
        assert slot.out == [_tok_u(u, prompt_lens[u])]
        assert slot.pending[0][2] == spec.slots[u].a if slot.a <= slot.pending[0][2] else True
        positions = [p for (_t, _h, p) in slot.pending]
        assert positions == list(range(positions[0], positions[0] + len(positions)))
        assert positions[-1] == len(slot.committed) - 2  # pair for the sampled token at pos c-1


def test_batched_schedule_positions(monkeypatch):
    """End-aligned drafter schedule: every user's last pending lands at leg
    width-K, chains are consecutive, and padding replays the first pending."""
    import models.demos.blackhole.qwen36.tt.spec_decode_batched as sb_mod

    fake_cls = _make_fake_batched_spec(3, [100, 70], monkeypatch)
    spec = fake_cls(_BatchedOracleDrafter(2))
    spec.seed()
    spec.slots[0].pending = [(1, None, 99), (2, None, 100), (3, None, 101)]
    spec.slots[1].pending = [(4, None, 69)]
    width = 3 + spec.draft_len - 1
    pos, pads = spec._schedule_positions(width)
    K = spec.draft_len
    assert pads == [0, 2]
    # Last pending at leg width-K for both users.
    assert int(pos[0, width - K]) == 101 and int(pos[1, width - K]) == 69
    # Padding legs replay the first pending position.
    assert int(pos[1, 0]) == 69 and int(pos[1, 1]) == 69
    # Chain legs are consecutive after the last pending.
    assert [int(p) for p in pos[0, width - K :]] == [101, 102, 103]
    assert [int(p) for p in pos[1, width - K :]] == [69, 70, 71]


# --------------------------------------------------------------------------- #
# Fused decode-width verify loop: the REAL Qwen36FusedSpeculativeDecoder
# generate() with fake device ops — per-user targets, commit-by-select rows,
# and the anchor==head invariant.
# --------------------------------------------------------------------------- #
def _make_fake_fused_spec(draft_len, prompt_lens, monkeypatch, use_trace=False):
    from types import SimpleNamespace

    import torch

    import models.demos.blackhole.qwen36.tt.spec_decode_batched as sb_mod
    import models.demos.blackhole.qwen36.tt.spec_decode_fused as sf_mod

    monkeypatch.setattr(sb_mod, "ttnn", SimpleNamespace(deallocate=lambda *_a, **_k: None))
    monkeypatch.setattr(sf_mod, "ttnn", SimpleNamespace(deallocate=lambda *_a, **_k: None))
    monkeypatch.setenv("TT_SPEC_TRACE", "1" if use_trace else "0")
    B = len(prompt_lens)

    class _FakeFused(sf_mod.Qwen36FusedSpeculativeDecoder):
        def __init__(self, mtp):
            # Bypass the device-heavy __init__ chain; wire only the bookkeeping.
            model_stub = SimpleNamespace(num_devices=8, layers=[], args=SimpleNamespace(dim=3))
            page_table = torch.arange(B * 4, dtype=torch.int32).reshape(B, 4)
            sb_mod.Qwen36BatchedSpeculativeDecoder.__init__(self, model_stub, mtp, page_table, draft_len=draft_len)
            self._use_trace = use_trace
            self.W = draft_len + 1
            self.commit_rows = []  # (iteration, [m_u])
            self._last_accepts = [0] * B
            self.verify_captured_after_drafter = None
            self.seeded_select = False

        def _seed_select_state(self):  # device seeding; here only the ordering matters
            self.seeded_select = True

        def _capture_verify_trace(self):  # device capture; here only the order matters
            self.verify_captured_after_drafter = getattr(self.mtp, "captured", False)

        def seed(self):
            for u, T in enumerate(prompt_lens):
                slot = sb_mod.SpecSlot(
                    committed=[_tok_u(u, i) for i in range(T)],
                    a=sb_mod.block_aligned_prefill_len(T),
                )
                slot.k_ema = float(self.draft_len)
                self.slots.append(slot)

        def _snapshot_from_live(self):
            pass

        def _restore_from_snapshot(self):
            pass

        def _batched_verify(self, chunks):  # the parent's chunk-shaped first verify
            results = []
            for u, (tokens, a_u, n_u, hid_start, score_start) in enumerate(chunks):
                n_score = min(sb_mod._VERIFY_ROWS, n_u - score_start)
                target_ids = [_tok_u(u, a_u + score_start + j + 1) for j in range(n_score)]
                hid = torch.stack([_hid_u(u, a_u + hid_start + j) for j in range(n_u - hid_start)])
                results.append((target_ids, hid))
            return results

        def _commit_user(self, u):
            slot = self.slots[u]
            slot.a += sb_mod.commit_advance(len(slot.committed) - slot.a)

        def _align_anchor_to_head(self):
            for slot in self.slots:
                slot.a = len(slot.committed) - 1

        def _fused_verify(self, drafts):
            results = []
            for u, slot in enumerate(self.slots):
                assert slot.a == len(slot.committed) - 1, f"user {u}: fused anchor must equal the committed head"
                c = len(slot.committed) - 1
                ids = [_tok_u(u, c + t + 1) for t in range(self.W)]
                hid = torch.stack([_hid_u(u, c + t) for t in range(self.W)])
                results.append((ids, hid))
            return results

        def _commit_select(self, accepts):
            self.commit_rows.append(list(accepts))
            for u, m in enumerate(accepts):
                if m is not None:
                    assert 0 <= m <= self.draft_len

    return _FakeFused


def test_fused_loop_desync(monkeypatch):
    """The fused loop (anchor==head, commit-by-select) emits every user's exact
    target sequence with per-user desyncing accepts."""
    prompt_lens = [100, 70, 130, 65]
    fake_cls = _make_fake_fused_spec(3, prompt_lens, monkeypatch)
    spec = fake_cls(_BatchedOracleDrafter(len(prompt_lens)))
    spec.seed()
    rows, stats = spec.generate(50)
    for u, (row, T) in enumerate(zip(rows, prompt_lens)):
        assert row == [_tok_u(u, T + i) for i in range(len(row))], f"user {u} diverged"
        assert len(row) >= 50
        assert spec.slots[u].a == len(spec.slots[u].committed) - 1
    assert len({tuple(s.accepts[:10]) for s in spec.slots}) > 1, "accepts did not desync"
    assert spec.commit_rows and all(len(r) == len(prompt_lens) for r in spec.commit_rows)
    # Every iteration's commit rows match that iteration's accepts.
    for it, row in enumerate(spec.commit_rows):
        for u, m in enumerate(row):
            if m is not None:
                assert m == spec.slots[u].accepts[it]


# --------------------------------------------------------------------------- #
# Traced batched drafter windows (host emulation): the REAL _draft_traced +
# _setup_drafter_traces scheduling against a window-shaped oracle.
# --------------------------------------------------------------------------- #
class _WindowOracleMTP(_BatchedOracleDrafter):
    """Batched-window drafter emulation over the same per-user oracle rule.

    step_batched consumes the staged position table exactly like the device
    traces (leg j reads column j; chained legs feed each user's previous
    hidden), so scheduling bugs surface as convention asserts or wrong
    drafts. Also enforces the ensure -> stage -> compile -> capture -> replay
    discipline."""

    def __init__(self, users):
        super().__init__(users)
        self._bwin = None
        self._staged = None
        self._h = [None] * users
        self.captured = False
        self.stage_count = 0

    def ensure_batched_window(self, w_max, users):
        assert users == self.users
        if self._bwin is None:
            self._bwin = {"w_max": w_max, "traces": {}}

    def stage_batched_window(self, pos_table):
        assert self._bwin is not None, "staged before ensure_batched_window"
        assert pos_table.shape[0] == self.users and pos_table.shape[1] <= self._bwin["w_max"]
        self._staged = pos_table.clone()
        self.stage_count += 1

    def compile_batched_window(self):
        if not self._bwin["traces"]:
            self._bwin["traces"] = {j: True for j in range(self._bwin["w_max"])}

    def capture_batched_window(self):
        assert self._bwin["traces"], "captured before the compile pass"
        self.captured = True

    def step_batched(self, j, tokens, hiddens=None, chain_hidden=False, want_tokens=False):
        assert self.captured, "replayed before capture"
        assert self._staged is not None and j < self._staged.shape[1]
        picks = []
        for u in range(self.users):
            pos = int(self._staged[u, j])
            h = self._h[u] if chain_hidden else hiddens[u]
            logits, g = self.step(int(tokens[u]), h, pos, user=u)
            self._h[u] = g
            picks.append(int(logits.argmax()))
        return picks if want_tokens else None


def _run_fused(prompt_lens, monkeypatch, use_trace, max_new=50, stop_tokens=None):
    fake_cls = _make_fake_fused_spec(3, prompt_lens, monkeypatch, use_trace=use_trace)
    mtp = _WindowOracleMTP(len(prompt_lens)) if use_trace else _BatchedOracleDrafter(len(prompt_lens))
    spec = fake_cls(mtp)
    if stop_tokens is not None:
        spec.stop_tokens = stop_tokens
    spec.seed()
    rows, stats = spec.generate(max_new)
    return spec, mtp, rows


def test_fused_traced_drafter_matches_eager(monkeypatch):
    """The traced batched drafter window (end-aligned schedule, chained legs,
    iteration-1 eager warmup) produces bit-identical outputs and accept traces
    to the eager per-user drafter."""
    prompt_lens = [100, 70, 130, 65]
    outs = {}
    for use_trace in (False, True):
        spec, mtp, rows = _run_fused(prompt_lens, monkeypatch, use_trace)
        for u, (row, T) in enumerate(zip(rows, prompt_lens)):
            assert row == [_tok_u(u, T + i) for i in range(len(row))], f"user {u} diverged (trace={use_trace})"
            assert len(row) >= 50
        outs[use_trace] = (rows, [list(s.accepts) for s in spec.slots], [list(s.k_used) for s in spec.slots])
        if use_trace:
            assert spec.seeded_select, "traced mode must seed the select-state before the loop"
            assert mtp.captured
            assert spec.verify_captured_after_drafter, "verify trace must capture after the drafter captures"
            # The commit rode as data: upload indices always stayed in-range.
            assert all(0 <= m <= spec.draft_len for m in spec._last_accepts)
            # One setup staging + one per traced iteration (iteration 1 is eager).
            iters = max(len(s.accepts) for s in spec.slots)
            assert mtp.stage_count == iters, f"{mtp.stage_count} stagings for {iters} iterations"
    assert outs[True] == outs[False], "the traced drafter changed the loop's outputs"


def _shape_sim_ttnn(untilize_pads):
    """Shape-semantics ttnn stand-in: every op propagates shapes only, reshape
    enforces volume equality, and untilize either strips or carries the padded
    tile rows (silicon shows both, per op variant — 56397)."""
    import math
    from types import SimpleNamespace

    class T:
        def __init__(self, shape, layout="TILE"):
            self.shape = list(shape)
            self.layout = layout

        def volume(self):
            return math.prod(self.shape)

    def _pad2(shape):
        s = list(shape)
        s[-2] = ((s[-2] + 31) // 32) * 32
        return s

    def _reshape(t, shape):
        assert math.prod(shape) == t.volume(), f"reshape volume: {t.shape} -> {list(shape)}"
        return T(list(shape), t.layout)

    fake = SimpleNamespace(
        TILE_LAYOUT="TILE",
        ROW_MAJOR_LAYOUT="RM",
        DRAM_MEMORY_CONFIG="DRAM",
        deallocate=lambda *a, **k: None,
        to_memory_config=lambda t, mc: t,
        to_layout=lambda t, layout: T(t.shape, layout),
        slice=lambda t, s, e: T([b - a for a, b in zip(s, e)], t.layout),
        reshape=_reshape,
        untilize=lambda t, use_multicore=False: T(_pad2(t.shape) if untilize_pads else list(t.shape), "RM"),
        pad=lambda t, spec, value: T([d + lo + hi for d, (lo, hi) in zip(t.shape, spec)], t.layout),
        argmax=lambda t, dim, keepdim=False, memory_config=None: T(t.shape[:-1], "RM"),
        max=lambda t, dim, memory_config=None: T(t.shape[:-1], t.layout),
    )
    return fake, T


@pytest.mark.parametrize("untilize_pads", [False, True], ids=["untilize-strips", "untilize-pads"])
def test_batched_step_body_shapes(monkeypatch, untilize_pads):
    """The REAL _batched_step_body against shape-simulating ttnn: the argmax
    row-merge must hold whether untilize strips the padded tile rows or
    carries them into the volume (the 56397 regression class)."""
    from types import SimpleNamespace

    import models.demos.blackhole.qwen36.tt.mtp as mtp_mod

    fake, T = _shape_sim_ttnn(untilize_pads)
    monkeypatch.setattr(mtp_mod, "ttnn", fake)
    B, w_max, rd, dim, vs = 8, 7, 64, 96, 320
    stub = SimpleNamespace(
        _bwin={
            "B": B,
            "w_max": w_max,
            "traces": {},
            "pos": T([B, w_max], "RM"),
            "cos": T([B, w_max, rd], "RM"),
            "sin": T([B, w_max, rd], "RM"),
            "tok": T([B, 1], "RM"),
            "h": T([B, 1, dim]),
            "pt": T([B, 4], "RM"),
        },
        rope=SimpleNamespace(head_dim=rd),
        _step_graph=lambda tok, h, cur_pos, cos, sin, page_table=None: (T([B, 1, vs]), T([B, 1, dim])),
    )
    idx, val, normed = mtp_mod.Qwen36MTPHead._batched_step_body(stub, 0)
    assert idx.shape == [1, 1, 32], f"idx shape {idx.shape}"
    assert val.shape == [1, 1, 32], f"val shape {val.shape}"
    assert normed.shape == [B, 1, dim]


def test_fused_traced_drafter_with_finished_user(monkeypatch):
    """A user finishing mid-run shrinks to ONE idempotent replay leg in the
    schedule; the traced and eager loops still agree exactly."""
    prompt_lens = [100, 70, 130]
    stop = {_tok_u(1, 78)}  # user 1 stops after ~8 tokens
    outs = {}
    for use_trace in (False, True):
        spec, _mtp, rows = _run_fused(prompt_lens, monkeypatch, use_trace, stop_tokens=stop)
        for u, (row, T) in enumerate(zip(rows, prompt_lens)):
            target = [_tok_u(u, T + i) for i in range(len(row))]
            assert row == target, f"user {u} diverged (trace={use_trace})"
        outs[use_trace] = (rows, [list(s.accepts) for s in spec.slots])
        done_users = [u for u, s in enumerate(spec.slots) if s.done]
        assert done_users, "the stop token never fired"
        for u in done_users:
            assert spec.slots[u].committed[-1] in stop
    assert outs[True] == outs[False]


# --------------------------------------------------------------------------- #
# Device-sim audit of the ENTIRE never-executed traced surface (one-shot rule):
# a fake ttnn with tensor LIFETIME tracking (storage-sharing aliases; any op
# touching a freed buffer fails), shape propagation (reshape enforces volume,
# matmul/concat/copy check dims), and capture discipline (no host readback and
# no fresh device upload inside a capture). It drives the REAL
# mtp.ensure/stage/compile/capture/step_batched and the REAL fused
# _seed_select_state / _ensure_verify_io / _verify_upload / _verify_graph /
# _verify_read / _capture_verify_trace / _draft_traced / _fused_verify across
# seeding, one eager iteration, trace setup, and TWO armed iterations.
#
# untilize is parametrized over the two behaviors consistent with all silicon
# evidence: strips padding always, or carries padded rows when the logical row
# dim is 1 (the 56397 [B,1,Vs] -> [B,32,Vs] case; the GDN sites at [B,3,D] and
# [B,4,D] returned stripped shapes on the same silicon).
# --------------------------------------------------------------------------- #
class _SimTensor:
    _n = 0

    def __init__(self, shape, layout="TILE", store=None, host=False):
        self.shape = list(int(s) for s in shape)
        self.layout = layout
        self.store = store if store is not None else {"alive": True}
        self.host = host
        _SimTensor._n += 1
        self.name = f"t{_SimTensor._n}"

    def volume(self):
        import math

        return math.prod(self.shape)


def _make_sim(untilize_pads_row1):
    import math
    from types import SimpleNamespace

    import torch as _torch

    T = _SimTensor
    state = {"capturing": False, "traces": set(), "next_trace": 0}

    def chk(*ts):
        for t in ts:
            assert isinstance(t, T), f"non-tensor operand {type(t)}"
            assert t.store["alive"], f"op consumed FREED tensor {t.name} shape {t.shape}"

    def dealloc(t, *a, **k):
        chk(t)
        t.store["alive"] = False

    def from_torch(ht, dtype=None, layout="RM", device=None, mesh_mapper=None, **k):
        if device is not None:
            assert not state["capturing"], "device allocation (from_torch) inside a trace capture"
        return T(list(ht.shape), layout, host=(device is None))

    def copy_h2d(src, dst):
        chk(dst)
        assert src.host, "copy_host_to_device src must be a host tensor"
        assert src.shape == dst.shape, f"staging shape {src.shape} != buffer {dst.shape}"

    def reshape(t, shape):
        chk(t)
        shape = [int(s) for s in shape]
        assert math.prod(shape) == t.volume(), f"reshape volume: {t.shape} -> {shape}"
        return T(shape, t.layout, store=t.store)  # view: shares the buffer

    def t_slice(t, start, end, memory_config=None):
        chk(t)
        assert all(0 <= a < b <= d for a, b, d in zip(start, end, t.shape)), f"slice [{start}:{end}] of {t.shape}"
        return T([b - a for a, b in zip(start, end)], t.layout)

    def untilize(t, use_multicore=False):
        chk(t)
        s = list(t.shape)
        if untilize_pads_row1 and len(s) >= 2 and s[-2] == 1:
            s[-2] = 32
        return T(s, "RM")

    def concat(ts, dim, memory_config=None):
        chk(*ts)
        base = list(ts[0].shape)
        for o in ts[1:]:
            assert all(
                i == dim or a == b for i, (a, b) in enumerate(zip(base, o.shape))
            ), f"concat dim mismatch {base} vs {o.shape}"
        base[dim] = sum(o.shape[dim] for o in ts)
        return T(base, ts[0].layout)

    def matmul(a, b, memory_config=None, **k):
        chk(a, b)
        assert a.shape[-1] == b.shape[-2], f"matmul {a.shape} @ {b.shape}"
        return T(list(a.shape[:-1]) + [b.shape[-1]], "TILE")

    def linear(a, w, compute_kernel_config=None, memory_config=None, **k):
        chk(a, w)
        assert a.shape[-1] == w.shape[-2], f"linear {a.shape} @ {w.shape}"
        return T(list(a.shape[:-1]) + [w.shape[-1]], "TILE")

    def eltwise2(a, b, memory_config=None, **k):
        chk(a, b)
        return T(a.shape, a.layout)

    def to_torch(t, mesh_composer=None, **k):
        chk(t)
        assert not state["capturing"], "host readback (to_torch) inside a trace capture"
        if mesh_composer is not None:
            return _torch.zeros([8] + t.shape)
        return _torch.zeros(t.shape)

    def begin_cap(device, cq_id=0):
        assert not state["capturing"], "nested trace capture"
        state["capturing"] = True
        state["next_trace"] += 1
        return state["next_trace"]

    def end_cap(device, tid, cq_id=0):
        assert state["capturing"]
        state["capturing"] = False
        state["traces"].add(tid)

    def execute(device, tid, cq_id=0, blocking=False):
        assert tid in state["traces"], f"replay of uncaptured trace {tid}"

    def copy(src, dst):
        chk(src, dst)
        assert src.shape == dst.shape, f"copy shape {src.shape} != {dst.shape}"

    fake = SimpleNamespace(
        _T=T,
        _state=state,
        TILE_LAYOUT="TILE",
        ROW_MAJOR_LAYOUT="RM",
        DRAM_MEMORY_CONFIG="DRAM",
        L1_MEMORY_CONFIG="L1",
        int32="int32",
        uint32="uint32",
        bfloat16="bfloat16",
        float32="float32",
        deallocate=dealloc,
        from_torch=from_torch,
        copy_host_to_device_tensor=copy_h2d,
        reshape=reshape,
        slice=t_slice,
        untilize=untilize,
        to_layout=lambda t, layout, **k: (chk(t), _SimTensor(t.shape, layout))[1],
        # Same-config to_memory_config returns a DISTINCT handle sharing the
        # buffer (the 56610 alias hazard); model every call as an alias.
        to_memory_config=lambda t, mc: (chk(t), _SimTensor(t.shape, t.layout, store=t.store))[1],
        pad=lambda t, spec, value: (chk(t), _SimTensor([d + lo + hi for d, (lo, hi) in zip(t.shape, spec)], t.layout))[
            1
        ],
        argmax=lambda t, dim, keepdim=False, memory_config=None: (chk(t), _SimTensor(t.shape[:-1], "RM"))[1],
        max=lambda t, dim, memory_config=None: (chk(t), _SimTensor(t.shape[:-1], t.layout))[1],
        rms_norm=lambda t, weight=None, epsilon=None, memory_config=None, **k: (
            chk(t, weight),
            _SimTensor(t.shape, t.layout),
        )[1],
        concat=concat,
        matmul=matmul,
        linear=linear,
        add=eltwise2,
        multiply=eltwise2,
        addcmul=lambda o, a, b, memory_config=None: (chk(o, a, b), _SimTensor(o.shape, o.layout))[1],
        silu=lambda t, memory_config=None: (chk(t), _SimTensor(t.shape, t.layout))[1],
        to_torch=to_torch,
        get_device_tensors=lambda t: [t],
        ConcatMeshToTensor=lambda device, dim=0: object(),
        ReplicateTensorToMesh=lambda device: object(),
        begin_trace_capture=begin_cap,
        end_trace_capture=end_cap,
        execute_trace=execute,
        copy=copy,
        synchronize_device=lambda device, **k: None,
    )
    return fake


def _sim_env(monkeypatch, untilize_pads_row1):
    """Patch every exercised module onto the device sim; return (fake, mods)."""
    import sys
    from types import SimpleNamespace

    import models.demos.blackhole.qwen36.tt.mtp as mtp_mod
    import models.demos.blackhole.qwen36.tt.spec_decode_batched as sb_mod
    import models.demos.blackhole.qwen36.tt.spec_decode_fused as sf_mod

    fake = _make_sim(untilize_pads_row1)
    T = fake._T
    for mod in (mtp_mod, sb_mod, sf_mod):
        monkeypatch.setattr(mod, "ttnn", fake)
    monkeypatch.setattr(
        mtp_mod,
        "rms_norm_ttnn",
        lambda x, w, eps=1e-6, memory_config=None: fake.rms_norm(x, weight=w, memory_config=memory_config),
    )
    # _gdn_verify_seq imports these INSIDE the function; route them to stubs.
    tpc_stub = SimpleNamespace(matmul_1d_decode=None)

    def _all_reduce(partial, mesh, ccl, cluster_axis=0, dim=3, topology=None, memory_config=None):
        assert partial.store["alive"], "tt_all_reduce consumed freed tensor"
        return T(partial.shape, partial.layout)

    ccl_stub = SimpleNamespace(tt_all_reduce=_all_reduce)
    monkeypatch.setitem(sys.modules, "models.demos.blackhole.qwen36.tt.tp_common", tpc_stub)
    monkeypatch.setitem(sys.modules, "models.tt_transformers.tt.ccl", ccl_stub)
    monkeypatch.setenv("TT_SPEC_FUSED_DEBUG", "0")
    return fake, (mtp_mod, sb_mod, sf_mod)


def _build_traced_rig(fake, mods):
    """REAL Qwen36MTPHead + Qwen36FusedSpeculativeDecoder over sim tensors."""
    from types import SimpleNamespace

    import torch

    mtp_mod, sb_mod, sf_mod = mods
    T = fake._T
    B, K = 8, 3
    W = K + 1
    R = B * W
    dim, rd, vs, nv, dk, dv, conv_k = 64, 16, 128, 2, 32, 32, 4
    D = 48  # qkv_dim_tp
    qkvzab_dim = D + nv * dv + 32

    def new(shape, layout="TILE"):
        return T(shape, layout)

    def chk(t):
        assert t.store["alive"], f"stub consumed FREED tensor {t.name} shape {t.shape}"

    dev = object()

    # --- the REAL drafter head over sim tensors ---
    head = mtp_mod.Qwen36MTPHead.__new__(mtp_mod.Qwen36MTPHead)
    head.device = dev
    head.args = SimpleNamespace(dim=dim, vocab_size=vs * 8)
    head.rope = SimpleNamespace(head_dim=rd, cos_cpu=torch.zeros(4096, rd), sin_cpu=torch.zeros(4096, rd))
    head._mesh_kwargs = {}
    head.norm_eps = 1e-6
    head.num_devices = 8
    head.lm_head_sharded = True
    head.embedding = lambda tok: (chk(tok), new([tok.shape[0], tok.shape[1], dim]))[1]
    head.fc_weight = new([2 * dim, dim])
    head.lm_head_weight = new([dim, vs])
    head.pre_fc_norm_embedding = new([dim])
    head.pre_fc_norm_hidden = new([dim])
    head.final_norm = new([dim])
    head.attn_norm_w = new([dim])
    head.ffn_norm_w = new([dim])
    head.compute_kernel_config = None
    head.attention = SimpleNamespace(
        forward=lambda x, cos, sin, position_tensor=None, page_table=None: (
            chk(x),
            chk(cos),
            chk(sin),
            chk(position_tensor),
            chk(page_table),
            new(x.shape),
        )[-1]
    )
    head.feed_forward = SimpleNamespace(forward=lambda t: (chk(t), new(t.shape))[1])
    head._page_tables_host = torch.arange(B * 4, dtype=torch.int32).reshape(B, 4)
    head._page_table_tt = None
    head._bwin = None
    head._win = None
    head._last_normed = None

    # --- the REAL fused decoder over sim tensors ---
    sf = sf_mod.Qwen36FusedSpeculativeDecoder.__new__(sf_mod.Qwen36FusedSpeculativeDecoder)

    class _Dn:
        def __init__(self):
            self.K = conv_k
            self.Nk = 1
            self.Nv = nv
            self.qkv_dim_tp = D
            self._fuse_ab = True
            self.rec_state = new([B, nv, dk, dv])
            self._batched_conv_carry = new([B, conv_k - 1, D])
            self.tw = {
                "qkvz": new([dim, qkvzab_dim]),
                "dt_bias": new([1, 1, nv]),
                "neg_exp_A": new([1, 1, nv]),
                "norm_w": new([1, 1, dv]),
                "out": new([nv * dv, dim]),
                "conv_taps": [new([1, 1, D]) for _ in range(conv_k)],
            }
            self.cfg = None
            self.tt_ccl = None

        def _col_proj(self, x3, w, progcfg, mc):
            chk(x3)
            chk(w)
            return new([1, x3.shape[1], qkvzab_dim])

        def _row_proj(self, g, w):
            chk(g)
            chk(w)
            return new([1, g.shape[1], dim])

        def _write_index(self, buf, src, idx, dim=0):
            chk(buf)
            chk(src)
            assert 0 <= idx < buf.shape[dim], f"_write_index row {idx} outside {buf.shape}"
            src.store["alive"] = False  # the real op consumes src

    dns = [_Dn(), _Dn()]

    class _Fop:
        @staticmethod
        def recurrence_seq_rows(conv, qkvzab, rec, stash, dtb, nega, w_, out, *, accepts=None, **kw):
            for t in (conv, qkvzab, rec, stash, dtb, nega, w_, out):
                chk(t)
            assert list(out.shape) == [1, R, nv * dv], f"gated {out.shape}"
            assert list(stash.shape) == [B * W, nv, dk, dv]
            if accepts is not None:
                chk(accepts)
                assert accepts.shape[-1] >= B and (accepts.shape[-1] * 4) % 32 == 0

    def attn_layer():
        def fwd(x, cos, sin, position_tensor=None, page_table=None, mode=None, kv_update_positions=None):
            for t in [x, cos, sin, position_tensor, page_table] + list(kv_update_positions or []):
                chk(t)
            x.store["alive"] = False  # production decode consumes its input
            return new([1, 1, R, dim])

        return SimpleNamespace(is_full_attention=True, forward=fwd)

    def gdn_layer(dn):
        return SimpleNamespace(
            is_full_attention=False,
            attention=dn,
            attention_norm=lambda h, mode=None, norm_config=None: (chk(h), new(h.shape))[1],
            ffn_norm=lambda h, mode=None, norm_config=None: (chk(h), new(h.shape))[1],
            feed_forward=SimpleNamespace(forward=lambda t: (chk(t), new(t.shape))[1]),
        )

    model = SimpleNamespace(
        device=dev,
        mesh_device=dev,
        num_devices=8,
        args=SimpleNamespace(
            dim=dim,
            rope_head_dim=rd,
            rope_theta=10000.0,
            vocab_size=vs * 8,
            get_norm_config=lambda *a, **k: None,
            gdn_qkvzab_progcfg=None,
            ccl_topology=lambda: None,
        ),
        embd=lambda tok: (chk(tok), new([tok.shape[0], tok.shape[1], dim]))[1],
        _final_norm_decode=lambda x: (chk(x), new(x.shape))[1],
        lm_head_weight=new([dim, vs]),
        layers=[gdn_layer(dns[0]), attn_layer(), gdn_layer(dns[1])],
        _lmhead_vocab_sharded=True,
    )

    sf.model = model
    sf.mtp = head
    sf.page_table = torch.arange(B * 4, dtype=torch.int32).reshape(B, 4)
    sf.draft_len = K
    sf.W = W
    sf.B = B
    sf.stop_tokens = set()
    sf._dns = dns
    sf._stash = [new([B * W, nv, dk, dv]) for _ in dns]
    sf._conv_win = [None] * len(dns)
    sf._win_buf = [None] * len(dns)
    sf._vio = None
    sf._win_sel = new([B, W * (conv_k - 1), (conv_k - 1) + W])
    sf._conv_k = conv_k
    sf._nv, sf._dk, sf._dv = nv, dk, dv
    sf._fop = _Fop()
    sf._use_trace = True
    sf._last_accepts = [0] * B
    sf._debug_done = True
    sf.slots = []
    for u in range(B):
        slot = sb_mod.SpecSlot(committed=[(u + i) % 97 for i in range(80)], a=79)
        c = len(slot.committed) - 1
        for i in range(3):  # armed catch-up pairs, as after _first_verify + align
            slot.pending.append((slot.committed[c - 2 + i], torch.zeros(dim), c - 3 + i + 1))
        sf.slots.append(slot)
    return sf, head, (B, K, W, R, dim)


@pytest.mark.parametrize("untilize_pads_row1", [False, True], ids=["untilize-strips", "untilize-pads-row1"])
def test_traced_surface_device_sim(monkeypatch, untilize_pads_row1):
    """Lifetime + shape + capture-discipline audit of the never-executed traced
    surface: seed select-state, one eager iteration (real fused verify graph),
    the full trace setup (all drafter windows compiled + captured, verify
    captured), then TWO armed iterations of traced drafting, verify replay,
    readback, and accepts re-upload."""
    import torch

    import models.demos.blackhole.qwen36.tt.spec_decode_fused as sf_mod

    fake, mods = _sim_env(monkeypatch, untilize_pads_row1)
    sf, head, (B, K, W, R, dim) = _build_traced_rig(fake, mods)

    def accept_and_rearm(results, drafts):
        accepts = []
        for u, slot in enumerate(sf.slots):
            target_ids, hid = results[u]
            c = len(slot.committed) - 1
            m, new_tokens = sf_mod.greedy_accept(drafts[u], target_ids[: K + 1])
            slot.accepts.append(m)
            accepts.append(m)
            for j2, tok in enumerate(new_tokens):
                slot.committed.append(tok)
                slot.out.append(tok)
                slot.pending.append((tok, hid[j2], c + j2))
            slot.a = len(slot.committed) - 1
        sf._last_accepts = [m if m is not None else 0 for m in accepts]
        return accepts

    # Pre-loop seeding, then iteration 1 fully eager (the real traced-mode graph).
    sf._seed_select_state()
    monkeypatch.setattr(head, "step", lambda tok, hid, pos, user=0: (torch.zeros(128 * 8), torch.zeros(dim)))
    drafts = sf._draft_eager(64)
    assert all(len(d) == K for d in drafts)
    results = sf._fused_verify(drafts)  # eager branch: builds _vio, runs _verify_graph
    accept_and_rearm(results, drafts)

    # Arm: all drafter windows compile + capture, verify captures last.
    sf._setup_drafter_traces()
    assert head._bwin["traces"] and sf._vio["trace"] is not None
    assert not fake._state["capturing"]

    # TWO armed iterations: traced drafting + verify replay + accepts re-upload.
    for _ in range(2):
        width = max(len(sf._sched_pending(s)) for s in sf.slots) + K - 1
        assert width <= head._bwin["w_max"], f"window {width} overflows"
        drafts = sf._draft_traced(width, 64)
        assert all(len(d) == K for d in drafts)
        results = sf._fused_verify(drafts)  # replay branch
        accept_and_rearm(results, drafts)

    # Every persistent handle must still be alive.
    persistents = (
        list(sf._vio["upd"])
        + [sf._vio[k] for k in ("tok", "pos", "cos", "sin", "pt", "gated", "acc", "sel")]
        + [sf._vio["trace"][k] for k in ("xn", "idx", "val")]
        + sf._win_buf
        + sf._stash
        + [dn.rec_state for dn in sf._dns]
        + [dn._batched_conv_carry for dn in sf._dns]
        + [head._bwin[k] for k in ("pos", "cos", "sin", "tok", "h", "pt")]
        + [tr[k] for tr in head._bwin["traces"].values() for k in ("idx", "val", "normed")]
    )
    for t in persistents:
        assert t.store["alive"], f"persistent {t.name} shape {t.shape} was freed"
