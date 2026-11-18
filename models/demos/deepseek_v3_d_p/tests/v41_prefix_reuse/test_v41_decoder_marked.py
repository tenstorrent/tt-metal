# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0073 marked decoder chunks (V41_PREFILL_DECODER=1 + V41_PREFILL_DECODER_MARKED=1, tt-d-gen ``pd_decoder_cone``) on a
HOST fake device (``fake_v41.py``; no chips, no real ttnn): the real runtime chunk path, ``kv_export.py`` and
``tt/v41/prefix_reuse.py``, and the split rule ``tt/v41/decoder_cone.py``.

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse

* the split rule: for every prompt length, both policies leave an exact export (the cone model: layer l's input exact from
  start + (l - 20) x 127 on) with the replay and decoder-chunk counts each policy promises, and never a longer replay than
  today's pd_migrate_tail rule;
* a marked burst runs layers 20..39 on its marked chunks only, layer 20 KV-only (with its slab rows) on the others, and the
  decoder starts at the first marked chunk from zero windows; the runtime's cone check sees it;
* with prefix reuse H1 / H2 on, every scenario's WHOLE export and model state equal a cold marked prefill's, and no decoder
  chunk is ever skipped or restored;
* unmarked headers run the decoder on every chunk (an engine that does not mark is slow, never inexact).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import fake_v41 as F  # noqa: E402 -- installs the fake ttnn before anything imports it
import pytest  # noqa: E402
import test_v41_prefix_reuse as T  # noqa: E402 -- the H1 harness (same directory)
import test_v41_prefix_store as ST  # noqa: E402 -- the H2 store env

from models.demos.common.prefill.prefix_reuse import (  # noqa: E402
    DECODER_CHUNK_FLAG,
    decoder_chunk_flag,
    split_slot_word,
)
from models.demos.deepseek_v3_d_p.tt.v41 import decoder_cone as DC  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v41 import prefix_reuse as PR  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v41 import runtime as RT  # noqa: E402

C, MARKED, LAST, CFG = T.C, T.MARKED, T.LAST, T.CFG
CONE = 700  # the fake's cone (C = 512): 1-2 decoder chunks, as 2541 is at the production chunk 4096
toks = T.toks
X, A, A2, B, AP, LONG, SHORT_SIB = T.X, T.A, T.A2, T.B, T.AP, T.LONG, T.SHORT_SIB
U = ST.U
assert not hasattr(sys.modules["ttnn"], "open_mesh_device"), "the REAL ttnn got imported"


# ------------------------------------------------------------------------------------------------------- the split rule
def _exact_cone_model(prefix: int, start, cone: int) -> bool:
    """The replay argument, per layer: with the decoder started at ``start`` from zero windows, layer l's input (l = 20..39)
    is exact at p >= start + (l - 20) x 127 (start = 0: everywhere); the export ships layer l's window rows of the prefix's
    last 128 positions, so each needs its input exact there. ``cone`` must be 20 x 127 + 1 for this to equal cone_exact.
    """
    if start is None:
        return False
    if start == 0:
        return True
    return all(prefix - 128 >= start + (l - 20) * 127 for l in range(20, 40))


def test_cone_is_the_replay_floor():
    assert DC.CONE == 2541 == 20 * 127 + 1
    for prefix in range(1, 40000, 37):
        for start in range(0, prefix, 61):
            assert _exact_cone_model(prefix, start, DC.CONE) == DC.cone_exact(prefix, start), (prefix, start)


@pytest.mark.parametrize("chunk", [4096, 8192, 2048])
@pytest.mark.parametrize("single", [False, True])
def test_split_rule_every_prompt_length(chunk, single):
    """Both policies (M: single=False, L: single=True) on every prompt length up to ~3 chunks past min_disagg, align 128."""
    cone, align = DC.CONE, 128
    for S in range(1, 6 * chunk + 4 * align, 7):
        n = DC.decoder_prefix_cap(S, chunk=chunk, cone=cone, align=align, single_chunk=single)
        old = DC.migrated_prefix_cap(S, cone, align)
        if n == 0:
            assert S - 1 < align, S  # only a prompt with less than one align unit before its last token stays local
            continue
        assert n % align == 0 and 1 <= S - n
        d0 = DC.decoder_start(n, chunk, cone)
        marked = [s0 for s0 in range(0, n, chunk) if DC.is_decoder_chunk(min(n, s0 + chunk), n, cone)]
        assert marked and marked[0] == d0 and marked == list(range(d0, n, chunk)), (S, n, d0, marked)
        assert _exact_cone_model(n, d0, cone), (S, n, d0)
        assert n >= old, f"S {S}: the decoder split replays more than today ({S - n} > {S - old})"
        span = -(-cone // chunk)  # the fewest chunks a cone can take (1 at the production chunk 4096 >= 2541)
        if single and chunk >= cone:
            assert len(marked) == 1, (S, n, marked)
            assert S - n < cone + align, (S, n)
        else:  # M (or L with chunk < cone, which is M's rule)
            assert n == DC.decoder_prefix_cap(S, chunk=chunk, cone=cone, align=align), (S, n)
            assert len(marked) <= span + 1 and S - n <= align, (S, n, marked)


def test_split_rule_max_tokens_cap():
    n = DC.decoder_prefix_cap(100_000, chunk=4096, max_tokens=65_536 + 100)
    assert n == 65_536 and DC.decoder_start(n, 4096) == 61_440


def test_slot_word_decoder_flag():
    assert DECODER_CHUNK_FLAG == 1 << 28
    assert decoder_chunk_flag(3) is None  # unmarked: nothing known
    assert decoder_chunk_flag(3 | MARKED) is False
    assert decoder_chunk_flag(3 | MARKED | DECODER_CHUNK_FLAG | LAST) is True
    assert split_slot_word(3 | MARKED | DECODER_CHUNK_FLAG | LAST) == (3, True)
    assert split_slot_word(5 | MARKED | DECODER_CHUNK_FLAG) == (5, False)
    assert decoder_chunk_flag(-1) is None and decoder_chunk_flag(0xFFFFFFFF) is None


# ------------------------------------------------------------------------------------------------------- the fake system
class MarkedSystem(ST.StoreSystem):
    """The harness's runner with marked decoder chunks: tt-d-gen's marks (bit 28 on the chunks covering the prompt's last
    CONE positions), every layer a block. ``did`` per chunk: S skipped, R restored, C computed (+ silent rebuilds)."""

    def __init__(self, *, flag=False, users=1):
        super().__init__(users=users, flag=flag, marked=True)

    def prompt(self, tokens, *, slot=0, marks=True, cut=None) -> list:
        S = len(tokens)
        parts = [(s, min(S, s + C)) for s in range(0, S, C)]
        parts = parts[:cut] if cut else parts
        did = []
        k0 = len(self.rt.pf.chunk_kinds)
        for k, (s, e) in enumerate(parts):
            word = slot
            if marks:
                word |= MARKED | (LAST if k == len(parts) - 1 else 0)
                word |= DECODER_CHUNK_FLAG if DC.is_decoder_chunk(e, S, CONE) else 0
            n0 = len(F.SYS.trace)
            self.rt.prefill_chunk(
                list(tokens[s:e]) + [1] * (C - (e - s)),
                self.kv,
                slot_id=word,
                actual_start=s,
                actual_end=e,
                request_id=self.request,
            )
            self.request += 1
            n = sum(1 for op in F.SYS.trace[n0:] if op[0] == "pf._chunk")
            if n == 0:
                did.append("R" if self.rt._prefix is not None and self.rt._prefix.last_restored else "S")
            else:
                did.append("C" if n == 1 else f"C+{n - 1}silent")
            if marks and DC.is_decoder_chunk(e, S, CONE):
                assert did[-1] not in ("S", "R"), f"decoder chunk [{s}, {e}) of {S} was {did[-1]}"
        self.last_did = did
        self.last_kinds = [kd for kd in self.rt.pf.chunk_kinds[k0:]]
        return did


@pytest.fixture
def marked(monkeypatch):
    monkeypatch.setattr(RT, "DECODER_MARKED", True)
    monkeypatch.setattr(RT, "DECODER_CONE", CONE)


_REFS: dict = {}


def cold_marked(layout, tokens, *, marks=True):
    """A cold marked prefill (prefix reuse off): an unrelated prompt, then ``tokens``; (export, live state, rows)."""
    key = (layout, tuple(tokens), marks)
    if key not in _REFS:
        with T._isolated_sys(), T._env(**T._base_env(layout, flag=False)):
            s = MarkedSystem()
            s.prompt(X, marks=marks)
            s.prompt(tokens, marks=marks)
            _REFS[key] = (s.state(0, len(tokens)), F.model_state(s.rt.pf, 0, len(tokens), live=False))
    return _REFS[key]


def assert_cold_marked(sys_, layout, tokens, what=""):
    (ref_exp, ref_live), ref_rows = cold_marked(layout, tokens)
    exp, live = sys_.state(0, len(tokens))
    bad = [k for k in ref_exp if not F.same(exp[k], ref_exp[k])]
    assert not bad, f"{what}: export tensors {bad} differ from a cold marked prefill's"
    assert F.same(F.model_state(sys_.rt.pf, 0, len(tokens), live=False), ref_rows), f"{what}: model rows differ"
    assert F.same(live, ref_live), f"{what}: counters / carries differ from a cold marked prefill's"


# ------------------------------------------------------------------------------------------------------- scenarios
def test_marked_burst_runs_the_decoder_on_its_cone_only(marked):
    with T._env(**T._base_env("cp2", flag=False)):
        s = MarkedSystem()
        s.prompt(X)
        S = len(A)  # 3C + 384
        assert s.prompt(A) == ["C"] * 4
        d0 = DC.decoder_start(S, C, CONE)
        assert d0 == 2 * C
        assert s.last_kinds == [(0, 0, False), (0, C, False), (0, 2 * C, True), (0, 3 * C, True)]
        assert s.rt.cone_checks[-1] == (0, S, d0) and DC.cone_exact(S, d0, CONE)
        pf = s.rt.pf
        for b in pf.blocks:
            assert b.state.kv_actual == S, (b.layer, b.state.kv_actual)  # the decoder caught up with the prompt
        st20 = pf.by_layer[CFG.first_decoder_layer].state
        assert st20.entry_count == S and bool((st20.slab_rm.data[0, 0, C : C + S] != 0).all()), "layer 20's slab rows"
        assert pf.decoder_from == d0
        # the decoder windows of the export were written (by the decoder chunks): rows of layer 39's ring are not all zero
        w39 = s.kv.win.data[s.kv.win_batch(0, 39)]
        assert bool((w39 != 0).any())
        assert F.SYS.late_allocs == []


def test_unmarked_headers_run_the_decoder_everywhere(marked):
    with T._env(**T._base_env("plain", flag=False)):
        s = MarkedSystem()
        s.prompt(A, marks=False)
        assert [k for _s, _st, k in s.last_kinds] == [True] * 4
        assert getattr(s.rt, "cone_checks", []) == []  # no marked last chunk: nothing to check


def test_a_short_prompt_is_all_cone(marked):
    with T._env(**T._base_env("plain", flag=False)):
        s = MarkedSystem()
        P = toks(C + 128, 41)  # shorter than C + CONE: the decoder from 0 (exact from the start)
        s.prompt(P)
        assert [k for _s, _st, k in s.last_kinds] == [True, True] and s.rt.cone_checks[-1] == (0, len(P), 0)


@pytest.mark.parametrize("layout", list(T.LAYOUTS))
def test_marked_reuse_sequence_equals_cold(marked, layout):
    """H1 on: every scenario of the default prefill's H1 test, now with marked decoder chunks -- skips and resumes before the
    cone, forced computes on it; the whole export and model state equal a cold marked prefill's."""
    with T._env(**T._base_env(layout, flag=True)):
        s = MarkedSystem(flag=True)
        s.prompt(X)
        s.prompt(A)
        assert_cold_marked(s, layout, A, "A computed")
        assert s.prompt(B)[0] == "S", "chunk 0 shared"
        assert_cold_marked(s, layout, B, "B after A")
        assert s.prompt(A)[0] == "S"
        assert_cold_marked(s, layout, A, "A after B")
        did = s.prompt(A2)
        assert did[:3] == ["S", "S", "S"], did  # A's chunks 0..2: none in A2's cone
        assert_cold_marked(s, layout, A2, "A2 extends A")
        did = s.prompt(A2)
        assert did[:4] == ["S"] * 4 and did[4:] == ["C", "C"], did  # an exact re-send: the cone is still computed
        assert_cold_marked(s, layout, A2, "A2 again")
        s.prompt(AP)
        assert_cold_marked(s, layout, AP, "AP: strict prefix on a boundary")
        s.prompt(A)
        assert_cold_marked(s, layout, A, "A after AP")
        s.prompt(LONG)
        assert s.prompt(SHORT_SIB) == ["C", "C", "C"]  # 2C + 128 tokens: chunk 0 already holds position S - CONE
        assert_cold_marked(s, layout, SHORT_SIB, "short sibling after LONG (all cone: computed, LONG's rows cleared)")
        assert F.SYS.late_allocs == [], f"zero blocks built after the island capture: {F.SYS.late_allocs}"
        st = s.rt._prefix.ret[0].stats
        assert st.forced_chunks > 0 and st.skipped_chunks > 0, s.rt._prefix.ret[0].summary()


def test_marked_store_sequence_equals_cold(marked):
    """H2 on (the host chunk store; refused before for any layout but the default prefill's): restores before the cone,
    forced computes on it; equal to a cold marked prefill."""
    layout = "cp2"
    with T._env(**ST.store_env(layout)):
        s = MarkedSystem(flag=True)
        assert s.rt._prefix.stores, "the store is on"
        s.prompt(X)
        s.prompt(A2)
        assert_cold_marked(s, layout, A2, "A2 computed")
        s.prompt(U)  # the device leaves the conversation
        did = s.prompt(A2)
        assert "R" in did and did[-1] != "R", did
        assert_cold_marked(s, layout, A2, "A2 restored from the store, cone computed")
        s.prompt(U)
        did = s.prompt(A)
        assert_cold_marked(s, layout, A, "A after U (restored prefix, cone computed)")
        assert s.rt._prefix.ret[0].stats.restored_chunks > 0


def test_store_carries_keep_the_default_layout(marked):
    """The checkpoints and the H2 tips hold the 21 carries of layers 0..20 (the default prefill's count), not 40."""
    with T._env(**ST.store_env("plain")):
        s = MarkedSystem(flag=True)
        assert len(PR.carry_states(s.rt.pf, 0)) == 21
        assert [b.layer for b in s.rt.pf.window_blocks()] == list(range(21))
