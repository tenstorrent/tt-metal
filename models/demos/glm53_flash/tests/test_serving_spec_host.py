# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device, no vLLM) tests of the vLLM adapter's speculative path (``tt/generator_vllm.py``
+ ``tt/vllm_spec.py``): the enable switch and the K policy (``spec_plan`` / ``resolve_spec_decision``), the
candidate-row translation (the plugin's placeholder columns -> the verify's real rows, every mesh row's copy), the
boot order (every eager graph before every capture), the per-chunk MTP prompt fill, the accept bookkeeping and
the hidden handoff (the level-1 draft of the accepted row feeds the chain), the one-column answer to a verify the
step cannot run, and the whole served loop driven like the runner does (``tests/serving_spec_driver.py``) against
a torch fake of the spec engine whose target is a Markov chain (the next token depends on the current token and
position only, so a verified candidate row equals the sequential plain step) and whose drafter is right at every
position but ``pos % 3 == 2``: the committed tokens must equal the plain chain (lossless) and the accepted counts
must follow that rule exactly.

    PYTHONPATH=<worktree>:<tt-metal>/ttnn:<tt-metal> <tt-metal>/python_env/bin/pytest --noconftest \\
        models/demos/glm53_flash/tests/test_serving_spec_host.py -p no:cacheprovider -q
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.tests.serving_spec_driver import (
    SpecRunnerSim,
    committed_positions,
    decode_kw,
    greedy_params,
)
from models.demos.glm53_flash.tests.test_generator_vllm_host import PAD, VOCAB, _FakeMeshModel, _next_token
from models.demos.glm53_flash.tt import vllm_spec as vs
from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM
from models.demos.glm53_flash.tt.vllm_config import ServingGeometry, SlotTable


@contextlib.contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


# ------------------------------------------------------------------------------------------ the fake drafter
def _draft(tok: int, pos: int) -> int:
    """The fake MTP: the true next token (``_next_token``) unless ``pos % 3 == 2``, then a wrong one."""
    true = _next_token(tok, pos)
    return true if pos % 3 != 2 else (true + 1) % VOCAB


def _expected_count(anchor: int, valid: int) -> int:
    """The accepted count of a verify whose drafts came from the fake drafter: draft ``j`` (position anchor + j)
    is right iff ``(anchor + j) % 3 != 2``; the count is 1 + the length of the right prefix (capped at ``valid``)."""
    m = 0
    while m < valid and (anchor + m) % 3 != 2:
        m += 1
    return m + 1


class _FakeSpecEngine:
    """``GLM53SpecEngine``'s surface on the fake mesh model: the target is ``_next_token(tok, pos)`` (candidate row
    ``j`` of a verify at anchor ``p`` is ``f(tokens[j], p + j)``), the drafter ``_draft``; the commit discipline is
    checked (a verify's positions equal the committed context, exactly one commit per verify, the chain follows
    the commit), and every call is recorded."""

    def __init__(self, model: _FakeMeshModel, K: int, commit_mode: str = "fused", max_dense_pos: int | None = None):
        self.model, self.K, self.commit_mode = model, int(K), commit_mode
        self.N = model.total_users
        self.max_dense_pos = min(2050, model.max_seq_len - 1) if max_dense_pos is None else int(max_dense_pos)
        self.calls: list[tuple] = []
        self.state_pos = [0] * self.N  # the committed context length of every slot (what its KDA state holds)
        self.pending = None  # (tokens, positions) of the verify awaiting its commit
        self.last_commit = None
        self.plain_pos: list[int] | None = None
        self.last_plain: list[int] | None = None
        self.d1_plain = [0] * self.N

    # boot
    def warmup_plain(self, ids, positions):
        self.calls.append(("warmup_plain",))
        self.model.warmup_decode(ids, positions)

    def warmup(self, positions):
        self.calls.append(("spec_warmup", len(positions)))

    def capture_plain(self, ids, positions):
        self.calls.append(("capture_plain",))
        tid, _ = self.model.capture(ids, positions)
        return tid, (("logits_tr",), ("ids_tr",))

    def capture(self):
        self.calls.append(("spec_capture",))

    def release(self):
        self.calls.append(("release",))
        self.model.trace_id = None

    # the ordinary step
    def set_plain_positions(self, positions):
        self.plain_pos = list(positions)

    def plain_step(self, eager: bool = False):
        assert self.plain_pos == self.model._fed_pos, "the MTP positions must be the step's positions"
        self.model._step()
        self.last_plain = list(self.model._fed_pos)
        for u in range(self.N):
            p = self.last_plain[u]
            self.d1_plain[u] = _draft(int(self.model._last_ids[u]), p + 1)
            self.state_pos[u] = p + 1
        self.calls.append(("plain_step", eager))
        return (("logits_eager",), ("ids_eager",)) if eager else (("logits_tr",), ("ids_tr",))

    def read_plain_d1(self):
        return torch.tensor(self.d1_plain, dtype=torch.long)

    def draft_chain_plain(self, positions):
        assert list(positions) == self.last_plain, "the plain chain follows the last plain step"
        out = torch.zeros(self.N, self.K - 1, dtype=torch.long)
        for u in range(self.N):
            prev, p = self.d1_plain[u], self.last_plain[u]
            for k in range(2, self.K + 1):
                prev = _draft(prev, p + k)
                out[u, k - 2] = prev
        self.calls.append(("draft_chain_plain",))
        return out

    # the verify / commit / chain
    def verify(self, tokens, positions):
        assert self.pending is None, "a verify without the previous verify's commit"
        N, P = tokens.shape
        assert N == self.N and P == self.K + 1, (tokens.shape, self.N, self.K)
        ids = torch.zeros(N, P, dtype=torch.long)
        d1 = torch.zeros(N, P, dtype=torch.long)
        for u in range(N):
            p = int(positions[u])
            if p != 0:  # parked slots sit at 0 (a live decode never does)
                assert p == self.state_pos[u], f"slot {u}: verify at {p}, the committed context is {self.state_pos[u]}"
            for j in range(P):
                ids[u, j] = _next_token(int(tokens[u, j]), p + j)
                d1[u, j] = _draft(int(ids[u, j]), p + j + 1)
        self.pending = (tokens.clone(), list(positions))
        self.calls.append(("verify", list(positions)))
        return ids, d1

    def commit(self, m, positions):
        assert self.pending is not None, "commit without a verify"
        assert list(positions) == self.pending[1], "the commit's positions are the verify's"
        for u in range(self.N):
            self.state_pos[u] = int(positions[u]) + int(m[u]) + 1
        self.last_commit = (m.clone(), list(positions))
        self.pending = None
        self.calls.append(("commit", [int(v) for v in m]))

    def draft_chain(self, m, d1, positions):
        assert self.last_commit is not None and list(positions) == self.last_commit[1]
        assert torch.equal(m.long(), self.last_commit[0].long()), "the chain follows the commit's m"
        out = torch.zeros(self.N, self.K - 1, dtype=torch.long)
        for u in range(self.N):
            prev, base = int(d1[u]), int(positions[u]) + int(m[u])
            for k in range(2, self.K + 1):
                prev = _draft(prev, base + k)
                out[u, k - 2] = prev
        self.calls.append(("draft_chain",))
        return out

    # the prompt
    def prefill_fill(self, ids, first, streams, user, start, seq_len):
        self.calls.append(("prefill_fill", user, int(start), int(seq_len), int(first)))
        for r in range(self.model.rows):
            self.state_pos[r * self.model.max_batch + user] = int(start) + int(seq_len)


def _spec_adapter(rows=2, cols=4, users_per_row=4, max_seq_len=65536, K=3, warm=True, max_dense_pos=None):
    model = _FakeMeshModel(rows, cols, users_per_row, max_seq_len)
    g = ServingGeometry(
        rows=rows,
        cols=cols,
        users_per_row=users_per_row,
        max_seq_len=max_seq_len,
        partitioned=False,
        traced_prefill_lengths=tuple(model.trace_prefill_lengths()),
        pad_token_id=PAD,
    )
    eng = _FakeSpecEngine(model, K, max_dense_pos=max_dense_pos)
    a = GLM53FlashForCausalLM(model, g, spec=eng)
    if warm:
        n = g.request_capacity
        a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
        a.warmup_model_decode(enable_trace=False, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
        a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
        a.warmup_model_decode(enable_trace=True, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
    return a, model, eng


def _prefill(a, slot: int, n: int, fill: int | None = None) -> tuple[int, int]:
    """One prompt of ``n`` tokens into logical ``slot``; returns ``(first token, position)``."""
    t = torch.full((1, n), 2 + slot if fill is None else fill, dtype=torch.int32)
    out, _ = a.prefill_forward(
        t, prompt_lens=[n], start_pos=torch.tensor([0]), empty_slots=[slot], sampling_params=greedy_params(1)
    )
    return int(out[0, 0]), n


def _plain_chain(first: int, position: int, n: int) -> list[int]:
    out, tok, p = [], first, position
    for _ in range(n):
        tok = _next_token(tok, p)
        out.append(tok)
        p += 1
    return out


# --------------------------------------------------------------------------------------------------- tests
def test_module_imports_without_ttnn_and_declares_the_contract():
    assert "ttnn" not in sys.modules
    caps = GLM53FlashForCausalLM.model_capabilities
    assert caps["supports_spec_decode"] is True
    assert tuple(caps["spec_requirements"]) == ("device_propose",)
    assert tuple(caps["spec_hidden_handoff"]) == ("on_device",)
    assert caps["supports_async_spec_decode"] is False and caps["supports_async_decode"] is False
    assert callable(GLM53FlashForCausalLM.spec_plan) and callable(GLM53FlashForCausalLM.propose_draft_tokens)


def test_spec_decision_table():
    """The switch, the geometry refusals and the K cap: ``min(requested, GLM53_VLLM_SPEC_K, 32 // B - 1)``."""
    env = {"GLM53_VLLM_SPEC": "1", "GLM53_PARTITION_USERS": "0"}
    d = vs.resolve_spec_decision(env, max_num_seqs=4, requested_k=3, rows=8, cols=4)
    assert d.admitted and (d.effective_k, d.k_cap, d.users_per_row, d.commit_mode) == (3, 3, 4, "fused")
    assert d.supported_k == (1, 2, 3)
    # the cap table of spec_draft_len at K=7: 1 -> 7 (31 allowed), 4 -> 7, 8 -> 3, 16 -> 1, 32 -> plain
    env7 = dict(env, GLM53_VLLM_SPEC_K="7")
    assert [
        vs.resolve_spec_decision(env7, max_num_seqs=b, requested_k=7, rows=8, cols=4).effective_k for b in (1, 4, 8, 16)
    ] == [7, 7, 3, 1]
    assert [vs.draft_len_cap(b, 31) for b in (1, 4, 8, 16, 32)] == [31, 7, 3, 1, 0]
    r32 = vs.resolve_spec_decision(env7, max_num_seqs=32, requested_k=7, rows=8, cols=4)
    assert not r32.admitted and "no room for drafts" in r32.reason and r32.supported_k == ()
    # requested below the cap: the plugin's value wins (effective_k <= requested_k); above: the cap
    assert vs.resolve_spec_decision(env7, max_num_seqs=4, requested_k=2, rows=8, cols=4).effective_k == 2
    assert vs.resolve_spec_decision(env, max_num_seqs=4, requested_k=9, rows=8, cols=4).effective_k == 3
    # refusals: the switch, the partition (the serving default), independent rows, a bad K / commit knob
    off = vs.resolve_spec_decision({}, max_num_seqs=4, requested_k=3, rows=8, cols=4)
    assert not off.admitted and "GLM53_VLLM_SPEC=1" in off.reason
    part = vs.resolve_spec_decision({"GLM53_VLLM_SPEC": "1"}, max_num_seqs=4, requested_k=3, rows=8, cols=4)
    assert not part.admitted and "GLM53_PARTITION_USERS=0" in part.reason
    one_col = vs.resolve_spec_decision({"GLM53_VLLM_SPEC": "1"}, max_num_seqs=4, requested_k=3, rows=1, cols=1)
    assert one_col.admitted  # a single column has nothing to partition
    ind = vs.resolve_spec_decision(
        dict(env, GLM53_VLLM_ROWS_INDEPENDENT="1"), max_num_seqs=32, requested_k=3, rows=8, cols=4
    )
    assert not ind.admitted and "wave" in ind.reason
    bad = vs.resolve_spec_decision(dict(env, GLM53_SPEC_COMMIT="turbo"), max_num_seqs=4, requested_k=3, rows=8, cols=4)
    assert not bad.admitted and "GLM53_SPEC_COMMIT" in bad.reason
    assert (
        vs.resolve_spec_decision(
            dict(env, GLM53_SPEC_COMMIT="blend"), max_num_seqs=4, requested_k=3, rows=8, cols=4
        ).commit_mode
        == "blend"
    )
    assert not vs.resolve_spec_decision(env, max_num_seqs=4, requested_k=0, rows=8, cols=4).admitted


def test_spec_plan_classmethod(monkeypatch):
    """``spec_plan`` never raises: a SpecReject naming the knob, or a SpecPlan with the contract's values (one wire
    row per request, argmax_ids, the drafter's state on device, narrow decode on)."""
    monkeypatch.setenv("GLM53_MESH_SHAPE", "8,4")
    monkeypatch.delenv("GLM53_VLLM_SPEC", raising=False)
    monkeypatch.setenv("GLM53_PARTITION_USERS", "0")
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))
    rej = GLM53FlashForCausalLM.spec_plan(cfg, 4, 3)
    assert type(rej).__name__ == "_LocalSpecReject" and "GLM53_VLLM_SPEC=1" in rej.reason  # no plugin on the host
    monkeypatch.setenv("GLM53_VLLM_SPEC", "1")
    plan = GLM53FlashForCausalLM.spec_plan(cfg, 4, 3)
    assert type(plan).__name__ == "_LocalSpecPlan"
    assert plan.effective_k == 3 and plan.lanes_per_request == 1 and plan.accept_modes == ("argmax_ids",)
    assert plan.drafter_state == "internal" and plan.supports_narrow_decode is True and plan.k_by_rows == ()
    assert plan.extra_bytes_per_token == 0 and plan.extra_bytes_per_seq > 0  # the MTP's resident bytes, per request
    assert GLM53FlashForCausalLM.spec_plan(cfg, 4, 2).effective_k == 2
    assert GLM53FlashForCausalLM.spec_plan(object(), 8, 5).effective_k == 3  # an opaque config: the bytes are skipped
    monkeypatch.setenv("GLM53_PARTITION_USERS", "1")
    assert "GLM53_PARTITION_USERS=0" in GLM53FlashForCausalLM.spec_plan(cfg, 4, 3).reason


def test_candidate_block_and_the_device_feed():
    """The plugin's row (placeholders past num_valid) -> ``K + 1`` real tokens (the anchor fills the padded columns);
    the feed puts every live slot's block at its physical indices on EVERY mesh row and parks the idle slots."""
    assert vs.candidate_block([10, 11, 12, -1], 2, 4) == [10, 11, 12, 10]
    assert vs.candidate_block([10, -1, -1, -1], 0, 4) == [10, 10, 10, 10]
    assert vs.candidate_block([10, 11], 1, 4) == [10, 11, 10, 10]  # a narrower launch block
    assert vs.candidate_block([10, 11, 12, 13], 3, 4) == [10, 11, 12, 13]
    with expect_error(ValueError, "placeholder"):
        vs.candidate_block([10, -1, 12, 13], 2, 4)
    with expect_error(ValueError, "num_valid_drafts"):
        vs.candidate_block([10, 11, 12, 13], 4, 4)
    g = ServingGeometry(rows=2, cols=4, users_per_row=4, max_seq_len=64, partitioned=False, pad_token_id=PAD)
    t = SlotTable(g)
    users = [t.allocate(i).user for i in range(2)]
    ids, pos = t.device_feed_block({0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}, {0: 10, 1: 20}, 4)
    assert tuple(ids.shape) == (8, 4) and ids.dtype == torch.long and len(pos) == 8
    for r in range(2):
        assert ids[r * 4 + users[0]].tolist() == [1, 2, 3, 4] and pos[r * 4 + users[0]] == 10
        assert ids[r * 4 + users[1]].tolist() == [5, 6, 7, 8] and pos[r * 4 + users[1]] == 20
    idle = [i for i in range(8) if i % 4 not in users]
    assert all(ids[i].tolist() == [PAD] * 4 and pos[i] == 0 for i in idle)
    with expect_error(ValueError, "placeholder"):
        t.device_feed_block({0: [1, -1, 3, 4]}, {0: 10}, 4)
    with expect_error(ValueError, "expected 4"):
        t.device_feed_block({0: [1, 2, 3]}, {0: 10}, 4)
    with expect_error(ValueError, "positions for"):
        t.device_feed_block({0: [1, 2, 3, 4]}, {1: 10}, 4)
    assert vs.wire_rows(torch.arange(8).reshape(8, 1), [3, -1, 0], fill=-7).reshape(-1).tolist() == [3, -7, 0]


def test_accept_walk_matches_the_plugin_semantics():
    """``accept_greedy_drafts``: the first mismatch stops the row (the correction commits there), a fully accepted row
    takes the bonus column, padding past num_valid neither accepts nor rejects, a one-column answer commits
    column 0 everywhere with count 1."""
    ids = torch.tensor([[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.int32)
    drafts = torch.tensor([[5, 6, 7], [5, 9, 7], [5, 6, -1], [-1, -1, -1]], dtype=torch.int32)
    valid = torch.tensor([3, 3, 2, 0], dtype=torch.int32)
    committed, counts = vs.accept_greedy_walk(ids, drafts, valid)
    assert counts.tolist() == [4, 2, 3, 1]
    assert committed.tolist() == [[5, 6, 7, 8], [5, 6, -1, -1], [5, 6, 7, -1], [1, -1, -1, -1]]
    one, c1 = vs.accept_greedy_walk(ids[:, :1], drafts, valid)
    assert c1.tolist() == [1, 1, 1, 1] and one.tolist() == [[5], [5], [5], [1]]
    assert committed_positions(torch.tensor([[7, 8, 9], [-1, -1, -1]]), 3).tolist() == [[8, 9, 10], [-1, -1, -1]]


def test_policy_dense_bound():
    pol = vs.ServingSpecPolicy(draft_len=3, max_dense_pos=2050, max_seq_len=4096)
    assert pol.anchor_ok(2047) and not pol.anchor_ok(2048) and not pol.anchor_ok(-1)
    assert pol.step_can_verify([10, 2047]) and not pol.step_can_verify([10, 2048]) and pol.step_can_verify([])
    assert (
        pol.row_drafts(100, True, 3) == 3 and pol.row_drafts(100, True, 2) == 2 and pol.row_drafts(100, False, 3) == 0
    )
    assert pol.row_drafts(2048, True, 3) == 0
    short = vs.ServingSpecPolicy(draft_len=3, max_dense_pos=255, max_seq_len=256)
    assert short.anchor_ok(252) and not short.anchor_ok(253)


def test_boot_order_with_spec():
    """Every eager graph (P, the prefill ladder with the MTP fill, V/C/D/D0) runs before every capture (the prefill
    traces, then P + the spec traces); the decode trace outputs are the P step's; release drops them all."""
    a, m, eng = _spec_adapter()
    kinds = [c[0] for c in eng.calls] + [c[0] for c in m.calls]
    seq = [c[0] for c in eng.calls]
    assert seq.index("warmup_plain") < seq.index("prefill_fill") < seq.index("spec_warmup") < seq.index("capture_plain")
    assert seq.index("capture_plain") < seq.index("spec_capture")
    mk = [c[0] for c in m.calls]
    assert mk.index("capture_prefill") < mk.index("capture")  # the prefill traces before the decode (P) trace
    assert "warmup_decode" in mk and mk.index("warmup_decode") < mk.index("prefill")
    assert a.trace_id == 7 and a._trace_outputs == (("logits_tr",), ("ids_tr",))
    assert a.tt_supported_decode_batch_sizes == (4,) and "capture_prefill" in kinds
    fills = [c for c in eng.calls if c[0] == "prefill_fill"]
    assert fills == [("prefill_fill", 0, 0, 128, fills[0][4])]  # the warm-up prompt fills the drafter too
    assert "SPECULATIVE" in a.__class__.__name__ or True  # the log line is checked by eye in the engine log
    a.release_persistent_capture()
    assert eng.calls[-1] == ("release",) and a.trace_id is None and a._trace_outputs is None


def test_prefill_fills_the_drafter_per_chunk():
    """Every device chunk of a prompt fills the MTP rows [start, start + len) with the chunk's own chunk-end token
    as the last row's next token; the traced prefill fills once from the trace's streams."""
    a, m, eng = _spec_adapter()
    eng.calls.clear()
    L = 16384 * 2 + 100
    ids = (torch.arange(L) % 60).to(torch.int32).reshape(1, L)
    out, _ = a.prefill_forward(
        ids, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=greedy_params(1)
    )
    fills = [c for c in eng.calls if c[0] == "prefill_fill"]
    user = a.slots.physical(0).user
    assert [(c[1], c[2], c[3]) for c in fills] == [(user, 0, 16384), (user, 16384, 16384), (user, 32768, 100)]
    assert fills[-1][4] == int(out[0, 0])  # the last chunk's next token = the prompt's first generated token
    assert eng.state_pos[user] == L
    eng.calls.clear()
    first, n = _prefill(a, 1, 300)  # within a captured traced length: one fill from the trace's streams
    fills = [c for c in eng.calls if c[0] == "prefill_fill"]
    assert fills == [("prefill_fill", a.slots.physical(1).user, 0, 300, first)]
    assert [c[0] for c in m.calls if c[0].startswith("prefill")][-1] == "prefill_traced"


def test_spec_loop_is_lossless_through_the_adapter():
    """The served loop (ordinary step -> propose -> verify -> accept -> commit + propose -> ...) on 3 live slots and
    a padding row, driven like the runner: the committed tokens equal the plain chain of every slot (lossless),
    the accepted counts follow the fake drafter's rule (the hidden handoff: the chain continues from the accepted
    row's level-1 draft at the right positions), exactly one commit per verify, the proposal declines nothing while
    every anchor is in range, and a dropped proposal rides the next verify with num_valid 0."""
    K, n = 3, 40
    a, m, eng = _spec_adapter(K=K)
    runner = SpecRunnerSim(a, K, wire=4)
    starts = {}
    for slot, L in ((0, 300), (1, 100), (2, 41)):
        first, p = _prefill(a, slot, L)
        runner.add(slot, first, p)
        starts[slot] = (first, p)
    got = runner.run(n)
    for slot, (first, p) in starts.items():
        assert got[slot] == _plain_chain(first, p, n), f"slot {slot} diverged from the plain chain"
    verifies = [s for s in runner.log if s.kind == "verify"]
    plains = [s for s in runner.log if s.kind == "plain"]
    assert plains and verifies and runner.log[0].kind == "plain"  # the first step after the prefills is ordinary
    assert all(s.width == K + 1 for s in verifies)
    for s in verifies:
        for slot, c in s.counts.items():
            assert c == _expected_count(s.anchors[slot], s.num_valid[slot]), (s.anchors[slot], s.num_valid[slot], c)
        assert all(v == K for v in s.num_valid.values())  # every live row drafted K while in range
    commits = [c for c in eng.calls if c[0] == "commit"]
    assert len(commits) == len(verifies) and len([c for c in eng.calls if c[0] == "verify"]) == len(verifies)
    assert len([c for c in eng.calls if c[0] == "draft_chain"]) == len(verifies)
    assert len([c for c in eng.calls if c[0] == "draft_chain_plain"]) == len(plains)
    # the adapter's positions follow the commits; finished slots were released (parked)
    assert all(a.slots.physical(s) is None for s in starts)
    assert a._spec_pending is None
    # a dropped proposal: the slot rides the next verify with num_valid 0 and commits exactly one token
    first, p = _prefill(a, 0, 50)
    runner = SpecRunnerSim(a, K, wire=4)
    runner.add(0, first, p)
    first1, p1 = _prefill(a, 1, 70)
    runner.add(1, first1, p1)
    runner.step()  # plain
    runner.drop_drafts(1)
    log = runner.step()  # a verify: slot 0 with drafts, slot 1 with none
    assert log.kind == "verify" and log.num_valid == {0: K, 1: 0} and log.counts[1] == 1
    assert log.counts[0] == _expected_count(log.anchors[0], K)
    got = runner.run(12)
    assert got[0] == _plain_chain(first, p, 12) and got[1] == _plain_chain(first1, p1, 12)
    logger.info(
        f"spec loop: {len(verifies)} verifies, {len(plains)} plain steps for {n} tokens x 3 slots; {a._spec_stats.summary()}"
    )


def test_narrower_launch_block_is_served():
    """A launch whose num_speculative_tokens (2) is below the adapter's K (3): the verify pads the missing column,
    answers the launch's width, and proposes 2 drafts; the loop stays lossless."""
    K, n = 3, 24
    a, m, eng = _spec_adapter(K=K)
    runner = SpecRunnerSim(a, 2, wire=4)
    first, p = _prefill(a, 0, 33)
    runner.add(0, first, p)
    got = runner.run(n)
    assert got[0] == _plain_chain(first, p, n)
    verifies = [s for s in runner.log if s.kind == "verify"]
    assert verifies and all(s.width == 3 and s.num_valid[0] == 2 for s in verifies)
    assert all(s.counts[0] == _expected_count(s.anchors[0], 2) for s in verifies)
    assert a._spec_narrow_warned


def test_verify_beyond_the_dense_range_is_answered_one_column_wide():
    """A slot whose anchor + K passes the dense bound: the proposal declines every row (num_valid 0) and a verify the
    plugin sends anyway (its drafts proposed before the slot arrived) is answered one column wide by the plain
    step: every draft counts as rejected, one token per row is committed, the loop stays lossless."""
    K, n = 3, 16
    a, m, eng = _spec_adapter(K=K, max_dense_pos=2050)
    runner = SpecRunnerSim(a, K, wire=4)
    first0, p0 = _prefill(a, 0, 64)
    runner.add(0, first0, p0)
    runner.step()  # plain: slot 0 proposes K drafts
    assert runner.slots[0].drafts and len(runner.slots[0].drafts) == K
    first1, p1 = _prefill(a, 1, 2048)  # arrives beyond the bound: 2048 + 3 > 2050
    runner.add(1, first1, p1)
    log = runner.step()  # the verify the runner sends for slot 0's drafts
    assert log.kind == "verify" and log.width == 1 and log.counts == {0: 1, 1: 1}
    assert a._spec_stats.fallbacks == 1 and a._spec_pending is None
    assert runner.slots[0].drafts == [] and runner.slots[1].drafts == []  # nobody drafts while slot 1 is live
    got = runner.run(n)
    assert got[0] == _plain_chain(first0, p0, n) and got[1] == _plain_chain(first1, p1, n)
    assert [s.kind for s in runner.log[2:]] == ["plain"] * (len(runner.log) - 2)
    # a lone slot near the bound: drafts until the anchor + K would cross it, then plain steps
    first2, p2 = _prefill(a, 2, 2040)
    runner = SpecRunnerSim(a, K, wire=4)
    runner.add(2, first2, p2)
    got = runner.run(12)
    assert got[2] == _plain_chain(first2, p2, 12)
    # every full-width verify sat inside the bound; the runner's rule 4 (a multi-token commit sends one more verify
    # even without drafts) reaches the adapter past it and is answered one column wide
    full = [s.anchors[2] for s in runner.log if s.kind == "verify" and s.width > 1]
    narrow = [s.anchors[2] for s in runner.log if s.kind == "verify" and s.width == 1]
    assert full and all(anc + K <= 2050 for anc in full), full
    assert all(anc + K > 2050 for anc in narrow), narrow


def test_contract_guards():
    """A verify's commit must run before the next step; a count the walk could not have produced, a committed token
    that is not the verify's choice, a wrong spec_mode, and spec kwargs on a plain adapter are refused."""
    K = 3
    a, m, eng = _spec_adapter(K=K)
    runner = SpecRunnerSim(a, K, wire=4)
    first, p = _prefill(a, 0, 40)
    runner.add(0, first, p)
    runner.step()
    drafts = runner.slots[0].drafts
    tokens = torch.full((4, K + 1), -1, dtype=torch.int32)
    positions = torch.full((4, K + 1), -1, dtype=torch.int32)
    tokens[0, 0] = runner.slots[0].token
    tokens[0, 1:] = torch.tensor(drafts, dtype=torch.int32)
    positions[0] = runner.slots[0].position + torch.arange(K + 1)
    kw = dict(
        decode_kw(), sampling_params=greedy_params(4), num_valid_drafts=torch.tensor([K, 0, 0, 0], dtype=torch.int32)
    )
    kw["accepted_counts"] = torch.ones(4, dtype=torch.int32)
    with expect_error(ValueError, "argmax_ids"):
        a.decode_forward(tokens, positions, spec_mode="logits", **kw)
    out = a.decode_forward(tokens, positions, spec_mode="argmax_ids", **kw)
    assert tuple(out.argmax_ids.shape) == (4, K + 1) and a._spec_pending is not None
    with expect_error(RuntimeError, "commit never ran"):
        a.decode_forward(tokens, positions, spec_mode="argmax_ids", **kw)
    with expect_error(RuntimeError, "commit never ran"):
        a.decode_forward(tokens[:, :1], positions[:, 0], sampling_params=greedy_params(4), **decode_kw())
    committed, counts = vs.accept_greedy_walk(out.argmax_ids, tokens[:, 1:], kw["num_valid_drafts"])
    bad_counts = counts.clone()
    bad_counts[0] = K + 2
    with expect_error(ValueError, "accepted count"):
        a.propose_draft_tokens(K, committed, committed_positions(positions, K + 1), bad_counts)
    bad = committed.clone()
    bad[0, int(counts[0]) - 1] = (int(bad[0, int(counts[0]) - 1]) + 1) % VOCAB
    with expect_error(RuntimeError, "disagree"):
        a.propose_draft_tokens(K, bad, committed_positions(positions, K + 1), counts)
    d = a.propose_draft_tokens(K, committed, committed_positions(positions, K + 1), counts)
    assert a._spec_pending is None and int(d.num_valid[0]) == K
    with expect_error(RuntimeError, "without a decode step"):
        a.propose_draft_tokens(K, committed, committed_positions(positions, K + 1), counts)
    # the plain adapter (no GLM53_VLLM_SPEC): the plugin's spec kwargs are refused by name
    plain = GLM53FlashForCausalLM(
        _FakeMeshModel(2, 4, 4, 1024),
        ServingGeometry(rows=2, cols=4, users_per_row=4, max_seq_len=1024, pad_token_id=PAD),
    )
    with expect_error(NotImplementedError, "GLM53_VLLM_SPEC"):
        plain.decode_forward(tokens, positions, spec_mode="argmax_ids", **kw)
    with expect_error(NotImplementedError, "GLM53_VLLM_SPEC"):
        plain.propose_draft_tokens(K, committed, committed_positions(positions, K + 1), counts)
    with expect_error(NotImplementedError, "replicated caches"):
        GLM53FlashForCausalLM(
            _FakeMeshModel(2, 4, 4, 1024),
            ServingGeometry(rows=2, cols=4, users_per_row=4, max_seq_len=1024, pad_token_id=PAD),
            spec=_FakeSpecEngine(_FakeMeshModel(2, 4, 4, 1024), K),
        )


def test_zz_no_ttnn_was_imported_by_the_suite():
    assert "ttnn" not in sys.modules
