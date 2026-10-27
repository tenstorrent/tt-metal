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

    def warmup_prefill_fill(self, lengths):
        self.calls.append(("warmup_prefill_fill", tuple(int(T) for T in lengths)))
        self.warmed_fill_lengths = list(lengths)

    # the allocation discipline: a constant footprint unless a test bumps it
    allocated = 1000

    def dram_allocated(self):
        return self.allocated


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
    assert seq.index("warmup_plain") < seq.index("prefill_fill") < seq.index("warmup_prefill_fill")
    assert seq.index("warmup_prefill_fill") < seq.index("spec_warmup") < seq.index("capture_plain")
    assert seq.index("capture_plain") < seq.index("spec_capture")
    mk = [c[0] for c in m.calls]
    assert mk.index("capture_prefill") < mk.index("capture")  # the prefill traces before the decode (P) trace
    assert "warmup_decode" in mk and mk.index("warmup_decode") < mk.index("prefill")
    # the drafter's fill warm-up (every served chunk length) precedes the first capture of any kind: the engine's
    # call count before ``capture_prefill`` is what the adapter made with every eager graph still uncompiled
    before_capture = len([c for c in m.calls[: mk.index("capture_prefill")]])
    assert before_capture > 0 and "warmup_prefill_fill" in seq[: seq.index("capture_plain")]
    assert a.trace_id == 7 and a._trace_outputs == (("logits_tr",), ("ids_tr",))
    assert a.tt_supported_decode_batch_sizes == (4,) and "capture_prefill" in kinds
    fills = [c for c in eng.calls if c[0] == "prefill_fill"]
    assert fills == [("prefill_fill", 0, 0, 128, fills[0][4])]  # the warm-up prompt fills the drafter too
    warmed = [c for c in eng.calls if c[0] == "warmup_prefill_fill"]
    # traced lengths 256..2048 (the fake's), the eager buckets up to the 16384 device chunk, within 65536 positions
    assert warmed == [("warmup_prefill_fill", (256, 512, 1024, 2048, 16384))]
    a.release_persistent_capture()
    assert eng.calls[-1] == ("release",) and a.trace_id is None and a._trace_outputs is None


def test_mtp_fill_plan_pins_the_row_contract():
    """``MtpFillPlan`` for the traced form (a 32-token prompt in the T=128 trace on 2 mesh rows: 64-row slabs) and
    the eager form (a 230-token chunk padded to 256: 128-row slabs; a 100-token continuation at start 16384): the
    ids of mesh row r are the chunk's tokens shifted by one, rows ``r t_loc ..``, the chunk-end token at row
    ``seq_len - 1`` (its mesh row = the owner row), the pad id past it; the row mask is 1 on rows < seq_len only."""
    chunk = torch.arange(100, 132)  # 32 tokens
    plan = vs.mtp_fill_plan(chunk, 999, rows=2, t_loc=64, seq_len=32, start=0, pad_token_id=PAD)
    assert (plan.t_pad, plan.masked, plan.owner_row, plan.last_local_row) == (128, True, 0, 31)
    assert tuple(plan.next_ids.shape) == (1, 1, 1, 128) and plan.next_ids.dtype == torch.int32
    assert plan.slab(0).tolist() == list(range(101, 132)) + [999] + [PAD] * 32
    assert plan.slab(1).tolist() == [PAD] * 64
    assert tuple(plan.row_mask.shape) == (1, 1, 128, 32) and plan.row_mask.dtype == torch.bfloat16
    assert plan.mask_slab(0).tolist() == [1.0] * 32 + [0.0] * 32 and plan.mask_slab(1).tolist() == [0.0] * 64
    assert plan.row_mask[0, 0, :, 1:].abs().sum() == 0  # column 0 only (the ones-row matmul expands it)
    # the eager form: 230 tokens padded to 256 (128 per mesh row): the chunk-end token sits on mesh row 1, row 101
    chunk = torch.arange(1000, 1230)
    plan = vs.mtp_fill_plan(chunk, 7, rows=2, t_loc=128, seq_len=230, start=0, pad_token_id=PAD)
    assert (plan.t_pad, plan.owner_row, plan.last_local_row) == (256, 1, 101)
    assert plan.slab(0).tolist() == list(range(1001, 1129))
    assert plan.slab(1).tolist() == list(range(1129, 1230)) + [7] + [PAD] * 26
    assert plan.mask_slab(0).tolist() == [1.0] * 128 and plan.mask_slab(1).tolist() == [1.0] * 102 + [0.0] * 26
    # a continuation chunk: start 16384, 100 tokens, 128-row slabs; one mesh row: a 256-row slab is the whole chunk
    plan = vs.mtp_fill_plan(torch.arange(100), 5, rows=2, t_loc=128, seq_len=100, start=16384, pad_token_id=PAD)
    assert plan.start == 16384 and plan.slab(0).tolist() == list(range(1, 100)) + [5] + [PAD] * 28
    one = vs.mtp_fill_plan(torch.arange(256), 5, rows=1, t_loc=256, seq_len=256, start=0, pad_token_id=PAD)
    assert not one.masked and one.slab(0).tolist() == list(range(1, 256)) + [5]
    for bad, message in (
        (dict(rows=2, t_loc=64, seq_len=129, start=0), "outside"),  # beyond the padded rows
        (dict(rows=2, t_loc=64, seq_len=0, start=0), "outside"),
        (dict(rows=2, t_loc=48, seq_len=32, start=0), "multiple of 32"),  # a slab that is not a tile multiple
        (dict(rows=2, t_loc=64, seq_len=32, start=16), "multiple of 32"),  # an unaligned start
    ):
        with expect_error(ValueError, message):
            vs.mtp_fill_plan(torch.arange(bad["seq_len"]), 1, pad_token_id=PAD, **bad)
    with expect_error(ValueError, "holds"):
        vs.mtp_fill_plan(torch.arange(31), 1, rows=2, t_loc=64, seq_len=32, start=0, pad_token_id=PAD)


def test_mtp_fill_lengths_cover_the_traced_and_eager_shapes():
    """The lengths the fill is warmed at: the traced lengths plus every eager bucket up to the device chunk (and the
    chunk's own padded length), each a multiple of the row unit and within the caches."""
    assert vs.mtp_fill_lengths((128, 256), rows=2, device_prefill_chunk=16384, max_seq_len=256) == [128, 256]
    assert vs.mtp_fill_lengths((256, 512, 1024, 2048), rows=8, device_prefill_chunk=16384, max_seq_len=2048) == [
        256,
        512,
        1024,
        2048,
    ]
    assert vs.mtp_fill_lengths((256, 512, 1024, 2048), rows=8, device_prefill_chunk=16384, max_seq_len=1 << 20) == [
        256,
        512,
        1024,
        2048,
        16384,
    ]
    # a 4096-token device chunk pads to the 16384 bucket (the smallest bucket that holds it): warmed there, not at 4096
    assert vs.mtp_fill_lengths((), rows=8, device_prefill_chunk=4096, max_seq_len=1 << 20) == [
        256,
        512,
        1024,
        2048,
        16384,
    ]
    assert vs.mtp_fill_lengths((128,), rows=2, device_prefill_chunk=256, max_seq_len=1024, buckets=()) == [128, 256]
    assert (
        vs.eager_pad_length(230, 64) == 256
        and vs.eager_pad_length(128, 64) == 256
        and vs.eager_pad_length(17000, 64) == 17024
    )
    assert vs.prefill_buckets({"GLM53_PREFILL_BUCKETS": "0"}) == () and vs.prefill_buckets({}) == (
        256,
        512,
        1024,
        2048,
        16384,
    )
    assert vs.prefill_buckets({"GLM53_PREFILL_BUCKETS": "512, 256"}) == (256, 512)


class _Rec:
    """A recorded device tensor of the stubbed fill."""

    def __init__(self, tag, shape, dtype=None, extra=None):
        self.tag, self.shape, self.dtype, self.extra = tag, tuple(shape), dtype, extra


def test_prefill_fill_device_call_sequence(monkeypatch):
    """``GLM53SpecEngine.prefill_fill`` with ttnn, the upload and the in-place refresh stubbed: the per-length inputs
    (the ``[1,1,1,t_pad]`` uint32 ids row sharded on its last dim, the ``[1,1,t_pad,32]`` bf16 mask row sharded on
    dim 2) are ALLOCATED once per length (``_fill_buffers``, the warm-up) and REFRESHED in place per chunk; the hidden
    is ``head.norm(head.collapse(streams))``, a chunk with rows past seq_len expands the mask with the ones row and
    zeroes the hidden through ``where`` (no mask refresh when the chunk fills every row), ``mtp.prefill_cache`` gets
    ``user`` / ``seq_len`` / ``start``, every transient is deallocated and the persistent inputs never are; once the
    mesh is frozen (a trace exists) an unwarmed length is refused instead of allocated."""
    from models.demos.glm53_flash.tt import generator_vllm as gv

    events = []
    freed = []

    class _Ttnn:
        uint32, bfloat16, float32 = "u32", "bf16", "f32"
        ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = "RM", "TILE", "DRAM"

        @staticmethod
        def deallocate(t):
            freed.append(t.tag)

        @staticmethod
        def matmul(a, b, dtype=None, compute_kernel_config=None):
            events.append(("matmul", a.tag, b.tag, dtype, compute_kernel_config))
            return _Rec("full", (1, 1, a.shape[2], b.shape[3]), dtype)

        @staticmethod
        def where(cond, a, b):
            events.append(("where", cond.tag, a.tag, b))
            return _Rec("clean", a.shape, a.dtype)

    def upload(mesh, t, dtype, layout, rows_dim=None):
        assert not mesh.allocations_frozen[0], "a device allocation under live traces"
        events.append(("upload", tuple(t.shape), str(t.dtype), dtype, layout, rows_dim))
        return _Rec("ids" if layout == "RM" else "mask", t.shape, dtype, t.clone())

    def refresh(mesh, t, dst, rows_dim=None):
        assert tuple(t.shape) == dst.shape, (tuple(t.shape), dst.shape)
        events.append(("refresh", dst.tag, tuple(t.shape), str(t.dtype), rows_dim))
        dst.extra = t.clone()

    monkeypatch.setattr(gv, "_ttnn", lambda: _Ttnn)
    monkeypatch.setattr(gv, "_spec_upload", upload)
    monkeypatch.setattr(gv, "_spec_refresh", refresh)

    class _Head:
        def collapse(self, s):
            events.append(("collapse", s.tag, s.shape))
            return _Rec("h", (1, 1, s.shape[2], 4096), "f32")

        def norm(self, h):
            events.append(("norm", h.tag))
            return _Rec("hidden", h.shape, "bf16")

    class _Mtp:
        norm_compute = "hifi4-fp32"

        def prefill_cache(self, ids, hidden, *, user, seq_len, start):
            events.append(("prefill_cache", ids.tag, hidden.tag, user, seq_len, start))
            return _Rec("out", hidden.shape)

    mesh = SimpleNamespace(rows=2, allocations_frozen=[False])
    eng = gv.GLM53SpecEngine.__new__(gv.GLM53SpecEngine)
    eng.model = SimpleNamespace(mesh=mesh, head=_Head())
    eng.mtp = _Mtp()
    eng.pad_token_id = PAD
    eng.mask_ones = _Rec("ones", (1, 1, 32, 4096), "bf16")
    eng.fill_ids, eng.fill_mask, eng.warmed_fill_lengths = {}, {}, []
    # the traced form: a 32-token prompt, T=128 streams (64-row slabs); the first call at T=128 allocates the length's
    # two persistent inputs, then refreshes them
    eng.prefill_fill(torch.arange(100, 132), 999, _Rec("streams", (1, 1, 64, 16384), "f32"), 2, 0, 32)
    kinds = [e[0] for e in events]
    assert kinds == ["upload", "upload", "refresh", "collapse", "norm", "refresh", "matmul", "where", "prefill_cache"]
    assert events[0][1:] == ((1, 1, 1, 128), "torch.int32", "u32", "RM", 3)
    assert events[1][1:] == ((1, 1, 128, 32), "torch.bfloat16", "bf16", "TILE", 2)
    assert events[2] == ("refresh", "ids", (1, 1, 1, 128), "torch.int32", 3)
    assert events[5] == ("refresh", "mask", (1, 1, 128, 32), "torch.bfloat16", 2)
    assert events[6] == ("matmul", "mask", "ones", "bf16", "hifi4-fp32") and events[7] == (
        "where",
        "full",
        "hidden",
        0.0,
    )
    assert events[8] == ("prefill_cache", "ids", "clean", 2, 32, 0)
    assert sorted(freed) == sorted(["h", "full", "hidden", "out", "clean"])  # never the persistent ids / mask
    assert eng.fill_ids[128].extra.reshape(-1)[:33].tolist() == list(range(101, 132)) + [999, PAD]
    assert eng.fill_mask[128].extra[0, 0, :, 0].tolist() == [1.0] * 32 + [0.0] * 96
    # a second chunk at the same length: refreshes only, no allocation (the frozen mesh accepts it)
    events.clear()
    freed.clear()
    mesh.allocations_frozen[0] = True
    eng.prefill_fill(torch.arange(40), 7, _Rec("streams", (1, 1, 64, 16384), "f32"), 1, 0, 40)
    assert [e[0] for e in events] == ["refresh", "collapse", "norm", "refresh", "matmul", "where", "prefill_cache"]
    assert events[-1] == ("prefill_cache", "ids", "clean", 1, 40, 0)
    assert eng.fill_mask[128].extra[0, 0, :, 0].tolist() == [1.0] * 40 + [0.0] * 88
    # an unwarmed length under live traces is refused before any device op; unfrozen it would allocate
    events.clear()
    with expect_error(RuntimeError, "no persistent inputs"):
        eng.prefill_fill(torch.arange(200), 7, _Rec("streams", (1, 1, 128, 16384), "f32"), 0, 0, 200)
    assert events == []
    mesh.allocations_frozen[0] = False
    eng.prefill_fill(torch.arange(200), 7, _Rec("streams", (1, 1, 128, 16384), "f32"), 0, 0, 200)
    assert [e[0] for e in events][:2] == ["upload", "upload"] and set(eng.fill_ids) == {128, 256}
    # the eager form on one mesh row with every row real: no mask refresh, no where; the ids are not row sharded
    events.clear()
    freed.clear()
    eng.model = SimpleNamespace(mesh=SimpleNamespace(rows=1, allocations_frozen=[False]), head=_Head())
    eng.prefill_fill(torch.arange(256), 5, _Rec("streams", (1, 1, 256, 16384), "f32"), 0, 16384, 256)
    assert [e[0] for e in events] == ["refresh", "collapse", "norm", "prefill_cache"]
    assert events[0] == ("refresh", "ids", (1, 1, 1, 256), "torch.int32", None)
    assert events[3] == ("prefill_cache", "ids", "hidden", 0, 256, 16384)
    assert sorted(freed) == sorted(["h", "hidden", "out"])
    assert "ttnn" not in sys.modules


def test_no_adapter_allocation_after_the_captures(monkeypatch):
    """The allocation discipline under live traces: once the adapter captured its traces, neither the adapter nor
    its engine may allocate a device tensor (``_spec_upload``); every per-step / per-prompt value goes through an
    in-place refresh. The upload is made to raise after the captures and the whole served flow (chunked and traced
    prefills, ordinary steps, verifies, proposals, the one-column fallback) runs through; with
    ``GLM53_VLLM_SPEC_ALLOC_CHECK=1`` the adapter compares the allocator's footprint with its post-capture baseline
    after every speculative call and refuses a growth."""
    from models.demos.glm53_flash.tt import generator_vllm as gv

    monkeypatch.setenv(vs.ALLOC_CHECK_ENV, "1")
    a, m, eng = _spec_adapter(K=3)
    assert a._spec_alloc_checking and a._spec_alloc_baseline == eng.allocated

    def refuse(*args, **kwargs):
        raise AssertionError("a device allocation after the captures")

    monkeypatch.setattr(gv, "_spec_upload", refuse)
    # a chunked eager prefill (two device chunks) beyond the dense range: plain steps only, no allocation
    runner = SpecRunnerSim(a, 3, wire=4)
    L = 16384 + 100
    ids = (torch.arange(L) % 60).to(torch.int32).reshape(1, L)
    out, _ = a.prefill_forward(
        ids, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=greedy_params(1)
    )
    runner.add(0, int(out[0, 0]), L)
    got = runner.run(4)
    assert len(got[0]) == 4 and all(s.kind == "plain" for s in runner.log)
    # two traced prefills inside the dense range: verifies and proposals, still no allocation
    runner = SpecRunnerSim(a, 3, wire=4)
    first, p = _prefill(a, 1, 300)
    runner.add(1, first, p)
    first2, p2 = _prefill(a, 2, 100)
    runner.add(2, first2, p2)
    got = runner.run(12)
    assert got[1] == _plain_chain(first, p, 12) and got[2] == _plain_chain(first2, p2, 12)
    assert any(s.kind == "verify" for s in runner.log)
    # the footprint check: a persistent allocation (the fake's footprint grows) is refused at the next spec call
    first3, p3 = _prefill(a, 3, 64)
    runner = SpecRunnerSim(a, 3, wire=4)
    runner.add(3, first3, p3)
    runner.step()
    eng.allocated += 4096
    with expect_error(RuntimeError, "footprint changed"):
        runner.step()


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
