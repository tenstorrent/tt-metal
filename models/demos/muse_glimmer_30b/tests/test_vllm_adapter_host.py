# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only test of the vLLM contract adapter: drives the plugin's call sequences against a fake engine."""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.muse_glimmer_30b.tt.generator_vllm import MuseGlimmerDFlash2ForCausalLM

pytest.importorskip("vllm_tt_plugin")

V = 11
K = 15
ROWS = 16

_K_BY = ((2, 15), (4, 7), (8, 3))


def _rows_for(n):
    return next(k + 1 for users, k in _K_BY if n <= users)


class FakeEngine:
    """Records calls; slots / p / anchor / commits behave like DFlash2BatchGenerator; verify and propose are
    deterministic (values encode the engine slot)."""

    rows = ROWS

    def __init__(self, max_seq_len=4096, max_users=8):
        self.model = SimpleNamespace(args=SimpleNamespace(max_seq_len=max_seq_len, vocab_size=V))
        self.max_users = max_users
        self.slots = {}
        self.calls = []
        self.traces_released = 0
        self.n_propose = 0
        self.q_draft = 0.25  # drafter probability of each sampled draft (the 3 other candidates share the rest)

    def start_slot(self, slot, prompt_ids, return_logits=False, **kwargs):
        assert return_logits and not kwargs
        self.calls.append(("start", slot, list(prompt_ids)))
        self.slots[slot] = SimpleNamespace(p=len(prompt_ids), anchor=None, sp=None, gen=None)
        return torch.arange(V, dtype=torch.float32) + 100 * slot

    def verify_slots(self, rows, mode):
        R = _rows_for(len(rows))
        out = {}
        for u, toks in rows.items():
            toks = list(toks)
            assert 1 <= len(toks) <= R, (u, toks, R)
            s = self.slots[u]
            self.calls.append(("verify", u, toks, mode, s.p))
            s.anchor = toks[0]
            if mode == "argmax":
                out[u] = [1000 * (u + 1) + j for j in range(R)]
            elif mode == "topk":
                ids = 10000 * (u + 1) + 100 * torch.arange(R).reshape(R, 1) + torch.arange(32).reshape(1, 32)
                out[u] = (torch.arange(32, 0, -1, dtype=torch.float32).reshape(1, 32).repeat(R, 1), ids)
            else:
                assert mode == "logits"
                out[u] = torch.arange(R, dtype=torch.float32).reshape(R, 1).repeat(1, V) + 100 * u
        return out

    def commit_slot(self, slot, count, last_token):
        assert 1 <= count <= ROWS
        self.calls.append(("commit", slot, count, last_token))
        s = self.slots[slot]
        s.p += count
        s.anchor = last_token

    def propose_slots(self, slots):
        R = _rows_for(len(slots))
        self.n_propose += 1
        self.calls.append(("propose", tuple(slots)))
        out = {}
        for u in slots:
            s = self.slots[u]
            assert s.anchor is not None
            d = [self.draft(u, j) for j in range(R - 1)]
            if s.sp is None:
                out[u] = (d, None, None)
            else:  # drafter candidates: the draft and 3 others, the draft with probability q_draft
                rest = (1 - self.q_draft) / 3
                out[u] = (
                    d,
                    torch.tensor([[self.q_draft, rest, rest, rest]] * len(d)),
                    torch.tensor([[t, t + 1, t + 2, t + 3] for t in d]),
                )
        return out

    def draft(self, u, j):
        return 200 + 20 * self.n_propose + 1000 * u + j

    def release_slot(self, slot):
        self.calls.append(("release", slot))
        self.slots.pop(slot, None)

    def release_traces(self):
        self.traces_released += 1

    def of(self, kind):
        return [c[1:] for c in self.calls if c[0] == kind]


def make(max_seq_len=4096, max_users=8):
    eng = FakeEngine(max_seq_len, max_users)
    return MuseGlimmerDFlash2ForCausalLM._from_engine(eng, kv_handles=["kv"]), eng


def prefill(ad, lens, slots):
    """One prefill call with an [N, max_len] token block."""
    n = len(lens)
    tokens = torch.zeros(n, max(lens), dtype=torch.int64)
    for r, ln in enumerate(lens):
        tokens[r, :ln] = torch.arange(1, 1 + ln, dtype=torch.int64) + 1000 * r
    return ad.prefill_forward(
        tokens=tokens, page_table=None, kv_cache=None, prompt_lens=list(lens), start_pos=[0] * n, empty_slots=slots
    )


def ordinary(ad, tokens, positions, rows=None, slot_remap=None, **extra):
    """Ordinary decode; positions -1 mark padding rows."""
    b = len(tokens) if rows is None else rows
    t = torch.zeros(b, 1, dtype=torch.int32)
    p = torch.full((b,), -1, dtype=torch.int32)
    t[: len(tokens), 0] = torch.tensor(tokens, dtype=torch.int32)
    p[: len(positions)] = torch.tensor(positions, dtype=torch.int32)
    kw = {} if slot_remap is None else {"slot_remap": torch.tensor(slot_remap, dtype=torch.int32)}
    return ad.decode_forward(t, p, page_table=None, **kw, **extra)


def propose(ad, counts, blocks, positions, k=K):
    """committed blocks [B, 1+K] padded with -1 past the accepted count; positions[i] = row's first position or -1."""
    b = len(blocks)
    block = torch.full((b, 1 + K), -1, dtype=torch.int32)
    pos = torch.full((b, 1 + K), -1, dtype=torch.int32)
    for i, toks in enumerate(blocks):
        block[i, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        if positions[i] >= 0:
            pos[i] = torch.arange(positions[i], positions[i] + 1 + K, dtype=torch.int32)
    return ad.propose_draft_tokens(k, block, pos, torch.tensor(counts, dtype=torch.int32))


def verify(ad, anchors, drafts, positions, mode, n=None, width=1 + K, slot_remap=None, **extra):
    """Verify block per row: anchors[i], drafts[i] (list), positions[i] (-1 = padding row)."""
    b = len(anchors)
    n = [len(d) for d in drafts] if n is None else n
    toks = torch.full((b, width), -1, dtype=torch.int32)
    start = torch.full((b, width), -1, dtype=torch.int32)
    for i in range(b):
        toks[i, 0] = anchors[i]
        toks[i, 1 : 1 + len(drafts[i])] = torch.tensor(drafts[i], dtype=torch.int32)
        if positions[i] >= 0:
            start[i, : 1 + len(drafts[i])] = torch.arange(positions[i], positions[i] + 1 + len(drafts[i]))
    kw = {} if slot_remap is None else {"slot_remap": torch.tensor(slot_remap, dtype=torch.int32)}
    return ad.decode_forward(
        toks,
        start,
        page_table=None,
        num_valid_drafts=torch.tensor(n, dtype=torch.int32),
        accepted_counts=torch.ones(b, dtype=torch.int32),
        spec_mode=mode,
        **kw,
        **extra,
    )


def test_declarations(expect_error):
    caps = MuseGlimmerDFlash2ForCausalLM.model_capabilities
    assert caps["supports_spec_decode"] is True
    assert tuple(caps["spec_requirements"]) == ("device_propose",)
    assert tuple(caps["spec_hidden_handoff"]) == ("on_device",)
    assert caps["output_tokens_per_step"] == 1
    assert not caps["supports_async_decode"]
    assert not caps["supports_async_spec_decode"] and not caps["supports_prefix_caching"]
    assert not caps["supports_chunked_prefill"]
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    cls = MuseGlimmerDFlash2ForCausalLM
    plan = cls.spec_plan(None, 1, 20)
    assert isinstance(plan, SpecPlan) and plan.effective_k == 15 and plan.supports_narrow_decode
    assert plan.accept_modes == ("argmax_ids", "logits")
    assert plan.k_by_rows == ((2, 15), (4, 7), (8, 3))
    plan = cls.spec_plan(None, 8, 7)
    assert isinstance(plan, SpecPlan) and plan.effective_k == 7
    assert plan.k_by_rows == ((2, 7), (4, 7), (8, 3))
    rej = cls.spec_plan(None, 9, 15)
    assert isinstance(rej, SpecReject) and "at most 8" in rej.reason
    assert cls.get_max_tokens_all_users(max_model_len=8192, max_num_seqs=4) == 32768
    with expect_error(ValueError, "131056"):
        cls.get_max_tokens_all_users(max_model_len=131057, max_num_seqs=1)
    with expect_error(ValueError, "max_num_seqs"):
        cls.get_max_tokens_all_users(max_model_len=8192, max_num_seqs=9)
    with expect_error(ValueError, "at most 8"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(1, 4)), 9, max_seq_len=8192)
    with expect_error(ValueError, "131056"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(1, 4)), 2, max_seq_len=131057)
    with expect_error(ValueError, "mesh"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(2, 2)), 2, max_seq_len=8192)


def test_spec_sequence():
    ad, eng = make(max_users=1)
    out = prefill(ad, [20], [0])
    assert out.shape == (1, 1, V) and out.dtype == torch.float32
    assert eng.slots[0].p == 20 and eng.slots[0].anchor is None and not ad._outstanding

    # ordinary decode of the first sampled token
    out = ordinary(ad, [7], [20])
    assert out.shape == (1, 1, V) and out.dtype == torch.float32 and torch.equal(out[0, 0], torch.zeros(V))
    assert eng.calls[-1] == ("verify", 0, [7], "logits", 20) and ad._outstanding == {0} and eng.slots[0].p == 20

    # propose after the ordinary step: count 1
    d = propose(ad, [1], [[7]], [21])
    assert eng.of("commit") == [(0, 1, 7)] and eng.slots[0].p == 21 and eng.slots[0].anchor == 7
    assert not ad._outstanding
    assert d.draft_token_ids.shape == (1, K) and d.draft_token_ids.dtype == torch.int32
    assert d.num_valid.dtype == torch.int32 and d.num_valid.tolist() == [K]
    drafts1 = d.draft_token_ids[0].tolist()
    assert drafts1 == [220 + j for j in range(K)]

    # verify argmax, all 15 drafts
    v = verify(ad, [7], [drafts1], [21], "argmax_ids")
    assert v.spec_mode == "argmax_ids" and v.argmax_ids.shape == (1, 1 + K) and v.argmax_ids.dtype == torch.int32
    assert v.argmax_ids[0].tolist() == [1000 + j for j in range(ROWS)] and v.hidden is None
    assert eng.calls[-1] == ("verify", 0, [7] + drafts1, "argmax", 21) and ad._outstanding == {0}

    # propose: 7 committed (6 drafts + bonus)
    committed = drafts1[:6] + [555]
    d = propose(ad, [7], [committed], [21])
    assert eng.of("commit")[-1] == (0, 7, 555) and eng.slots[0].p == 28 and eng.slots[0].anchor == 555
    drafts2 = d.draft_token_ids[0].tolist()
    assert d.num_valid.tolist() == [K]

    # verify logits
    v = verify(ad, [555], [drafts2], [28], "logits")
    assert v.spec_mode == "logits" and v.logits.shape == (1, 1 + K, V) and v.logits.dtype == torch.float32
    assert v.logits[0, :, 0].tolist() == [float(j) for j in range(ROWS)]
    assert eng.calls[-1] == ("verify", 0, [555] + drafts2, "logits", 28)

    # propose: all 16 committed
    committed = drafts2 + [666]
    d = propose(ad, [16], [committed], [28])
    assert eng.of("commit")[-1] == (0, 16, 666) and eng.slots[0].p == 44 and eng.slots[0].anchor == 666
    drafts3 = d.draft_token_ids[0].tolist()

    # verify with 3 valid drafts (the scheduler or grammar cut the proposal), padded block
    v = verify(ad, [666], [drafts3[:3]], [44], "argmax_ids", n=[3])
    assert v.argmax_ids.shape == (1, 1 + K)
    assert eng.calls[-1] == ("verify", 0, [666] + drafts3[:3], "argmax", 44)

    # propose: 2 committed
    d = propose(ad, [2], [[drafts3[0], 777]], [44])
    assert eng.of("commit")[-1] == (0, 2, 777) and eng.slots[0].p == 46 and eng.slots[0].anchor == 777
    assert d.num_valid.tolist() == [K] and not ad._outstanding

    # a narrower launch (K=7): propose returns 7, verify block is 8 wide
    d = propose(ad, [1], [[777]], [46], k=7)  # not outstanding: no commit
    assert eng.of("commit")[-1] == (0, 2, 777) and d.draft_token_ids.shape == (1, 7) and d.num_valid.tolist() == [7]


def test_narrow_verify_width_and_mismatch_warning():
    ad, eng = make(max_users=1)
    prefill(ad, [10], [0])
    ordinary(ad, [5], [10])
    d = propose(ad, [1], [[5]], [11], k=7)
    drafts = d.draft_token_ids[0].tolist()
    assert d.draft_token_ids.shape == (1, 7)
    v = verify(ad, [5], [drafts], [11], "argmax_ids", width=8)
    assert v.argmax_ids.shape == (1, 8) and v.argmax_ids[0].tolist() == [1000 + j for j in range(8)]
    propose(ad, [1], [[9]], [12], k=7)
    # drafts that differ from the proposal still verify (and log a warning)
    messages = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        v = verify(ad, [9], [[1, 2, 3]], [12], "logits", width=8)
    finally:
        logger.remove(sink)
    assert v.logits.shape == (1, 8, V) and eng.calls[-1] == ("verify", 0, [9, 1, 2, 3], "logits", 12)
    assert any("differ from the proposed" in m for m in messages)
    # zero valid drafts after a decline: a one-row verify
    propose(ad, [1], [[4]], [13], k=7)
    v = verify(ad, [4], [[]], [13], "argmax_ids", width=8)
    assert eng.calls[-1] == ("verify", 0, [4], "argmax", 13) and v.argmax_ids.shape == (1, 8)


def test_no_spec_server_commit_inferred_from_start_pos(expect_error):
    ad, eng = make(max_users=1)
    prefill(ad, [20], [0])
    tokens = [31, 32, 33, 34, 35]
    for i, t in enumerate(tokens):
        out = ordinary(ad, [t], [20 + i])
        assert out.shape == (1, 1, V)
        assert eng.calls[-1] == ("verify", 0, [t], "logits", 20 + i)
    # commits happen one step late, each of count 1 with the new anchor
    assert eng.of("commit") == [(0, 1, t) for t in tokens[1:]]
    assert eng.slots[0].p == 24 and eng.slots[0].anchor == 35 and ad._outstanding == {0}
    # a position that does not follow the engine is refused
    with expect_error(RuntimeError, "cannot commit"):
        ordinary(ad, [99], [45])
    ad2, _ = make(max_users=1)
    prefill(ad2, [20], [0])
    with expect_error(RuntimeError, "is at"):
        ordinary(ad2, [1], [25])


def test_decode_reconciles_multi_token_commit_without_propose():
    ad, eng = make(max_users=1)
    prefill(ad, [20], [0])
    ordinary(ad, [7], [20])
    propose(ad, [1], [[7]], [21])
    verify(ad, [7], [[220 + j for j in range(3)]], [21], "argmax_ids", n=[3])
    # no propose in between (a declined proposal path): the next ordinary decode starts 3 tokens on
    ordinary(ad, [88], [24])
    assert eng.of("commit")[-1] == (0, 3, 88) and eng.calls[-1] == ("verify", 0, [88], "logits", 24)


def test_propose_declines_near_max_seq_len(expect_error):
    ad, eng = make(max_seq_len=256, max_users=1)
    prefill(ad, [200], [0])
    ordinary(ad, [7], [200])
    d = propose(ad, [1], [[7]], [201])
    assert d.num_valid.tolist() == [K] and eng.slots[0].p == 201
    # 240 + 16 > 256: no room for a 16-row verify
    eng.slots[0].p = 241
    eng.slots[0].anchor = 3
    ad._outstanding = {0}
    n_propose = eng.n_propose
    d = propose(ad, [1], [[3]], [241])
    assert d.num_valid.tolist() == [0] and d.draft_token_ids.shape == (1, K)
    assert eng.n_propose == n_propose and eng.slots[0].p == 242 and not ad._outstanding
    # a prompt that leaves no room is refused
    with expect_error(ValueError, "does not fit"):
        prefill(ad, [250], [0])


def test_propose_without_anchor_declines():
    ad, eng = make(max_users=1)
    prefill(ad, [20], [0])
    d = propose(ad, [1], [[7]], [20])
    assert d.num_valid.tolist() == [0] and eng.n_propose == 0


def test_release_request_resets(expect_error):
    ad, eng = make(max_users=1)
    prefill(ad, [20], [0])
    ordinary(ad, [7], [20])
    propose(ad, [1], [[7]], [21])
    verify(ad, [7], [[220 + j for j in range(K)]], [21], "argmax_ids")
    assert ad._outstanding == {0}
    ad.release_request(0)
    assert eng.of("release") == [(0,)] and 0 not in eng.slots
    assert not ad._outstanding and not ad._last_drafts
    # the next request starts clean
    prefill(ad, [12], [0])
    assert eng.slots[0].p == 12
    ordinary(ad, [3], [12])
    assert eng.calls[-1] == ("verify", 0, [3], "logits", 12)
    ad.release_persistent_capture()
    assert eng.traces_released == 1
    ad.note_state_slots_moved({0: 0})
    assert ad.allocate_kv_cache_per_layer([]) == ["kv"]


def test_refusals(expect_error):
    ad, _ = make(max_users=1)
    with expect_error(ValueError, "empty_slots"):
        ad.prefill_forward(tokens=torch.zeros(2, 4, dtype=torch.int64), prompt_lens=[4, 4])
    with expect_error(ValueError, "start_pos"):
        ad.prefill_forward(tokens=torch.zeros(1, 4, dtype=torch.int64), prompt_lens=[4], start_pos=[2])
    prefill(ad, [20], [0])
    ordinary(ad, [7], [20])
    propose(ad, [1], [[7]], [21])
    with expect_error(ValueError, "spec_mode"):
        verify(ad, [7], [[1]], [21], "fused_sample")


def test_two_rows_prefill_decode_verify_propose():
    ad, eng = make()
    out = prefill(ad, [20, 30], [0, 1])
    assert out.shape == (2, 1, V) and out.dtype == torch.float32
    for r in range(2):
        assert torch.equal(out[r, 0], torch.arange(V, dtype=torch.float32) + 100 * r)

    out = ordinary(ad, [5, 6], [20, 30])
    assert eng.of("verify") == [(0, [5], "logits", 20), (1, [6], "logits", 30)]
    assert out.shape == (2, 1, V)
    assert torch.equal(out[0, 0], torch.zeros(V)) and torch.equal(out[1, 0], torch.full((V,), 100.0))

    d = propose(ad, [1, 1], [[5], [6]], [21, 31])
    assert eng.of("commit") == [(0, 1, 5), (1, 1, 6)]
    assert eng.of("propose") == [((0, 1),)]
    assert d.num_valid.tolist() == [15, 15] and d.draft_token_ids.shape == (2, K)
    d0, d1 = d.draft_token_ids[0].tolist(), d.draft_token_ids[1].tolist()
    assert d0 == [220 + j for j in range(15)] and d1 == [1220 + j for j in range(15)]

    v = verify(ad, [5, 6], [d0, d1[:7]], [21, 31], "argmax_ids", n=[15, 7])
    assert [len(c[2]) for c in eng.calls if c[0] == "verify"][-2:] == [16, 8]
    assert v.argmax_ids[0].tolist() == [1000 + j for j in range(16)]
    assert v.argmax_ids[1].tolist() == [2000 + j for j in range(16)]

    block0 = d0[:2] + [901]
    block1 = d1[:7] + [902]
    propose(ad, [3, 8], [block0, block1], [21, 31])
    assert eng.of("commit")[-2:] == [(0, 3, block0[2]), (1, 8, block1[7])]


def test_padding_rows_and_eight_rows():
    ad, eng = make()
    prefill(ad, [20, 30], [0, 1])
    out = ordinary(ad, [5, 6, 0, 0], [20, 30, -1, -1])
    assert eng.of("verify") == [(0, [5], "logits", 20), (1, [6], "logits", 30)]
    assert out.shape == (4, 1, V) and torch.count_nonzero(out[2:]) == 0

    ad, eng = make()
    lens = [20 + i for i in range(8)]
    prefill(ad, lens, list(range(8)))
    toks = [1 + i for i in range(8)]
    ordinary(ad, toks, lens)
    d = propose(ad, [1] * 8, [[t] for t in toks], [n + 1 for n in lens])
    assert eng.of("propose") == [(tuple(range(8)),)]
    assert d.num_valid.tolist() == [3] * 8
    drafts = [d.draft_token_ids[i, :3].tolist() for i in range(8)]
    v = verify(ad, toks, drafts, [n + 1 for n in lens], "logits", n=[3] * 8)
    assert v.logits.shape == (8, 16, V)
    for i in range(8):
        want = torch.arange(4, dtype=torch.float32).reshape(4, 1).repeat(1, V) + 100 * i
        assert torch.equal(v.logits[i, :4], want)


def test_slot_remap_is_bookkeeping(expect_error):
    ad, eng = make(max_users=2)
    prefill(ad, [20, 30], [0, 1])
    ordinary(ad, [5, 6], [20, 30])
    ad.release_request(0)
    assert eng.of("release") == [(0,)]

    # B moves from plugin slot 1 to row 0; nothing moves on the device
    out = ordinary(ad, [9, 0], [31, -1], slot_remap=[1, 0])
    assert out.shape == (2, 1, V)
    assert eng.of("commit")[-1] == (1, 1, 9) and eng.calls[-1] == ("verify", 1, [9], "logits", 31)
    assert ad.phys == [1, 0]

    propose(ad, [1, 1], [[10], [-1]], [32, -1])
    assert eng.of("commit")[-1] == (1, 1, 10) and eng.of("propose") == [((1,),)]

    # a new request in plugin slot 1 lands in engine slot 0
    prefill(ad, [12], [1])
    assert eng.calls[-1][:2] == ("start", 0)

    # refused decodes leave phys alone
    with expect_error(RuntimeError, "permutation"):
        ordinary(ad, [1, 0], [12, -1], slot_remap=[0, 0])
    assert ad.phys == [1, 0]
    with expect_error(RuntimeError, "is at"):
        ordinary(ad, [1, 0], [999, -1], slot_remap=[1, 0])
    assert ad.phys == [1, 0]

    ad.release_request(0)
    assert eng.of("release")[-1] == (1,)


def _sampling(temperature, top_k, seed, top_p=None):
    from vllm_tt_plugin.model_input import TTSamplingParams

    n = len(temperature)
    return TTSamplingParams(
        temperature=list(temperature),
        top_k=list(top_k),
        top_p=[1.0] * n if top_p is None else list(top_p),
        seed=list(seed),
    )


def _sampled_verify_ids(seed, n=(3, 2), width=4):
    """Prefill two rows, then one argmax_ids verify with row 0 sampled (top_k 32) and row 1 greedy."""
    ad, eng = make()
    prefill(ad, [20, 30], [0, 1])
    sp = _sampling([1.0, 0.0], [32, 1], [seed, None])
    drafts = [[5, 6, 7][: n[0]], [8, 9, 10][: n[1]]]
    v = verify(ad, [1, 2], drafts, [20, 30], "argmax_ids", n=list(n), width=width, sampling_params=sp)
    return v.argmax_ids, eng


def test_declares_device_sampled_ids():
    caps = MuseGlimmerDFlash2ForCausalLM.model_capabilities
    assert caps["supports_sample_on_device"] is True and caps["max_device_top_k"] == 32
    assert caps["spec_ids_are_sampled"] is True and caps["supports_device_penalties"] is False


def test_device_sampled_decode_returns_ids():
    ad, eng = make()
    prefill(ad, [20, 30], [0, 1])
    sp = _sampling([1.0, 0.0], [8, 1], [7, None])
    out = ordinary(ad, [5, 6], [20, 30], sampling_params=sp)
    assert out.shape == (2,) and out.dtype == torch.int32
    assert int(out[0]) in {10000 + c for c in range(8)}
    assert int(out[1]) == 20000
    assert [c[3] for c in eng.calls if c[0] == "verify"] == ["topk", "topk"]


def test_device_sampled_verify_draws_every_column():
    ids, eng = _sampled_verify_ids(11)
    assert ids.shape == (2, 4) and ids.dtype == torch.int32
    for j in range(4):  # row 0: num_valid 3, columns 0..3 are draws from row j of slot 0
        assert int(ids[0, j]) in {10000 + 100 * j + c for c in range(32)}
    for j in range(3):  # row 1: num_valid 2, greedy columns 0..2 are the top candidate of row j of slot 1
        assert int(ids[1, j]) == 20000 + 100 * j
    assert int(ids[1, 3]) == 0
    assert [c[3] for c in eng.calls if c[0] == "verify"] == ["topk", "topk"]


def test_device_sampled_seed_reproducible():
    a, _ = _sampled_verify_ids(11)
    b, _ = _sampled_verify_ids(11)
    assert torch.equal(a, b)
    others = [_sampled_verify_ids(s)[0][0] for s in (12, 13, 14, 15)]
    assert any(not torch.equal(a[0], o) for o in others)


def _rejection_round(draft, q_draft, seed=3):
    """One sampled request: a sampled ordinary decode (sets the slot's sampling), a propose (drafts drawn from q), then
    an argmax_ids verify of the 15 drafts. Returns (drafts, verify ids [16])."""
    ad, eng = make(max_users=1)
    prefill(ad, [20], [0])
    sp = _sampling([1.0], [32], [seed])
    first = int(ordinary(ad, [7], [20], sampling_params=sp)[0])
    assert eng.slots[0].sp == (1.0, 1.0, 32) and eng.slots[0].gen is ad._rng[0]
    eng.draft, eng.q_draft = draft, q_draft
    d = propose(ad, [1], [[first]], [21])
    drafts = d.draft_token_ids[0].tolist()
    v = verify(ad, [first], [drafts], [21], "argmax_ids", sampling_params=sp)
    assert eng.calls[-1][3] == "topk"
    return drafts, v.argmax_ids[0].tolist()


def test_device_sampled_verify_rejection_sampling():
    # Each draft is the target's top candidate (p ~ 0.63) with q = 0.25: p >= q, so every draft is accepted and the
    # bonus is a draw from the last row.
    drafts, ids = _rejection_round(lambda u, j: 10000 + 100 * j, 0.25)
    assert ids[:K] == drafts and ids[K] in {10000 + 100 * K + c for c in range(32)}
    # Each draft is outside the target's candidates (p = 0) with q = 1: the first draft is rejected and the residual
    # draw from row 0 commits in its place.
    drafts, ids = _rejection_round(lambda u, j: 99999, 1.0)
    assert ids[0] in {10000 + c for c in range(32)} and ids[1:] == [0] * K


def test_greedy_verify_with_sampling_params_uses_argmax():
    ad, eng = make()
    prefill(ad, [20, 30], [0, 1])
    sp = _sampling([0.0, 1e-6], [1, 32], [None, 3])
    v = verify(ad, [1, 2], [[5, 6], [8]], [20, 30], "argmax_ids", n=[2, 1], width=3, sampling_params=sp)
    assert [c[3] for c in eng.calls if c[0] == "verify"] == ["argmax", "argmax"]
    ad2, _ = make()
    prefill(ad2, [20, 30], [0, 1])
    ref = verify(ad2, [1, 2], [[5, 6], [8]], [20, 30], "argmax_ids", n=[2, 1], width=3)
    assert torch.equal(v.argmax_ids, ref.argmax_ids)


def test_full_galaxy_mesh_is_carved_to_one_tp8_submesh(monkeypatch, expect_error):
    """An (8, 4) system mesh goes through tp_submesh(mesh, 8); a (1, tp) mesh is used as is."""
    from models.demos.muse_glimmer_30b.tt import mesh_utils

    calls = []

    def fake_tp_submesh(mesh, tp):
        calls.append((tuple(mesh.shape), tp))
        return SimpleNamespace(shape=(2, 2))  # not a 1xTP mesh: stops initialize right after the carve

    monkeypatch.setattr(mesh_utils, "tp_submesh", fake_tp_submesh)
    cls = MuseGlimmerDFlash2ForCausalLM
    with expect_error(ValueError, "mesh"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(8, 4)), 2, max_seq_len=8192)
    assert calls == [((8, 4), 8)]
    with expect_error(ValueError, "mesh"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(4, 8)), 2, max_seq_len=8192)
    assert len(calls) == 1
