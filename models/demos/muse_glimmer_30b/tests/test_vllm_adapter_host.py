# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only test of the vLLM contract adapter: drives the plugin's call sequences against a fake engine."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.muse_glimmer_30b.tt.generator_vllm import MuseGlimmerDFlash2ForCausalLM

pytest.importorskip("vllm_tt_plugin")

V = 11
K = 15
ROWS = 16


class FakeEngine:
    """Records calls; p / anchor / commit behave like DFlash2Generator, verify and propose are deterministic."""

    rows = ROWS

    def __init__(self, max_seq_len=4096):
        self.model = SimpleNamespace(args=SimpleNamespace(max_seq_len=max_seq_len, vocab_size=V))
        self.p = 0
        self.anchor = None
        self.pending = None
        self.calls = []
        self.traces_released = 0
        self.n_propose = 0

    def start(self, prompt_ids, sampling=None, seed=0, return_logits=False):
        assert return_logits and sampling is None
        self.calls.append(("start", list(prompt_ids)))
        self.p = len(prompt_ids)
        self.anchor = None
        self.pending = None
        return torch.arange(V, dtype=torch.float32)

    def verify(self, tokens, mode):
        tokens = list(tokens)
        assert 1 <= len(tokens) <= ROWS
        self.calls.append(("verify", tokens, mode, self.p))
        self.anchor = tokens[0]
        if mode == "argmax":
            return [100 + j for j in range(ROWS)]
        assert mode == "logits"
        return torch.arange(ROWS, dtype=torch.float32).reshape(ROWS, 1).repeat(1, V)

    def commit(self, count, last_token):
        assert 1 <= count <= ROWS
        self.calls.append(("commit", count, last_token))
        self.pending = (self.p, count)
        self.p += count
        self.anchor = last_token

    def propose(self):
        assert self.anchor is not None
        self.n_propose += 1
        self.calls.append(("propose", self.p, self.anchor))
        return [200 + self.n_propose * 20 + j for j in range(ROWS - 1)], None, None

    def release_traces(self):
        self.traces_released += 1

    def commits(self):
        return [c[1:] for c in self.calls if c[0] == "commit"]


def make(max_seq_len=4096):
    eng = FakeEngine(max_seq_len)
    return MuseGlimmerDFlash2ForCausalLM._from_engine(eng, kv_handles=["kv"]), eng


def prefill(ad, n=20):
    tokens = torch.arange(1, 1 + n, dtype=torch.int64).reshape(1, n)
    return ad.prefill_forward(tokens=tokens, page_table=None, kv_cache=None, prompt_lens=[n], start_pos=[0])


def ordinary(ad, token, pos):
    return ad.decode_forward(
        torch.tensor([[token]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), page_table=None
    )


def propose(ad, counts, tokens, k=K):
    """committed block [1, 1+K] padded with -1 past the accepted count."""
    block = torch.full((1, 1 + K), -1, dtype=torch.int32)
    block[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    pos = torch.zeros(1, 1 + K, dtype=torch.int32)
    return ad.propose_draft_tokens(k, block, pos, torch.tensor([counts], dtype=torch.int32))


def verify(ad, anchor, drafts, pos, mode, n=None, width=1 + K):
    n = len(drafts) if n is None else n
    toks = torch.full((1, width), -1, dtype=torch.int32)
    toks[0, 0] = anchor
    toks[0, 1 : 1 + len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
    start = torch.full((1, width), -1, dtype=torch.int32)
    start[0, : 1 + len(drafts)] = torch.arange(pos, pos + 1 + len(drafts), dtype=torch.int32)
    return ad.decode_forward(
        toks,
        start,
        page_table=None,
        num_valid_drafts=torch.tensor([n], dtype=torch.int32),
        accepted_counts=torch.tensor([1], dtype=torch.int32),
        spec_mode=mode,
    )


def test_declarations(expect_error):
    caps = MuseGlimmerDFlash2ForCausalLM.model_capabilities
    assert caps["supports_spec_decode"] is True
    assert tuple(caps["spec_requirements"]) == ("device_propose",)
    assert tuple(caps["spec_hidden_handoff"]) == ("on_device",)
    assert caps["output_tokens_per_step"] == 1
    assert not caps["supports_sample_on_device"] and not caps["supports_async_decode"]
    assert not caps["supports_async_spec_decode"] and not caps["supports_prefix_caching"]
    assert not caps["supports_chunked_prefill"]
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    plan = MuseGlimmerDFlash2ForCausalLM.spec_plan(None, 1, 20)
    assert isinstance(plan, SpecPlan) and plan.effective_k == 15 and plan.supports_narrow_decode
    assert plan.accept_modes == ("argmax_ids", "logits")
    assert MuseGlimmerDFlash2ForCausalLM.spec_plan(None, 1, 7).effective_k == 7
    rej = MuseGlimmerDFlash2ForCausalLM.spec_plan(None, 2, 7)
    assert isinstance(rej, SpecReject) and "batch 1" in rej.reason
    cls = MuseGlimmerDFlash2ForCausalLM
    assert cls.get_max_tokens_all_users(max_model_len=8192, max_num_seqs=1) == 8192
    with expect_error(ValueError, "131056"):
        cls.get_max_tokens_all_users(max_model_len=131057, max_num_seqs=1)
    with expect_error(ValueError, "supports one sequence"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(1, 4)), 2, max_seq_len=8192)
    with expect_error(ValueError, "131056"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(1, 4)), 1, max_seq_len=131057)
    with expect_error(ValueError, "mesh"):
        cls.initialize_vllm_model(None, SimpleNamespace(shape=(2, 2)), 1, max_seq_len=8192)


def test_spec_sequence():
    ad, eng = make()
    out = prefill(ad, 20)
    assert out.shape == (1, 1, V) and out.dtype == torch.float32
    assert eng.p == 20 and eng.anchor is None and not ad._outstanding

    # ordinary decode of the first sampled token
    out = ordinary(ad, 7, 20)
    assert out.shape == (1, 1, V) and out.dtype == torch.float32 and torch.equal(out[0, 0], torch.zeros(V))
    assert eng.calls[-1] == ("verify", [7], "logits", 20) and ad._outstanding and eng.p == 20

    # propose after the ordinary step: count 1
    d = propose(ad, 1, [7])
    assert eng.commits() == [(1, 7)] and eng.p == 21 and eng.anchor == 7 and not ad._outstanding
    assert d.draft_token_ids.shape == (1, K) and d.draft_token_ids.dtype == torch.int32
    assert d.num_valid.dtype == torch.int32 and d.num_valid.tolist() == [K]
    drafts1 = d.draft_token_ids[0].tolist()
    assert drafts1 == [220 + j for j in range(K)]

    # verify argmax, all 15 drafts
    v = verify(ad, 7, drafts1, 21, "argmax_ids")
    assert v.spec_mode == "argmax_ids" and v.argmax_ids.shape == (1, 1 + K) and v.argmax_ids.dtype == torch.int32
    assert v.argmax_ids[0].tolist() == [100 + j for j in range(ROWS)] and v.hidden is None
    assert eng.calls[-1] == ("verify", [7] + drafts1, "argmax", 21) and ad._outstanding

    # propose: 7 committed (6 drafts + bonus)
    committed = drafts1[:6] + [555]
    d = propose(ad, 7, committed)
    assert eng.commits()[-1] == (7, 555) and eng.p == 28 and eng.anchor == 555
    drafts2 = d.draft_token_ids[0].tolist()
    assert d.num_valid.tolist() == [K]

    # verify logits
    v = verify(ad, 555, drafts2, 28, "logits")
    assert v.spec_mode == "logits" and v.logits.shape == (1, 1 + K, V) and v.logits.dtype == torch.float32
    assert v.logits[0, :, 0].tolist() == [float(j) for j in range(ROWS)]
    assert eng.calls[-1][2:] == ("logits", 28)

    # propose: all 16 committed
    committed = drafts2 + [666]
    d = propose(ad, 16, committed)
    assert eng.commits()[-1] == (16, 666) and eng.p == 44 and eng.anchor == 666
    drafts3 = d.draft_token_ids[0].tolist()

    # verify with 3 valid drafts (the scheduler or grammar cut the proposal), padded block
    v = verify(ad, 666, drafts3[:3], 44, "argmax_ids", n=3)
    assert v.argmax_ids.shape == (1, 1 + K)
    assert eng.calls[-1] == ("verify", [666] + drafts3[:3], "argmax", 44)

    # propose: 2 committed
    d = propose(ad, 2, [drafts3[0], 777])
    assert eng.commits()[-1] == (2, 777) and eng.p == 46 and eng.anchor == 777
    assert d.num_valid.tolist() == [K] and not ad._outstanding

    # a narrower launch (K=7): propose returns 7, verify block is 8 wide
    d = propose(ad, 1, [777], k=7)  # not outstanding: no commit
    assert eng.commits()[-1] == (2, 777) and d.draft_token_ids.shape == (1, 7) and d.num_valid.tolist() == [7]


def test_narrow_verify_width_and_mismatch_warning(capsys):
    ad, eng = make()
    prefill(ad, 10)
    ordinary(ad, 5, 10)
    d = propose(ad, 1, [5], k=7)
    drafts = d.draft_token_ids[0].tolist()
    assert d.draft_token_ids.shape == (1, 7)
    v = verify(ad, 5, drafts, 11, "argmax_ids", width=8)
    assert v.argmax_ids.shape == (1, 8) and v.argmax_ids[0].tolist() == [100 + j for j in range(8)]
    propose(ad, 1, [9], k=7)
    # drafts that differ from the proposal still verify (and log a warning)
    v = verify(ad, 9, [1, 2, 3], 12, "logits", width=8)
    assert v.logits.shape == (1, 8, V) and eng.calls[-1] == ("verify", [9, 1, 2, 3], "logits", 12)
    # zero valid drafts after a decline: a one-row verify
    propose(ad, 1, [4], k=7)
    v = verify(ad, 4, [], 13, "argmax_ids", width=8)
    assert eng.calls[-1] == ("verify", [4], "argmax", 13) and v.argmax_ids.shape == (1, 8)


def test_no_spec_server_commit_inferred_from_start_pos(expect_error):
    ad, eng = make()
    prefill(ad, 20)
    tokens = [31, 32, 33, 34, 35]
    for i, t in enumerate(tokens):
        out = ordinary(ad, t, 20 + i)
        assert out.shape == (1, 1, V)
        assert eng.calls[-1] == ("verify", [t], "logits", 20 + i)
    # commits happen one step late, each of count 1 with the new anchor
    assert eng.commits() == [(1, t) for t in tokens[1:]]
    assert eng.p == 24 and eng.anchor == 35 and ad._outstanding
    # a position that does not follow the engine is refused
    with expect_error(RuntimeError, "implies"):
        ordinary(ad, 99, 45)
    ad2, _ = make()
    prefill(ad2, 20)
    with expect_error(RuntimeError, "engine is at"):
        ordinary(ad2, 1, 25)


def test_decode_reconciles_multi_token_commit_without_propose():
    ad, eng = make()
    prefill(ad, 20)
    ordinary(ad, 7, 20)
    propose(ad, 1, [7])
    verify(ad, 7, [220 + j for j in range(3)], 21, "argmax_ids", n=3)
    # no propose in between (a declined proposal path): the next ordinary decode starts 3 tokens on
    ordinary(ad, 88, 24)
    assert eng.commits()[-1] == (3, 88) and eng.calls[-1] == ("verify", [88], "logits", 24)


def test_propose_declines_near_max_seq_len(expect_error):
    ad, eng = make(max_seq_len=256)
    prefill(ad, 200)
    ordinary(ad, 7, 200)
    d = propose(ad, 1, [7])
    assert d.num_valid.tolist() == [K] and eng.p == 201
    # 240 + 16 > 256: no room for a 16-row verify
    eng.p = 241
    eng.anchor = 3
    ad._outstanding = True
    n_propose = eng.n_propose
    d = propose(ad, 1, [3])
    assert d.num_valid.tolist() == [0] and d.draft_token_ids.shape == (1, K)
    assert eng.n_propose == n_propose and eng.p == 242 and not ad._outstanding
    # a prompt that leaves no room is refused
    with expect_error(ValueError, "does not fit"):
        prefill(ad, 250)


def test_propose_without_anchor_declines():
    ad, eng = make()
    prefill(ad, 20)
    d = propose(ad, 1, [7])
    assert d.num_valid.tolist() == [0] and eng.n_propose == 0


def test_release_request_resets(expect_error):
    ad, eng = make()
    prefill(ad, 20)
    ordinary(ad, 7, 20)
    propose(ad, 1, [7])
    verify(ad, 7, [220 + j for j in range(K)], 21, "argmax_ids")
    assert ad._outstanding
    ad.release_request(0)
    assert eng.p == 0 and eng.anchor is None and eng.pending is None
    assert not ad._outstanding and ad._last_drafts is None
    # the next request starts clean
    prefill(ad, 12)
    assert eng.p == 12
    ordinary(ad, 3, 12)
    assert eng.calls[-1] == ("verify", [3], "logits", 12)
    ad.release_persistent_capture()
    assert eng.traces_released == 1
    ad.note_state_slots_moved({0: 0})
    with expect_error(RuntimeError, "single-slot adapter"):
        ad.note_state_slots_moved({0: 1})
    assert ad.allocate_kv_cache_per_layer([]) == ["kv"]


def test_refusals(expect_error):
    ad, _ = make()
    with expect_error(ValueError, "one sequence"):
        ad.prefill_forward(tokens=torch.zeros(2, 4, dtype=torch.int64), prompt_lens=[4, 4])
    with expect_error(ValueError, "start_pos"):
        ad.prefill_forward(tokens=torch.zeros(1, 4, dtype=torch.int64), prompt_lens=[4], start_pos=[2])
    prefill(ad, 20)
    ordinary(ad, 7, 20)
    propose(ad, 1, [7])
    with expect_error(ValueError, "spec_mode"):
        verify(ad, 7, [1], 21, "fused_sample")
