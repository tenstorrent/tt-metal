# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests of the vLLM interface (tt/vllm_state.py, tt/generator_vllm.py): batch padding, slot map / plugin ``slot_remap`` handling, prefill / decode input
building and output scatter, sampling fallbacks, block accounting, bundle metadata, chat template. A fake model stands in for ``tt/dsv41_model.Model``.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_vllm_interface.py
"""

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt import vllm_state as VS
from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture
def expect_error():
    """Local stand-in of the repo conftest fixture so that this hardware-free file also runs with ``--noconftest`` (the repo conftest brings up a device mesh)."""
    import contextlib
    import re

    @contextlib.contextmanager
    def expect_error_(error, message):
        try:
            yield
        except error as e:
            assert re.search(message, str(e)), f"{e!r} does not match {message!r}"
        else:
            raise AssertionError(f"{error} not raised")

    return expect_error_


# ---- vllm_state ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,expect", [(1, 4), (4, 4), (5, 8), (16, 16), (126, 128), (128, 128)])
def test_padded_batch(n, expect):
    assert VS.padded_batch(n) == expect


@pytest.mark.parametrize("n", [0, 129, 256])
def test_padded_batch_rejects(n, expect_error):
    with expect_error(ValueError, ".*"):
        VS.padded_batch(n)


def test_s_pad_bucket(expect_error):
    assert VS.s_pad_bucket(100, 128) == 1024
    assert VS.s_pad_bucket(1025, 512) == 2048
    assert VS.s_pad_bucket(33000, 64) == 65536
    assert VS.s_pad_bucket(1000, None) == 1024
    assert VS.s_pad_bucket(1000, 256, "exact") == 1024
    assert VS.s_pad_bucket(1025, 256, "exact") == 1280
    assert VS.s_pad_bucket(10, 128, "4096") == 4096
    with expect_error(ValueError, ".*"):
        VS.s_pad_bucket(5000, 128, "4096")
    with expect_error(ValueError, ".*"):
        VS.s_pad_bucket(10, 128, "nonsense")


def test_slot_table_remap_follows_requests():
    st = VS.SlotTable(8, 8)
    assert st.claim([0, 1, 2]) == [0, 1, 2]
    # plugin moves the request of old slot 2 to row 0, old 0 to row 1, old 1 to row 2 (a permutation of all 8 slots)
    st.apply_remap([2, 0, 1, 3, 4, 5, 6, 7])
    assert st.phys[:3] == [2, 0, 1]
    assert st.live == {0, 1, 2}
    assert st.release(0) == 2  # logical 0 now holds the request that lives on user 2
    assert st.live == {0, 1}
    assert st.release(0) is None
    # a new request into the freed logical slot 0 gets the free user 2 again (permutation: a free logical slot maps to a free user)
    assert st.claim([0]) == [2]


def test_slot_table_rejects_bad_remap_and_slots(expect_error):
    st = VS.SlotTable(4, 8)
    with expect_error(ValueError, ".*"):
        st.apply_remap([0, 0, 1, 2])
    with expect_error(ValueError, ".*"):
        st.claim([4])
    with expect_error(ValueError, ".*"):
        st.claim([1, 1])
    with expect_error(ValueError, ".*"):
        VS.SlotTable(9, 8)


def test_build_prefill_batch_masks_and_pads():
    toks = torch.tensor([[5, 6, 7, 0], [9, 8, 0, 0]], dtype=torch.int32)
    t, lens, act, idx = VS.build_prefill_batch(8, [3, 6], toks, [3, 2])
    assert t.shape == (8, 3) and t.dtype == torch.long
    assert lens.tolist() == [0, 0, 0, 3, 0, 0, 2, 0]
    assert act.tolist() == [False, False, False, True, False, False, True, False]
    assert t[3].tolist() == [5, 6, 7] and t[6].tolist() == [9, 8, 0]
    assert idx == {3: 0, 6: 1}
    assert t[0].eq(0).all()


def test_build_prefill_batch_refill_live_users():
    toks = torch.tensor([[1, 2]], dtype=torch.int32)
    ctx = {4: torch.tensor([10, 11, 12, 13, 14])}
    t, lens, act, _ = VS.build_prefill_batch(8, [0], toks, [2], ctx)
    assert t.shape == (8, 5)
    assert lens.tolist() == [2, 0, 0, 0, 5, 0, 0, 0] and act[4] and act[0] and int(act.sum()) == 2
    assert t[4].tolist() == [10, 11, 12, 13, 14]


def test_build_prefill_batch_rejects_bad_length(expect_error):
    with expect_error(ValueError, ".*"):
        VS.build_prefill_batch(4, [0], torch.zeros(1, 3, dtype=torch.int32), [0])
    with expect_error(ValueError, ".*"):
        VS.build_prefill_batch(4, [0], torch.zeros(1, 3, dtype=torch.int32), [4])


def test_decode_inputs_and_scatter_follow_the_slot_table():
    st = VS.SlotTable(6, 8)
    st.claim([0, 1, 2])
    st.apply_remap([1, 2, 0, 3, 4, 5])  # logical 0 -> user 1, 1 -> user 2, 2 -> user 0
    tokens = torch.tensor([[11], [22], [33], [0], [0], [0]], dtype=torch.int32)
    pos = torch.tensor([5, 6, 7, -1, -1, -1])
    tok_b, pos_b, rows = VS.build_decode_inputs(8, st, tokens, pos)
    assert tok_b.tolist() == [33, 11, 22, 0, 0, 0, 0, 0]
    assert pos_b.tolist() == [7, 5, 6, 0, 0, 0, 0, 0]
    assert rows == [(0, 1, 5), (1, 2, 6), (2, 0, 7)]
    out = torch.arange(100, 108).reshape(8, 1)  # model output per user
    scat = VS.scatter_rows(out, rows, 6)
    assert scat.reshape(-1).tolist() == [101, 102, 100, 0, 0, 0]


def test_sampling_helpers():
    sp = SimpleNamespace(temperature=[0.0, 0.7, 0.0], top_k=[1, 40, -1], enable_log_probs=[False, False, False])
    assert not VS.sampling_wants_greedy(sp)
    assert VS.sampling_wants_greedy(sp, [0, 2])
    assert not VS.sampling_wants_greedy(sp, [1])
    sp2 = SimpleNamespace(temperature=torch.tensor([0.5]), top_k=torch.tensor([1]))
    assert VS.sampling_wants_greedy(sp2)  # top_k 1 is greedy
    assert not VS.wants_logprobs(sp)
    assert VS.wants_logprobs(SimpleNamespace(enable_log_probs=torch.tensor([False, True])))
    assert not VS.wants_logprobs(SimpleNamespace(enable_log_probs=torch.tensor([False, True])), [0])


# ---- the adapter against a fake model ---------------------------------------------------------------------------------------------------------
VOCAB = 11


class FakeModel:
    """Stand-in of dsv41_model.Model: first token = (sum of the prompt) % 7, decode token = (fed token + 1) % 7; records every call."""

    def __init__(self, B, use_indexer=False, max_ctx=4096):
        self.B, self.use_indexer, self.max_ctx, self.num_pages = B, use_indexer, max_ctx, 100
        self.timing = {}
        self.pool = SimpleNamespace(dtype="bf16", released=[], release=lambda p: self.pool.released.append(p))
        self.args = SimpleNamespace(vocab_size=VOCAB)
        self.log = []
        self.last = None

    def release_trace(self):
        self.log.append("release_trace")

    def prefill_forward(
        self,
        tokens,
        prompt_lens,
        chunk=None,
        max_new_tokens=0,
        want_logits=False,
        enable_trace=True,
        s_pad_max=None,
        active=None,
    ):
        self.log.append(("prefill", tokens.clone(), prompt_lens.clone(), chunk, s_pad_max, active.clone()))
        first = torch.full((self.B,), -1, dtype=torch.long)
        logits = torch.zeros(self.B, VOCAB)
        for b in range(self.B):
            if bool(active[b]):
                first[b] = int(tokens[b, : int(prompt_lens[b])].sum()) % 7
                logits[b, int(first[b])] = 1.0
        return first, logits if want_logits else None

    def decode_forward(self, tokens, current_pos, enable_trace=True, reload_inputs=True):
        self.log.append(("decode", tokens.clone(), current_pos.clone()))
        out = (tokens + 1) % 7
        self.last = torch.zeros(self.B, VOCAB)
        self.last[torch.arange(self.B), out] = 1.0
        return out

    def read_logits(self):
        return self.last


def make(max_num_seqs=8, **kw):
    B = VS.padded_batch(max_num_seqs)
    m = FakeModel(B, **kw)
    gen = SimpleNamespace(m=m, prefill_chunk=None, auto_chunk=lambda max_len: 256)
    return DeepseekV41ForCausalLM(gen, max_num_seqs, 4096), m


GREEDY = SimpleNamespace(temperature=[0.0] * 8, top_k=[1] * 8, enable_log_probs=[False] * 8)
HOT = SimpleNamespace(temperature=[0.8] * 8, top_k=[-1] * 8, enable_log_probs=[False] * 8)


def test_capabilities_declared():
    c = DeepseekV41ForCausalLM.model_capabilities
    assert c["supports_sample_on_device"] and c["max_device_top_k"] == 1
    assert not c["supports_prefix_caching"] and not c["supports_chunked_prefill"] and not c["supports_async_decode"]
    assert not c["supports_device_penalties"]


def test_max_tokens_all_users_and_kv_cache_shape(monkeypatch, expect_error):
    f = DeepseekV41ForCausalLM.get_max_tokens_all_users
    assert f(max_model_len=65536, max_num_seqs=128) == 65536 * 128
    monkeypatch.setenv("DSV41_VLLM_MAX_TOKENS_ALL_USERS", "1000")
    assert f(max_model_len=65536, max_num_seqs=128) == 1000
    monkeypatch.delenv("DSV41_VLLM_MAX_TOKENS_ALL_USERS")
    gen, _ = make()
    cache = gen.allocate_kv_cache((8 * 33, 1, 128, 512), torch.bfloat16, 40)
    assert len(cache) == 40 and cache[0][0].numel() == 0
    with expect_error(ValueError, ".*"):
        gen.allocate_kv_cache((100, 1, 64, 512), torch.bfloat16, 40)  # not 128-token blocks


def test_prefill_decode_roundtrip_with_remap_and_release():
    gen, m = make(8)
    toks = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.int32)
    first = gen.prefill_forward(toks, prompt_lens=[3, 2], empty_slots=[0, 1], sampling_params=GREEDY)
    assert first.dtype == torch.int32 and first.tolist() == [6 % 7, 9 % 7]
    log = m.log[-1]
    assert (
        log[0] == "prefill"
        and log[2].tolist() == [3, 2, 0, 0, 0, 0, 0, 0]
        and log[5].tolist() == [True, True] + [False] * 6
    )
    assert m.log[0] == "release_trace"
    assert log[3] == 256 and log[4] == 1024  # chunk, S_pad bucket
    # decode, plugin padded to 8 rows, request order swapped by the plugin (slot_remap row0 <- old slot 1, row1 <- old slot 0)
    t = torch.tensor([[first[1]], [first[0]], [0], [0], [0], [0], [0], [0]], dtype=torch.int32)
    pos = torch.tensor([2, 3, -1, -1, -1, -1, -1, -1])
    remap = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
    out = gen.decode_forward(t, pos, sampling_params=GREEDY, slot_remap=remap)
    assert out.shape == (8, 1) and out.dtype == torch.int32
    # logical row 0 is the request of user 1 (prompt 4,5) fed its first token at pos 2
    assert out[0, 0] == (int(first[1]) + 1) % 7 and out[1, 0] == (int(first[0]) + 1) % 7 and out[2:].eq(0).all()
    d = m.log[-1]
    assert d[1][1] == first[1] and d[1][0] == first[0] and d[2][:2].tolist() == [3, 2]
    assert gen.book.n[1] == 3 and gen.book.n[0] == 4
    # finish the request now in logical row 0 (user 1)
    gen.release_request(0)
    assert m.pool.released == [1] and 1 not in gen.slots.live and gen.slots.live == {0}
    # next decode: only the request in logical row 1 ... after the plugin's gather (identity here: row 1 stays)
    out = gen.decode_forward(t, torch.tensor([-1, 4, -1, -1, -1, -1, -1, -1]), sampling_params=GREEDY)
    assert out[1, 0] == (int(first[0]) + 1) % 7 and out[0, 0] == 0


def test_second_prefill_keeps_live_users_when_indexer_off():
    gen, m = make(8, use_indexer=False)
    gen.prefill_forward(
        torch.tensor([[1, 2, 3]], dtype=torch.int32), prompt_lens=[3], empty_slots=[0], sampling_params=GREEDY
    )
    gen.prefill_forward(
        torch.tensor([[7, 7]], dtype=torch.int32), prompt_lens=[2], empty_slots=[1], sampling_params=GREEDY
    )
    log = m.log[-1]
    assert log[5].tolist() == [False, True] + [False] * 6  # user 0 (live) is left alone
    assert log[2].tolist()[0] == 0


def test_second_prefill_reprefills_live_users_when_indexer_on():
    gen, m = make(8, use_indexer=True)
    first = gen.prefill_forward(
        torch.tensor([[1, 2, 3]], dtype=torch.int32), prompt_lens=[3], empty_slots=[0], sampling_params=GREEDY
    )
    gen.decode_forward(
        torch.tensor([[int(first[0])]] + [[0]] * 7, dtype=torch.int32),
        torch.tensor([3] + [-1] * 7),
        sampling_params=GREEDY,
    )
    f2 = gen.prefill_forward(
        torch.tensor([[7, 7]], dtype=torch.int32), prompt_lens=[2], empty_slots=[1], sampling_params=GREEDY
    )
    log = m.log[-1]
    assert log[5].tolist()[:2] == [True, True]
    assert log[2].tolist()[:2] == [4, 2]  # user 0 re-prefilled from prompt (3) + the fed token (1)
    assert log[1][0, :4].tolist() == [1, 2, 3, int(first[0])]
    assert f2.tolist() == [14 % 7]


def test_host_sampling_returns_logits_and_hot_requests_fail_on_device(expect_error):
    gen, m = make(8)
    toks = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    logits = gen.prefill_forward(toks, prompt_lens=[3], empty_slots=[0], sampling_params=None)
    assert logits.shape == (1, 1, VOCAB) and int(logits.argmax()) == 6 % 7
    dl = gen.decode_forward(
        torch.tensor([[6]] + [[0]] * 7, dtype=torch.int32), torch.tensor([3] + [-1] * 7), sampling_params=None
    )
    assert dl.shape == (8, 1, VOCAB) and int(dl[0, 0].argmax()) == 0 and dl[1:].abs().sum() == 0
    with expect_error(ValueError, "temperature"):
        gen.prefill_forward(toks, prompt_lens=[3], empty_slots=[1], sampling_params=HOT)
    with expect_error(ValueError, "temperature"):
        gen.decode_forward(
            torch.tensor([[6]] + [[0]] * 7, dtype=torch.int32), torch.tensor([3] + [-1] * 7), sampling_params=HOT
        )
    lp = SimpleNamespace(temperature=[0.0] * 8, top_k=[1] * 8, enable_log_probs=[True] * 8)
    with expect_error(ValueError, "logprobs"):
        gen.prefill_forward(toks, prompt_lens=[3], empty_slots=[1], sampling_params=lp)


def test_decode_without_prefill_and_overlong_prompt_fail_loudly(expect_error):
    gen, _ = make(8)
    with expect_error(RuntimeError, "never prefilled"):
        gen.decode_forward(
            torch.tensor([[1]] + [[0]] * 7, dtype=torch.int32), torch.tensor([3] + [-1] * 7), sampling_params=GREEDY
        )
    with expect_error(ValueError, "max_model_len"):
        gen.prefill_forward(
            torch.zeros(1, 5000, dtype=torch.int32) + 3, prompt_lens=[5000], empty_slots=[0], sampling_params=GREEDY
        )


def test_default_empty_slots_are_the_lowest_free():
    gen, m = make(8)
    gen.prefill_forward(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), prompt_lens=[2, 2], sampling_params=GREEDY)
    assert gen.slots.live == {0, 1}
    gen.prefill_forward(torch.tensor([[5, 6]], dtype=torch.int32), prompt_lens=[2], sampling_params=GREEDY)
    assert gen.slots.live == {0, 1, 2}


# ---- registration / config shim --------------------------------------------------------------------------------------------------------------
def test_bundle_metadata_resolves_to_the_adapter():
    meta = json.loads((PKG / "vllm_metadata.json").read_text())
    assert meta["arch"] == "DeepseekV41ForCausalLM"  # the plugin registers it as TTDeepseekV41ForCausalLM
    mod, _, cls = meta["main_class"].partition(":")
    assert getattr(importlib.import_module(mod), cls) is DeepseekV41ForCausalLM


def test_vllm_config_shim_matches_checkpoint_dims():
    shim = json.loads((PKG / "vllm_config" / "config.json").read_text())
    ckpt = Path(os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")) / "config.json"
    if not ckpt.exists():
        pytest.skip("checkpoint config not available")
    real = json.loads(ckpt.read_text())["text_config"]
    for k in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        assert shim[k] == real[k], k
    assert "quantization_config" not in shim


def test_chat_template_matches_checkpoint_encoder():
    ckpt = Path(os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash"))
    if not (ckpt / "encoding" / "encoding.py").exists():
        pytest.skip("checkpoint encoder not available")
    import sys

    from transformers import AutoTokenizer

    sys.path.insert(0, str(ckpt / "encoding"))
    from encoding import encode_messages

    tok = AutoTokenizer.from_pretrained(str(ckpt))
    tpl = (PKG / "vllm_config" / "chat_template.jinja").read_text()
    for msgs in (
        [{"role": "user", "content": "Hi"}],
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}],
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C\nx <y>"},
        ],
    ):
        got = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True)
        assert got == encode_messages(msgs, thinking_mode="chat")


# ---- interleaved / chunked prefill (DSV41_VLLM_INTERLEAVE=1; tt/dsv41_model.Model.prefill_interleaved) ------------------------------------------
def test_slot_table_bind_keeps_a_permutation():
    st = VS.SlotTable(6, 8)
    st.claim([0, 1])
    st.bind(4, 1)  # the continuation of the request on user 1 shows up in logical slot 4
    assert st.phys[4] == 1 and st.phys[1] == 4 and sorted(st.phys) == list(range(6))
    st.bind(4, 1)  # no-op
    assert st.phys[4] == 1


def test_prefill_tracker_finds_continuations_by_prefix():
    tr = VS.PrefillTracker()
    t = torch.arange(100, 1100)
    tr.update(3, 512, t)
    assert tr.find(512, t[:512]) == 3
    assert tr.find(512, t[:512] + 1) is None and tr.find(256, t[:256]) is None
    assert tr.parked() == {3: 512}
    tr.drop(3)
    assert tr.parked() == {} and tr.find(512, t[:512]) is None


def test_decode_inputs_park_in_progress_users():
    st = VS.SlotTable(8, 8)
    st.claim([0, 5])
    tokens = torch.tensor([[9]] + [[0]] * 7, dtype=torch.int32)
    pos = torch.tensor([3] + [-1] * 7)
    _, pos_b, rows = VS.build_decode_inputs(8, st, tokens, pos, parked={5: 512, 0: 77})
    assert pos_b.tolist() == [3, 0, 0, 0, 0, 512, 0, 0] and rows == [(0, 0, 3)]  # a real row is never overridden


class FakeInterleaveModel(FakeModel):
    """prefill_interleaved of the fake: first token = (sum of tokens[:end]) % 7; keeps pf_resume like the real model (aligned ends are resumable)."""

    def __init__(self, B, **kw):
        super().__init__(B, **kw)
        self.pf_resume = {}
        self.calls = []

    def prefill_interleaved(self, items, chunk, s_pad_max=None, want_logits=False):
        self.calls.append(([(b, st, e) for b, _, st, e in items], chunk, s_pad_max, want_logits))
        out = {}
        for b, t, st, e in items:
            assert st == 0 or self.pf_resume.get(b) == st, "continuation without carried state"
            self.pf_resume.pop(b, None)
            if e % chunk == 0:
                self.pf_resume[b] = e
            tok = int(t[:e].sum()) % 7
            lg = torch.zeros(VOCAB)
            lg[tok] = 1.0
            out[b] = (tok, lg if want_logits else None)
        return out

    def release_user(self, p):
        self.pool.released.append(p)
        self.pf_resume.pop(p, None)


@pytest.fixture
def interleave_gen(monkeypatch):
    monkeypatch.setenv("DSV41_VLLM_INTERLEAVE", "1")
    monkeypatch.setenv("DSV41_VLLM_CHUNK", "512")
    B = VS.padded_batch(8)
    m = FakeInterleaveModel(B, use_indexer=True)
    gen = SimpleNamespace(m=m, prefill_chunk=None, auto_chunk=lambda max_len: 256)
    return DeepseekV41ForCausalLM(gen, 8, 4096), m


def test_chunked_prefill_capability_is_opt_in(monkeypatch):
    import models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm as GV

    assert DeepseekV41ForCausalLM.model_capabilities["supports_chunked_prefill"] is False  # default: off
    monkeypatch.setenv("DSV41_VLLM_INTERLEAVE", "1")
    try:
        assert importlib.reload(GV).DeepseekV41ForCausalLM.model_capabilities["supports_chunked_prefill"] is True
    finally:
        monkeypatch.delenv("DSV41_VLLM_INTERLEAVE")
        importlib.reload(GV)


def test_interleaved_prefill_does_not_reprefill_live_users(interleave_gen):
    gen, m = interleave_gen
    a = torch.arange(1, 301, dtype=torch.int32).reshape(1, -1)
    fa = gen.prefill_forward(a, prompt_lens=[300], empty_slots=[0], sampling_params=GREEDY)
    gen.decode_forward(
        torch.tensor([[int(fa[0])]] + [[0]] * 7, dtype=torch.int32),
        torch.tensor([300] + [-1] * 7),
        sampling_params=GREEDY,
    )
    b = torch.arange(5, 105, dtype=torch.int32).reshape(1, -1)
    fb = gen.prefill_forward(b, prompt_lens=[100], empty_slots=[1], sampling_params=GREEDY)
    items, chunk, s_pad, _ = m.calls[-1]
    assert (
        items == [(1, 0, 100)] and chunk == 512
    )  # ONLY the new user is prefilled (no re-prefill of user 0 although the indexer is on)
    assert (
        fb.tolist() == [int(b.sum()) % 7] and "release_trace" not in m.log[1:]
    )  # the decode trace is not torn down per prefill


def test_chunked_prompt_over_several_steps_with_slot_change_and_parking(interleave_gen):
    gen, m = interleave_gen
    t = torch.arange(10, 1210, dtype=torch.int32).reshape(
        1, -1
    )  # 1200-token prompt, chunk 512: [0,512) [512,1024) [1024,1200)
    other = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    fo = gen.prefill_forward(other, prompt_lens=[3], empty_slots=[0], sampling_params=GREEDY)
    gen.prefill_forward(t[:, :512], prompt_lens=[512], start_pos=[0], empty_slots=[3], sampling_params=None)
    assert m.calls[-1][0] == [(3, 0, 512)] and gen.inprog.parked() == {
        0: 3,
        3: 512,
    }  # (user 0 just finished and has not decoded yet)
    # a decode step of the other request in between: the in-progress user is parked at 512 (token 0), not at position 0
    gen.decode_forward(
        torch.tensor([[int(fo[0])]] + [[0]] * 7, dtype=torch.int32),
        torch.tensor([3] + [-1] * 7),
        sampling_params=GREEDY,
    )
    d = m.log[-1]
    assert d[0] == "decode" and d[2][3] == 512 and d[1][3] == 0 and d[2][0] == 3
    # the plugin re-picks the state slot of the continuation (slot 5 now): it is recognised by its token prefix and mapped back to user 3
    gen.prefill_forward(t[:, :1024], prompt_lens=[1024], start_pos=[512], empty_slots=[5], sampling_params=None)
    assert m.calls[-1][0] == [(3, 512, 1024)] and gen.slots.phys[5] == 3 and gen.slots.phys[3] == 5
    f = gen.prefill_forward(t, prompt_lens=[1200], start_pos=[1024], empty_slots=[5], sampling_params=GREEDY)
    assert m.calls[-1][0] == [(3, 1024, 1200)] and f.tolist() == [int(t.sum()) % 7]
    # now it decodes: it is no longer parked
    gen.decode_forward(
        torch.tensor([[0]] * 5 + [[int(f[0])]] + [[0]] * 2, dtype=torch.int32),
        torch.tensor([-1] * 5 + [1200, -1, -1]),
        sampling_params=GREEDY,
    )
    d = m.log[-1]
    assert d[1][3] == int(f[0]) and d[2][3] == 1200 and gen.inprog.parked() == {}


def test_continuation_without_carried_state_recomputes_from_zero(interleave_gen):
    gen, m = interleave_gen
    t = torch.arange(10, 1210, dtype=torch.int32).reshape(1, -1)
    gen.prefill_forward(t[:, :512], prompt_lens=[512], start_pos=[0], empty_slots=[2], sampling_params=None)
    m.pf_resume.clear()  # e.g. the trace was re-captured: the carried state is gone
    gen.prefill_forward(t[:, :1024], prompt_lens=[1024], start_pos=[512], empty_slots=[2], sampling_params=None)
    assert m.calls[-1][0] == [(2, 0, 1024)]
    # an unknown continuation (prefix does not match anything in progress) is recomputed as well
    gen.prefill_forward(t[:, :700] + 1, prompt_lens=[700], start_pos=[512], empty_slots=[4], sampling_params=None)
    assert m.calls[-1][0] == [(gen.slots.phys[4], 0, 700)]


def test_release_drops_in_progress_user(interleave_gen):
    gen, m = interleave_gen
    t = torch.arange(10, 1210, dtype=torch.int32).reshape(1, -1)
    gen.prefill_forward(t[:, :512], prompt_lens=[512], start_pos=[0], empty_slots=[1], sampling_params=None)
    assert gen.inprog.parked() == {1: 512}
    gen.release_request(1)
    assert gen.inprog.parked() == {} and m.pool.released == [1]


def test_start_position_without_interleave_is_rejected(expect_error):
    gen, _ = make(8)
    with expect_error(ValueError, "INTERLEAVE"):
        gen.prefill_forward(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            prompt_lens=[4],
            start_pos=[2],
            empty_slots=[0],
            sampling_params=GREEDY,
        )


def test_claim_balanced_spreads_new_requests_over_mesh_rows():
    st = VS.SlotTable(8, 8)  # 2 users per mesh row would be rows 0..3 of users 0,1 | 2,3 | 4,5 | 6,7
    load = {}
    got = [st.claim_balanced(s, load, 2) for s in (0, 1, 2, 3)]
    assert sorted(g // 2 for g in got) == [0, 1, 2, 3] and sorted(st.phys) == list(
        range(8)
    )  # four requests, four different rows
    assert st.live == set(got)
    st2 = VS.SlotTable(8, 8)
    p = st2.claim_balanced(5, {0: 5, 1: 5, 2: 5, 3: 0}, 2)  # row 3 is the least loaded: the request lands there
    assert p // 2 == 3 and sorted(st2.phys) == list(range(8)) and st2.phys[5] == p


def test_interleaved_prefill_balances_rows_with_few_prefill_slots(interleave_gen):
    gen, m = interleave_gen
    m.U, m.Up = 2, 1
    t1 = torch.arange(10, 2058, dtype=torch.int32).reshape(1, -1)
    t2 = t1 + 7
    gen.prefill_forward(t1[:, :512], prompt_lens=[512], start_pos=[0], empty_slots=[0], sampling_params=None)
    gen.prefill_forward(t2[:, :512], prompt_lens=[512], start_pos=[0], empty_slots=[1], sampling_params=None)
    users = [c[0][0][0] for c in m.calls]
    assert users[0] // 2 != users[1] // 2  # the second concurrent prompt goes to another mesh row
