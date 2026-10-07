# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests of the vLLM interface (tt/vllm_state.py, tt/generator_vllm.py): batch padding, slot map / plugin ``slot_remap`` handling, prefill / decode input
building and output scatter, sampling fallbacks, block accounting, bundle metadata, chat template. A fake model stands in for ``tt/dsv41_model.Model``.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_vllm_interface.py
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


def test_class_satisfies_vllm_text_generation_protocol():
    """ModelConfig resolves --runner generate through vLLM's is_text_generation_model before the TT plugin loads the model (stubs: embed_input_ids / forward / compute_logits)."""
    pytest.importorskip("vllm")
    from vllm.model_executor.models.interfaces_base import is_text_generation_model

    from models.demos.blackhole.deepseek_v41_flash.tt.generator_vllm import DeepseekV41ForCausalLM

    assert is_text_generation_model(DeepseekV41ForCausalLM)
