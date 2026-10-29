# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The single-user greedy speculative serving classes (Qwen36ResidentMTPForCausalLM, Qwen36ResidentDFlashForCausalLM)
on a fake decoder (CPU, no device): the prefill's logits are masked to the token the device decodes from (the plugin
samples a block-output model's prefill at temperature 1), the runner's anchor token must be that token, every decode
step returns exactly the class's block (EOS-padded only at the position limit), a release stops the decoder; and the
plugin resolves the DFlash class's capabilities (8-token adaptive block output, its KV extent).

usage: PYTHONPATH=<vllm-tt-plugin src>:<tt-metal> python -m pytest --noconftest -c /dev/null \\
    qwen36_resident/tests/test_resident_spec_serving.py   (a venv with vLLM)
"""
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("vllm_tt_plugin")
from qwen36_resident import generator_vllm as G

raises = pytest.raises  # allow-pytest.raises: run with --noconftest, there is no expect_error fixture

VOCAB, EOS = 50, 7
CONFIG = dict(vocab_size=VOCAB, eos_token_id=EOS, num_hidden_layers=4)


class FakeDecoder:
    """start(prompt) -> [vocab] logits; next_block: up to n committed tokens, fewer only at the position limit;
    stop()"""

    def __init__(self, mesh=None, ck=None, max_pos=None, prefill=True):
        self.stops, self.pos, self.first, self.max_pos, self.iterations = 0, None, None, max_pos, 0

    @property
    def max_prompt_tokens(self):
        return self.max_pos - 2 * 8 - 64

    def stats(self):
        return dict(iterations=self.iterations, committed=0, mean_accept=0.0)

    def start(self, prompt):
        lg = torch.randn(VOCAB, generator=torch.Generator().manual_seed(len(prompt)))
        self.first, self.pos = int(lg.argmax()), len(prompt) + 1
        return lg

    def next_block(self, n, *rest):
        limit, out = rest[-1], []
        while len(out) < n and self.pos < limit:
            out.append((self.pos * 3 + 1) % VOCAB if (self.pos * 3 + 1) % VOCAB != EOS else 0)
            self.pos += 1
        self.iterations += 1
        return out

    def stop(self):
        self.stops += 1
        self.iterations = 0


def build(cls, max_seq_len=64):
    if cls is G.Qwen36ResidentDFlashForCausalLM:
        fake = types.ModuleType("qwen36_resident.dflash")
        fake.ResidentDFlash = FakeDecoder
        with patch.dict(sys.modules, {"qwen36_resident.dflash": fake}):
            return cls(None, SimpleNamespace(config=CONFIG), max_seq_len, False)
    m = object.__new__(cls)  # the MTP class builds its device decoder in __init__
    m.mesh, m.mrope, m.vocab, m.max_seq_len, m.eos = None, False, VOCAB, max_seq_len, EOS
    m.spec, m._first = FakeDecoder(), None
    return m


CLASSES = [G.Qwen36ResidentMTPForCausalLM, G.Qwen36ResidentDFlashForCausalLM]


@pytest.mark.parametrize("cls", CLASSES, ids=["mtp", "dflash2"])
def test_prefill_is_masked_to_the_device_token(cls):
    m = build(cls)
    lg = m.prefill_forward(torch.arange(10)[None], None, None, torch.tensor([10]))
    assert lg.shape == (1, 1, VOCAB)
    finite = torch.isfinite(lg.reshape(-1)).nonzero().reshape(-1).tolist()
    assert finite == [m.spec.first] and float(lg.reshape(-1)[m.spec.first]) == 0.0
    # temperature 1 over the masked logits can only draw that token
    assert int(torch.multinomial(torch.softmax(lg.reshape(-1), -1), 1)) == m.spec.first


@pytest.mark.parametrize("cls", CLASSES, ids=["mtp", "dflash2"])
def test_decode_returns_exact_blocks_then_pads_at_the_limit(cls):
    m = build(cls, max_seq_len=40)
    m.prefill_forward(torch.arange(10)[None], None, None, torch.tensor([10]))
    anchor, widths = m.spec.first, []
    for _ in range(12):
        out = m.decode_forward(torch.tensor([[anchor]]), torch.tensor([10]))
        assert out.shape == (1, cls.block) and out.dtype == torch.int32
        widths.append(int((out != EOS).sum()))
        anchor = int(out[0, -1])
    full = (40 - 11) // cls.block
    assert widths[:full] == [cls.block] * full, "full blocks while the limit is far"
    assert widths[full] == (40 - 11) % cls.block and widths[-1] == 0, "EOS-padded at the position limit"


@pytest.mark.parametrize("cls", CLASSES, ids=["mtp", "dflash2"])
def test_a_wrong_anchor_raises_and_release_stops(cls):
    m = build(cls)
    m.prefill_forward(torch.arange(10)[None], None, None, torch.tensor([10]))
    with raises(RuntimeError, match="greedy only"):
        m.decode_forward(torch.tensor([[(m.spec.first + 1) % VOCAB]]), torch.tensor([10]))
    m.release_request(0)
    m.release_persistent_capture()
    m.release_request(0)
    assert m.spec.stops == 3 and m._first is None


def test_an_overlong_block_is_refused():
    m = build(G.Qwen36ResidentDFlashForCausalLM)
    m.prefill_forward(torch.arange(10)[None], None, None, torch.tensor([10]))
    m.spec.next_block = lambda n, limit: list(range(n + 1))
    with raises(RuntimeError, match="committed"):
        m.decode_forward(torch.tensor([[m.spec.first]]), torch.tensor([10]))


def test_the_plugin_resolves_the_dflash_block_output():
    from vllm_tt_plugin.config import get_tt_output_tokens_per_step, is_tt_adaptive_block_output_model

    from qwen36_resident.tests.test_vllm_rows_engine import vllm_config

    cls = G.Qwen36ResidentDFlashForCausalLM
    assert cls.greedy_only and cls.block == G.DFLASH_BLOCK == 8
    cfg = vllm_config(1, True, True, cls=cls)
    assert get_tt_output_tokens_per_step(cfg) == 8 and is_tt_adaptive_block_output_model(cfg)
    assert cls.model_capabilities["tt_block_kv_extent_tokens"] == 8 + 2 * G.DFLASH_VERIFY
    assert cfg.model_config.generation_config == "vllm", "a block-output model owns its generation defaults"


def test_dflash_takes_every_prompt_vllm_admits():
    """built DFLASH_POS_MARGIN past max_model_len, the decoder's prompt ceiling covers every admissible prompt"""
    m = build(G.Qwen36ResidentDFlashForCausalLM, max_seq_len=4096)
    assert m.spec.max_pos == 4096 + G.DFLASH_POS_MARGIN and m.spec.max_prompt_tokens >= 4096
    short = types.ModuleType("qwen36_resident.dflash")
    short.ResidentDFlash = lambda mesh, ck, max_pos, prefill=True: FakeDecoder(max_pos=max_pos - 1)
    with patch.dict(sys.modules, {"qwen36_resident.dflash": short}), raises(RuntimeError, match="below max_model_len"):
        G.Qwen36ResidentDFlashForCausalLM(None, SimpleNamespace(config=CONFIG), 4096, False)


def test_dflash_logs_a_request_s_stats_once():
    from loguru import logger

    m = build(G.Qwen36ResidentDFlashForCausalLM)
    m.prefill_forward(torch.arange(10)[None], None, None, torch.tensor([10]))
    m.decode_forward(torch.tensor([[m.spec.first]]), torch.tensor([10]))
    got = []
    sink = logger.add(lambda r: got.append(r.record["message"]), level="INFO")
    try:
        m.release_request(0)
        m.release_persistent_capture()
        m.release_request(0)
    finally:
        logger.remove(sink)
    assert [g for g in got if g.startswith("DFlash2 request")] == [
        "DFlash2 request: " + str(dict(iterations=1, committed=0, mean_accept=0.0))
    ]
