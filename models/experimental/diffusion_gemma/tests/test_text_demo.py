# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.demo import text_demo
from models.experimental.diffusion_gemma.tt import denoise_forward as DF
from models.experimental.diffusion_gemma.tt import generate as G
from models.experimental.diffusion_gemma.tt.generate import PromptPrefill


def test_prefill_prompt_tokenizes_and_writes_prompt_kv(monkeypatch):
    calls = {}
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)
    result = PromptPrefill(prompt_len=3, cache_len=32)

    def fake_tokenize_prompt(tokenizer, prompt):
        calls["tokenize"] = (tokenizer, prompt)
        return prompt_tokens

    def fake_prefill_prompt_tokens(tt_model, tokens):
        calls["prefill"] = (tt_model, tokens)
        return result

    monkeypatch.setattr(G, "tokenize_prompt", fake_tokenize_prompt)
    monkeypatch.setattr(G, "prefill_prompt_tokens", fake_prefill_prompt_tokens)

    checkpoint_model_inputs = SimpleNamespace(tokenizer="tokenizer", tt_model="tt-model")

    out = text_demo._prefill_prompt(checkpoint_model_inputs, "hello")

    assert out == result
    assert calls["tokenize"] == ("tokenizer", "hello")
    assert calls["prefill"] == ("tt-model", prompt_tokens)


def test_text_demo_rejects_conflicting_smoke_modes():
    parser = text_demo.build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--build-only", "--prefill-only"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--prefill-only", "--adapter-only"])


def test_adapter_logits_once_prefills_and_calls_real_builder(monkeypatch):
    calls = {}
    prompt_tokens = torch.tensor([[4, 5, 6]], dtype=torch.long)

    class _FakeDeviceCanvas:
        shape = (1, 1, 256, 1)

        def __init__(self):
            self.deallocated = False

        def deallocate(self, force):
            self.deallocated = force

    class _FakeLogits:
        shape = (1, 1, 256, 1024)

    class _FakeAdapter:
        def __init__(self):
            self.reset_called = False

        def __call__(self, canvas, step):
            calls["adapter_call"] = (canvas, step)
            return _FakeLogits()

        def reset(self):
            self.reset_called = True
            calls["adapter_reset"] = True

    fake_canvas = _FakeDeviceCanvas()
    fake_adapter = _FakeAdapter()

    def fake_tokenize_prompt(tokenizer, prompt):
        calls["tokenize"] = (tokenizer, prompt)
        return prompt_tokens

    def fake_prefill_prompt_tokens(tt_model, tokens):
        calls["prefill"] = (tt_model, tokens)
        return PromptPrefill(prompt_len=3, cache_len=32)

    def fake_builder_factory(state_dict, **kwargs):
        calls["builder_factory"] = (state_dict, kwargs)

        def fake_builder(tt_model, **builder_kwargs):
            calls["builder"] = (tt_model, builder_kwargs)
            return fake_adapter

        return fake_builder

    def fake_host_canvas_to_device(mesh_device, canvas):
        calls["host_canvas"] = (mesh_device, canvas)
        return fake_canvas

    monkeypatch.setattr(G, "tokenize_prompt", fake_tokenize_prompt)
    monkeypatch.setattr(G, "prefill_prompt_tokens", fake_prefill_prompt_tokens)
    monkeypatch.setattr(G, "host_canvas_to_device", fake_host_canvas_to_device)
    monkeypatch.setattr(DF, "make_generation_logits_fn_builder_from_checkpoint_state", fake_builder_factory)

    tt_model = SimpleNamespace(mesh_device="mesh", hf_config=SimpleNamespace(vocab_size=1024))
    checkpoint_model_inputs = SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=1024),
        tt_model=tt_model,
        state_dict={"raw": "state"},
    )

    out = text_demo._adapter_logits_once(checkpoint_model_inputs, "hello", canvas_length=256, seed=123)

    assert out == (1, 1, 256, 1024)
    assert calls["tokenize"] == (checkpoint_model_inputs.tokenizer, "hello")
    assert calls["prefill"][0] is tt_model
    assert calls["prefill"][1] is prompt_tokens
    assert calls["builder_factory"] == ({"raw": "state"}, {"config": tt_model.hf_config})
    assert calls["builder"][0] is tt_model
    assert calls["builder"][1]["prompt_tokens"] is prompt_tokens
    assert calls["builder"][1]["prompt_len"] == 32
    assert calls["host_canvas"][0] == "mesh"
    assert calls["host_canvas"][1].shape == (1, 256)
    assert calls["adapter_call"] == (fake_canvas, 0)
    assert calls["adapter_reset"] is True
    assert fake_canvas.deallocated is True
