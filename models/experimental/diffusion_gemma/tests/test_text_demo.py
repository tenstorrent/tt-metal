# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.demo import text_demo
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
