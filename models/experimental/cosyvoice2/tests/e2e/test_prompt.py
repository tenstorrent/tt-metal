# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt/prompt.py: the mode table, `PromptContext`'s checks, `RandomSources`, and `from_npz` over every corpus
case `scripts/prepare_inputs.py` wrote (`COSYVOICE2_INPUTS`; skipped when unset). Host only; no device."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest
import torch

from models.experimental.cosyvoice2.tt.prompt import MODES, PromptContext, RandomSources, describe_mode

INPUTS_DIR = os.environ.get("COSYVOICE2_INPUTS", "")


def _ctx(**over):
    s = 5
    kw = dict(
        mode="zero_shot",
        lang="en",
        prompt_text_ids=torch.ones(1, 3, dtype=torch.int32),
        llm_prompt_speech_tokens=torch.ones(1, s, dtype=torch.int32),
        flow_prompt_speech_tokens=torch.ones(1, s, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 2 * s, 80),
        embedding=torch.zeros(1, 192),
    )
    kw.update(over)
    return PromptContext(**kw)


def test_mode_table(expect_error):
    assert MODES == ("zero_shot", "cross_lingual", "instruct2")
    assert all(describe_mode(m)["flow_prompt"] for m in MODES)
    assert describe_mode("zero_shot") == {"llm_prompt_text": True, "llm_prompt_speech": True, "flow_prompt": True}
    with expect_error(ValueError, "unknown mode 'sft'"):
        describe_mode("sft")  # CosyVoice2-0.5B ships no spk2info.pt


def test_prompt_context_refuses_inconsistent_fields(expect_error):
    _ctx()  # well formed
    with expect_error(ValueError, "llm_prompt_speech_tokens must be set"):
        _ctx(llm_prompt_speech_tokens=None)  # zero_shot needs the LLM's speech prompt
    with expect_error(ValueError, "llm_prompt_speech_tokens must be None"):
        _ctx(mode="cross_lingual", prompt_text_ids=None)  # ... and cross_lingual must not have it
    with expect_error(ValueError, "prompt_feat .* is not"):
        _ctx(prompt_feat=torch.zeros(1, 9, 80))  # feat must be 2 x tokens
    with expect_error(ValueError, "embedding .* is not"):
        _ctx(embedding=torch.zeros(1, 1, 192))
    ctx = _ctx(mode="cross_lingual", prompt_text_ids=None, llm_prompt_speech_tokens=None)
    assert (ctx.n_prompt_tokens, ctx.prompt_mel_frames) == (5, 10)


def test_random_sources(expect_error):
    fresh = RandomSources().sine_noise_for(480, 9)
    assert fresh.shape == (1, 480, 9)
    captured = torch.randn(1, 480, 9)
    assert RandomSources(sine_noise=captured).sine_noise_for(480, 9) is captured
    with expect_error(ValueError, "captured sine_noise"):
        RandomSources(sine_noise=captured).sine_noise_for(960, 9)


def _inputs():
    return sorted(glob.glob(os.path.join(INPUTS_DIR, "*.npz"))) if INPUTS_DIR else []


@pytest.mark.skipif(not _inputs(), reason="set COSYVOICE2_INPUTS to scripts/prepare_inputs.py's --out-dir")
def test_from_npz_reads_every_corpus_case():
    for path in _inputs():
        ctx = PromptContext.from_npz(path)
        with np.load(path) as d:
            case = json.loads(str(d["case_json"]))
            assert torch.equal(ctx.flow_prompt_speech_tokens, torch.from_numpy(d["flow_prompt_speech_tokens"]))
            # zero_shot: upstream's frontend gives the LLM and the flow the same prompt tokens and embedding
            assert np.array_equal(d["llm_prompt_speech_tokens"], d["flow_prompt_speech_tokens"])
            assert np.array_equal(d["llm_embedding"], d["flow_embedding"])
        assert (ctx.mode, ctx.lang) == (case["mode"], case["lang"]) == ("zero_shot", "en")
        assert ctx.prompt_text_ids.dtype == ctx.flow_prompt_speech_tokens.dtype == torch.int32
        assert ctx.prompt_feat.dtype == ctx.embedding.dtype == torch.float32
        assert ctx.prompt_mel_frames == 2 * ctx.n_prompt_tokens
        assert ctx.meta["case"]["case_id"] == case["case_id"]
        assert len(ctx.meta["segments"]) == len(ctx.meta["segment_text_ids"]) >= 1
