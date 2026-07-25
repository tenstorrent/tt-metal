# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The thinking prompt contract on the offline path (#48291).

``enable_thinking`` was absent from the whole DiffusionGemma tree, so every offline
measurement -- including the early-halt probe whose "no-op" conclusion rests on entropy
floors of 0.155/0.138/0.506 nats -- ran the model against the non-thinking prompt even though
the released checkpoint is post-trained with a ``<|think|>`` turn.

These tests are CPU-only and weight-free (tokenizer only, no device, no model load). They pin
three things: the default render is unchanged, thinking actually changes the render, and a
template that IGNORES the flag raises instead of silently downgrading.
"""

import os

import pytest
import torch

from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

DG_CKPT = os.getenv("DG_CKPT", "google/diffusiongemma-26B-A4B-it")
PROMPT = "What is 17*23?"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(DG_CKPT, local_files_only=True)
    except Exception as load_error:  # gated / not cached in this environment
        pytest.skip(f"DiffusionGemma tokenizer unavailable ({type(load_error).__name__}): {load_error}")


class _IgnoresThinkingTokenizer:
    """A chat tokenizer whose template drops unknown kwargs -- the silent-downgrade shape."""

    def apply_chat_template(self, messages, *, add_generation_prompt=True, tokenize=False, **ignored):
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if not tokenize:
            return rendered
        return {"input_ids": list(range(len(rendered.split())))}


class _NoChatTemplateTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))


def test_default_render_is_unchanged_by_the_new_parameter(tokenizer):
    """enable_thinking=None must pass NOTHING to the template, byte-for-byte as before."""
    baseline = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True
    )
    got = tokenize_prompt(tokenizer, PROMPT)
    assert got.flatten().tolist() == list(baseline["input_ids"])


def test_thinking_changes_the_prompt_and_is_longer(tokenizer):
    plain = tokenize_prompt(tokenizer, PROMPT)
    thinking = tokenize_prompt(tokenizer, PROMPT, enable_thinking=True)
    assert thinking.shape[1] > plain.shape[1], "the <|think|> turn must add tokens"
    assert thinking.flatten().tolist() != plain.flatten().tolist()
    # The thinking turn is rendered as a system turn, so the user turn is pushed to the right.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False, enable_thinking=True
    )
    assert "<|think|>" in rendered


def test_explicit_false_matches_the_template_default(tokenizer):
    assert (
        tokenize_prompt(tokenizer, PROMPT, enable_thinking=False).flatten().tolist()
        == tokenize_prompt(tokenizer, PROMPT).flatten().tolist()
    )


def test_template_that_ignores_the_flag_raises_instead_of_downgrading(expect_error):
    """The regression that produced the malformed contract: a silently ignored request."""
    with expect_error(ValueError, match="being ignored"):
        tokenize_prompt(_IgnoresThinkingTokenizer(), PROMPT, enable_thinking=True)


def test_ignored_flag_is_allowed_when_thinking_is_not_requested():
    """enable_thinking=False on a template without thinking support is not an error."""
    assert tokenize_prompt(_IgnoresThinkingTokenizer(), PROMPT, enable_thinking=False).shape[0] == 1


def test_pretokenized_prompt_rejects_thinking(expect_error):
    ids = torch.tensor([[2, 105, 2364]], dtype=torch.long)
    with expect_error(ValueError, match="pre-tokenized"):
        tokenize_prompt(object(), ids, enable_thinking=True)


def test_tokenizer_without_chat_template_rejects_thinking(expect_error):
    with expect_error(ValueError, match="apply_chat_template"):
        tokenize_prompt(_NoChatTemplateTokenizer(), PROMPT, enable_thinking=True)
