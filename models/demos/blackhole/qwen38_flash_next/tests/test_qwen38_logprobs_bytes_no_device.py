# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Logprobs ``bytes`` are each token's raw UTF-8 bytes, so their concatenation rebuilds the text (no device)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as step
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import template_decoder, template_token_bytes

SPECIAL = "<|im_end|>"


@pytest.fixture(scope="module")
def template():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {char: index for index, char in enumerate(sorted(alphabet))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast.add_special_tokens({"additional_special_tokens": [SPECIAL]})
    return SimpleNamespace(tokenizer=fast)


@pytest.mark.parametrize("text", ["\U0001fae0", "\ua66e", "a\U0001f9d1\u200d\U0001f680b", "hello", "caf\u00e9"])
def test_concatenated_token_bytes_rebuild_the_text(template, text) -> None:
    ids = template.tokenizer.encode(text, add_special_tokens=False)
    decode = lambda token: template_decoder(template)([token])  # noqa: E731
    token_bytes = template_token_bytes(template)
    items = [step.logprobs_content_item(None, token, decode, token_bytes) for token in ids]
    assert b"".join(bytes(item["bytes"]) for item in items) == text.encode("utf-8")


def test_a_byte_fallback_token_keeps_its_own_bytes_and_text(template) -> None:
    ids = template.tokenizer.encode("\U0001fae0", add_special_tokens=False)
    assert len(ids) > 1
    decode = lambda token: template_decoder(template)([token])  # noqa: E731
    item = step.logprobs_content_item(None, ids[0], decode, template_token_bytes(template))
    assert item["bytes"] == [0xF0] and item["token"] == "\ufffd"
    # Without a byte function the item keeps the old behaviour (bytes of the decoded text).
    assert step.logprobs_content_item(None, ids[0], decode)["bytes"] == [239, 191, 189]


def test_whole_character_and_special_tokens_are_unchanged(template) -> None:
    decode = lambda token: template_decoder(template)([token])  # noqa: E731
    token_bytes = template_token_bytes(template)
    (ascii_id,) = template.tokenizer.encode("a", add_special_tokens=False)
    assert token_bytes(ascii_id) == b"a"
    special_id = template.tokenizer.convert_tokens_to_ids(SPECIAL)
    assert token_bytes(special_id) == SPECIAL.encode()
    assert step.logprobs_content_item(None, special_id, decode, token_bytes)["bytes"] == list(SPECIAL.encode())
