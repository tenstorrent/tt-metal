# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from models.common.readiness_check.generate import (
    _chat_or_plain_prompt_tokens,
    _generation_stop_ids,
    _normal_token_ids,
    _resolve_prompt_text,
    _safe_pad_id,
)


@pytest.mark.parametrize("chat_template, expected", [(True, [2, 1, 3]), (False, [1])])
def test_prompt_tokens_are_flat_ids_with_hf_tokenizer(chat_template, expected):
    backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "user": 2, "assistant": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=(
            "{{ messages[0]['role'] }} {{ messages[0]['content'] }} "
            "{% if add_generation_prompt %}assistant{% endif %}"
        ),
    )

    assert _chat_or_plain_prompt_tokens(tokenizer, "hello", chat_template=chat_template) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, []), (7, [7]), ([7, None, 9], [7, 9])],
)
def test_normal_token_ids(value, expected):
    assert _normal_token_ids(value) == expected


def test_generation_stop_ids_combines_and_deduplicates_tokenizer_and_model_ids():
    tokenizer = SimpleNamespace(
        eos_token_id=[2, 3],
        eot_token_id=None,
        unk_token_id=0,
        get_vocab=lambda: {"<|eot_id|>": 4},
        convert_tokens_to_ids=lambda token: 4 if token == "<|eot_id|>" else 0,
    )
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[3, 5]))

    assert _generation_stop_ids(tokenizer, model) == [2, 3, 5, 4]


def test_generation_stop_ids_requires_eos_or_eot():
    tokenizer = SimpleNamespace(
        eos_token_id=None,
        eot_token_id=None,
        unk_token_id=0,
        get_vocab=lambda: {},
        convert_tokens_to_ids=lambda _token: 0,
    )
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=None))

    with pytest.raises(RuntimeError, match="Could not determine eos/eot"):
        _generation_stop_ids(tokenizer, model)


def test_safe_pad_id_prefers_non_bos_pad_and_falls_back_to_stop_id():
    assert _safe_pad_id(SimpleNamespace(pad_token_id=8, bos_token_id=1), [2]) == 8
    assert _safe_pad_id(SimpleNamespace(pad_token_id=1, bos_token_id=1), [2]) == 2
    assert _safe_pad_id(SimpleNamespace(pad_token_id=None, bos_token_id=1), []) is None


def test_resolve_prompt_text_supports_literal_and_file_sources(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")
    unused_aime_file = tmp_path / "aime.json"

    assert (
        _resolve_prompt_text(
            "text",
            prompt="literal prompt",
            prompt_file=None,
            aime24_prompts_file=unused_aime_file,
            aime24_prompt_index=0,
        )
        == "literal prompt"
    )
    assert (
        _resolve_prompt_text(
            "file",
            prompt=None,
            prompt_file=prompt_file,
            aime24_prompts_file=unused_aime_file,
            aime24_prompt_index=0,
        )
        == "file prompt"
    )


def test_resolve_prompt_text_requires_source_specific_argument(tmp_path):
    with pytest.raises(ValueError, match="--prompt is required"):
        _resolve_prompt_text(
            "text",
            prompt=None,
            prompt_file=None,
            aime24_prompts_file=tmp_path / "aime.json",
            aime24_prompt_index=0,
        )
