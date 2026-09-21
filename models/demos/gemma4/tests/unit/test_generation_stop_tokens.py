# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt import generator


class FakeTokenizer:
    def __init__(
        self,
        *,
        eos_token_id=None,
        eot_token_id=None,
        stop_tokens=None,
        special_tokens_map=None,
        vocab=None,
        unknown_conversion=0,
    ):
        self.eos_token_id = eos_token_id
        self.eot_token_id = eot_token_id
        if stop_tokens is not None:
            self.stop_tokens = stop_tokens
        self.special_tokens_map = special_tokens_map or {}
        self._vocab = vocab or {}
        self.unk_token_id = unknown_conversion
        self.unk_token = "<unk>"

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


@pytest.mark.parametrize("declared", [7, [7, 9, 7], (7, 9), [], None])
def test_resolve_stop_tokens_normalizes_generation_config(monkeypatch, declared):
    monkeypatch.setattr(
        generator.GenerationConfig,
        "from_pretrained",
        lambda model_path: SimpleNamespace(eos_token_id=declared),
    )
    tokenizer = FakeTokenizer(eos_token_id=1)
    expected = sorted({1, *generator._normalize_stop_token_ids(declared)})
    assert generator._resolve_stop_tokens(tokenizer, "checkpoint") == expected


def test_resolve_stop_tokens_merges_existing_and_named_tokens(monkeypatch):
    monkeypatch.setattr(
        generator.GenerationConfig,
        "from_pretrained",
        lambda model_path: SimpleNamespace(eos_token_id=[1, 106, 50]),
    )
    tokenizer = FakeTokenizer(
        eos_token_id=None,
        eot_token_id=106,
        stop_tokens=(50, 99),
        special_tokens_map={"eot_token": "<turn|>"},
        vocab={"<turn|>": 106},
    )
    assert generator._resolve_stop_tokens(tokenizer, "checkpoint") == [1, 50, 99, 106]


def test_resolve_stop_tokens_falls_back_when_generation_config_is_unreadable(monkeypatch):
    def fail(model_path):
        raise OSError("missing generation_config")

    monkeypatch.setattr(generator.GenerationConfig, "from_pretrained", fail)
    tokenizer = FakeTokenizer(eos_token_id=1, stop_tokens=[9])
    assert generator._resolve_stop_tokens(tokenizer, "checkpoint") == [1, 9]


def test_resolve_stop_tokens_ignores_unknown_named_special_token(monkeypatch):
    monkeypatch.setattr(
        generator.GenerationConfig,
        "from_pretrained",
        lambda model_path: SimpleNamespace(eos_token_id=None),
    )
    tokenizer = FakeTokenizer(
        eos_token_id=None,
        special_tokens_map={"eot_token": "<not-in-vocab>"},
        vocab={"<unk>": 0},
        unknown_conversion=0,
    )
    assert generator._resolve_stop_tokens(tokenizer, "checkpoint") == []
