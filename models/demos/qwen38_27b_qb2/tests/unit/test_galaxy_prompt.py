# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Catch tokenizer-contract failures before a full Galaxy loads eight models."""

import json
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer, BatchEncoding

from models.demos.qwen38_27b_qb2.tests.galaxy_prompt import qualification_prompt


class Tokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "formatted prompt"

    def __call__(self, text, *, add_special_tokens):
        assert text == "formatted prompt" and not add_special_tokens
        return BatchEncoding(dict(input_ids=self.tokens))


def test_batch_encoding_is_unwrapped_before_generation_and_receipt():
    tokens = qualification_prompt(Tokenizer([1, 2, 3]))
    assert json.loads(json.dumps(dict(prompt_tokens=tokens))) == dict(prompt_tokens=[1, 2, 3])


@pytest.mark.parametrize("tokens", ([], [[1, 2]], ["input_ids"], [-1]))
def test_invalid_prompt_rejected_before_device_work(tokens, expect_error):
    with expect_error(ValueError, "flat list"):
        qualification_prompt(Tokenizer(tokens))


@pytest.mark.skipif(not os.getenv("MODEL_WEIGHTS_DIR"), reason="requires the pinned local tokenizer")
def test_real_tokenizer_matches_passing_single_replica_prompt():
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_WEIGHTS_DIR"], local_files_only=True)
    baseline = Path(__file__).resolve().parents[2] / "galaxy-evidence/baseline-v2/full-model.json"
    assert qualification_prompt(tokenizer) == json.loads(baseline.read_text())["prompt_tokens"]
