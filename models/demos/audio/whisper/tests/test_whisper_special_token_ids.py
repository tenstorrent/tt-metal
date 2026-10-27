# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Whisper special-token ID resolution across model variants.

large-v2 and large-v3 / large-v3-turbo do NOT share the same special-token IDs:
v3 added the ``<|yue|>`` (Cantonese) language token, shifting every special token
after the language block (``<|startofprev|>``, ``<|nospeech|>``, ``<|notimestamps|>``
and the timestamp range) up by one relative to v2. These tests pin the resolved IDs
so the generator stays correct as variants are added.

Host-only: downloads tiny config/tokenizer files, no model weights, no device.
"""

import pytest
from transformers import AutoConfig, AutoProcessor, GenerationConfig

from models.demos.audio.whisper.tt.whisper_generator import resolve_special_token_ids

# fmt: off
EXPECTED = {
    "openai/whisper-large-v3": dict(
        eos=50257, startoftranscript=50258, startofprev=50362,
        no_speech=50363, notimestamps=50364, timestamp_begin=50365, timestamp_end=51864,
    ),
    "openai/whisper-large-v3-turbo": dict(
        eos=50257, startoftranscript=50258, startofprev=50362,
        no_speech=50363, notimestamps=50364, timestamp_begin=50365, timestamp_end=51864,
    ),
    "openai/whisper-large-v2": dict(
        eos=50257, startoftranscript=50258, startofprev=50361,
        no_speech=50362, notimestamps=50363, timestamp_begin=50364, timestamp_end=51863,
    ),
}
# fmt: on


@pytest.mark.parametrize("model_repo", list(EXPECTED.keys()))
def test_resolve_special_token_ids(model_repo):
    cfg = AutoConfig.from_pretrained(model_repo)
    gen = GenerationConfig.from_pretrained(model_repo)
    proc = AutoProcessor.from_pretrained(model_repo)

    ids = resolve_special_token_ids(cfg, gen, proc.tokenizer)

    expected = EXPECTED[model_repo]
    actual = {field: getattr(ids, field) for field in expected}
    assert actual == expected, f"{model_repo}: {actual} != {expected}"
