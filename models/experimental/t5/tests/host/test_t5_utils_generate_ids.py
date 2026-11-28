# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the T5 copy of the shared generation loop.

`models.experimental.t5.t5_utils.run_generate` duplicates `models.common.generation_utils.run_generate` and had
the same defect: it passed the padded encoder ids to every logits processor and handed them the whole fixed
decoder window instead of the current step row, so the processors only ever reached the last row of that window
and computed `repetition_penalty` from the source sentence.

Every case puts the setting on the model config, so the stock code path reads the same value, and each scripted
logit window is built so the stock and fixed argmax differ.

No device and no download: a scripted TT model returns fixed logits and the tokenizer is a stub.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import T5Config
from transformers.generation.configuration_utils import GenerationConfig

from models.experimental.t5.t5_utils import run_generate

VOCAB = 16
EOS = 99
STEPS = 64


class _ScriptedTtModel:
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, input_ids=None, decoder_input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.logits, encoder_outputs=None)


class _StubTokenizer:
    def __init__(self, input_ids):
        self.input_ids = torch.tensor([input_ids])

    def __call__(self, text, return_tensors="pt"):
        return SimpleNamespace(input_ids=self.input_ids, attention_mask=torch.ones_like(self.input_ids))

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(token)) for token in ids)


def _logits(default_rows, last_row=None):
    logits = torch.full((1, STEPS, VOCAB), -5.0)
    for step in range(STEPS):
        for token, score in default_rows.items():
            logits[0, step, token] = score
    if last_row is not None:
        logits[0, STEPS - 1, :] = -5.0
        for token, score in last_row.items():
            logits[0, STEPS - 1, token] = score
    return logits


def _config(**fields):
    config = T5Config()
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _generate(config, logits, source=(11, 11, 11, 7)):
    hf_reference_model = SimpleNamespace(config=config, generation_config=GenerationConfig.from_model_config(config))
    decoded = run_generate(
        "source",
        _StubTokenizer(source),
        hf_reference_model,
        _ScriptedTtModel(logits),
        device=None,
        log=False,
    )
    return [int(token) for token in decoded.split(",")]


def test_begin_suppress_fires_once_at_the_first_decoder_position():
    logits = _logits({5: 6.0, 3: 1.0})
    tokens = _generate(_config(begin_suppress_tokens=[5]), logits)
    assert tokens[1] == 3, "the begin position must suppress token 5"
    assert tokens[2] == 5, "later steps must not suppress token 5 any more"


def test_repetition_penalty_does_not_read_the_source():
    """Token 11 repeats three times in the source, so the stock loop penalises it; it never gets generated."""
    logits = _logits({7: 6.0, 3: 2.0}, last_row={11: 5.0, 3: 4.0})
    tokens = _generate(_config(repetition_penalty=2.0), logits, source=(11, 11, 11, 7))
    assert tokens[1] == 7
    assert tokens[STEPS] == 11, "token 11 is not in the decoder prefix, so it must not be penalised"


def test_repetition_penalty_applies_to_a_generated_token():
    """Token 7 fills the decoder prefix, so the penalty must demote it below token 11."""
    logits = _logits({7: 6.0, 3: 2.0}, last_row={7: 5.0, 11: 4.0})
    tokens = _generate(_config(repetition_penalty=2.0), logits)
    assert tokens[1] == 7
    assert tokens[STEPS] == 11, "token 7 is in the decoder prefix, so it is penalised to 2.5"


def test_settings_come_from_the_generation_config():
    """A value that only lives in generation_config.json must be honoured."""
    config = _config()
    generation_config = GenerationConfig.from_model_config(config)
    generation_config.repetition_penalty = 2.0
    logits = _logits({7: 6.0, 3: 2.0}, last_row={7: 5.0, 11: 4.0})

    hf_reference_model = SimpleNamespace(config=config, generation_config=generation_config)
    decoded = run_generate(
        "source",
        _StubTokenizer((11, 11, 11, 7)),
        hf_reference_model,
        _ScriptedTtModel(logits),
        device=None,
        log=False,
    )
    tokens = [int(token) for token in decoded.split(",")]
    assert tokens[STEPS] == 11


def test_default_t5_config_keeps_greedy_unchanged():
    logits = _logits({3: 6.0, 5: 1.0})
    tokens = _generate(_config(), logits)
    assert tokens[:3] == [0, 3, 3], tokens
