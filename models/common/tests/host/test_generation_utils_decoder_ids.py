# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the shared encoder-decoder generation harness.

`models.common.generation_utils.run_generate` must hand the logits processors the decoder prefix (the tokens
generated so far, starting from the decoder start token), the same tensor HF `generate` feeds them, and keep the
padded encoder ids for the encoder-side processors only. Before this change every processor saw the padded
encoder ids, so `repetition_penalty` was computed from the source sentence and `begin_suppress_tokens` was
resolved against the padded encoder width, which made it fire at every step instead of only at the first decoder
position. The step logits were also the whole fixed decoder window rather than the current row, so the
processors only ever reached its last row.

Every case puts the field on the model config, so the stock code path reads the same setting and the tests
compare decoded token ids. The repetition cases probe the row behind the final decoded token (index STEPS), which
is the one row the stock 3-D call could reach.

No device and no download: a scripted model returns fixed logits and the tokenizer is a stub.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import T5Config
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import SuppressTokensAtBeginLogitsProcessor

from models.common.generation_utils import get_logits_processor, run_generate

VOCAB = 16
EOS = 99
STEPS = 64


class _ScriptedModel:
    """Returns fixed logits for the whole decoder window; the harness reads one row per step."""

    def __init__(self, config, generation_config, logits):
        self.config = config
        self.generation_config = generation_config
        self.logits = logits

    def __call__(self, input_ids=None, decoder_input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.logits)


class _StubTokenizer:
    def __init__(self, input_ids):
        self.input_ids = torch.tensor([input_ids])

    def __call__(self, text, return_tensors="pt"):
        return SimpleNamespace(input_ids=self.input_ids, attention_mask=torch.ones_like(self.input_ids))

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(token)) for token in ids)


def _logits(default_rows, probe_row=None):
    logits = torch.full((1, STEPS, VOCAB), -5.0)
    for step in range(STEPS):
        for token, score in default_rows.items():
            logits[0, step, token] = score
    if probe_row is not None:
        logits[0, STEPS - 1, :] = -5.0
        for token, score in probe_row.items():
            logits[0, STEPS - 1, token] = score
    return logits


def _config(**fields):
    config = T5Config()
    for name, value in fields.items():
        setattr(config, name, value)
    return config


def _generate(config, logits, source=(11, 11, 11, 5)):
    model = _ScriptedModel(config, GenerationConfig.from_model_config(config), logits)
    decoded = run_generate(
        "source",
        _StubTokenizer(source),
        lambda device: (None, model),
        device=None,
        run_tt_model=False,
        log=False,
    )
    return [int(token) for token in decoded.split(",")]


def test_begin_suppress_index_comes_from_the_decoder_prefix():
    """`begin_index` is a decoder-prefix length, never the padded encoder width."""
    config = _config(begin_suppress_tokens=[5], pad_token_id=0, eos_token_id=EOS)
    generation_config = GenerationConfig.from_model_config(config)
    encoder_ids = torch.zeros(1, 32, dtype=torch.long)
    decoder_prefix = torch.zeros(1, 1, dtype=torch.long)

    processors = get_logits_processor(decoder_prefix, generation_config, encoder_input_ids=encoder_ids)
    begin_suppress = [
        processor for processor in processors if isinstance(processor, SuppressTokensAtBeginLogitsProcessor)
    ]
    assert len(begin_suppress) == 1
    assert begin_suppress[0].begin_index == decoder_prefix.shape[-1]


def test_begin_suppress_stops_after_the_first_decoder_position():
    """`begin_suppress_tokens` fires once; the stock loop suppressed at every step."""
    logits = _logits({5: 6.0, 3: 1.0})
    tokens = _generate(_config(begin_suppress_tokens=[5]), logits)
    assert tokens[1] == 3, "the begin position must suppress token 5"
    assert tokens[2] == 5, "later steps must not suppress token 5 any more"


def test_repetition_penalty_uses_the_decoder_prefix_not_the_source():
    """A token that repeats only in the source must not be penalised."""
    # Token 7 is generated at every step, so it is in the decoder prefix; token 11 is not.
    logits = _logits({7: 6.0, 3: 4.0}, probe_row={7: 6.0, 11: 5.9})
    tokens = _generate(_config(repetition_penalty=2.0), logits)
    assert tokens[1] == 7
    assert tokens[STEPS] == 11, "token 7 is in the decoder prefix, so the penalty must demote it below token 11"


def test_repetition_penalty_ignores_a_token_that_only_repeats_in_the_source():
    """The same penalty must leave a source-only token alone."""
    logits = _logits({7: 6.0, 3: 4.0}, probe_row={11: 6.0, 3: 5.9})
    tokens = _generate(_config(repetition_penalty=2.0), logits)
    assert tokens[STEPS] == 11, "token 11 repeats in the source only, so it must win"


def test_no_repeat_ngram_reads_the_decoder_prefix():
    """The n-gram constraint is computed from the generated prefix, not from the source sentence."""
    config = _config(no_repeat_ngram_size=2)
    generation_config = GenerationConfig.from_model_config(config)
    encoder_ids = torch.tensor([[3, 5, 3, 5]])
    decoder_prefix = torch.tensor([[0, 3, 5, 3]])

    processors = get_logits_processor(decoder_prefix, generation_config, encoder_input_ids=encoder_ids)
    scores = torch.full((1, VOCAB), -5.0)
    scores[0, 5] = 6.0  # closes the 2-gram (3, 5) that the decoder prefix already contains
    scores[0, 7] = 5.9
    processed = processors(decoder_prefix, scores)
    assert torch.argmax(processed, dim=-1).item() == 7


def test_default_t5_config_keeps_greedy_unchanged():
    """The stock T5 config parses to no active processor, so the harness stays plain greedy argmax."""
    logits = _logits({3: 6.0, 5: 1.0})
    tokens = _generate(_config(), logits)
    assert tokens[:3] == [0, 3, 3], tokens
