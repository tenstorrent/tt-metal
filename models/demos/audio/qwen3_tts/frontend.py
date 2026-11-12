# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host front-end for Qwen3-TTS: text to token ids, and the scaffolding around them.

Nothing here runs on device, and nothing here is going to: the tokenizer is a byte-level
BPE, a string-to-integers map with no tensor math to accelerate. It is also the whole of
the text path. Upstream applies no cleaners, no grapheme-to-phoneme stage and no
romanisation, so all ten languages go raw into the same vocabulary, CJK included.

What this module adds over calling transformers directly:

  * the three prompt wrappers upstream builds before tokenising
    (`qwen_tts/inference/qwen3_tts_model.py:269-276`),
  * the special ids read from config.json rather than hardcoded, so a checkpoint that
    renumbers them fails loudly,
  * language names resolved to codec-vocabulary ids. Language never enters the text
    stream: `talker_config.codec_language_id` maps the ten names onto ids 2050 to 2071,
    which ride the codec stream instead.

One deviation from upstream: an unsupported language raises `ValueError` naming the
supported set, where `Qwen3TTSForConditionalGeneration.generate` raises
`NotImplementedError`. A caller's typo is a bad argument, not a missing feature.
"""

import functools

from transformers import Qwen2Tokenizer

from models.demos.audio.qwen3_tts import weights

# config.json fields that carry a token id, keyed by the name used here.
SPECIAL_TOKEN_FIELDS = {
    "im_start": "im_start_token_id",
    "im_end": "im_end_token_id",
    "assistant": "assistant_token_id",
    "tts_bos": "tts_bos_token_id",
    "tts_eos": "tts_eos_token_id",
    "tts_pad": "tts_pad_token_id",
}

# `language="Auto"` leaves the tag off and lets the talker infer from the text.
AUTO_LANGUAGE = "auto"


@functools.lru_cache(maxsize=None)
def tokenizer(checkpoint=None):
    """The checkpoint's byte-level BPE.

    Named explicitly rather than through `AutoTokenizer`, which warns about the
    `qwen3_tts` model type it does not recognise. The repository ships `vocab.json` and
    `merges.txt` but no `tokenizer.json`, so transformers converts on load; that cost is
    paid once per process.
    """
    return Qwen2Tokenizer.from_pretrained(checkpoint or weights.checkpoint_dir())


@functools.lru_cache(maxsize=None)
def special_tokens():
    """Token ids the prompt is built from, read from config.json."""
    config = weights.model_config()
    return {name: config[field] for name, field in SPECIAL_TOKEN_FIELDS.items()}


@functools.lru_cache(maxsize=None)
def language_ids():
    """Language name (lower case) -> codec-vocabulary id."""
    return dict(weights.model_config()["talker_config"]["codec_language_id"])


def supported_languages():
    """The language names this checkpoint accepts, sorted."""
    return tuple(sorted(language_ids()))


def language_id(language):
    """Resolve a language name to its codec id. `Auto` resolves to None, meaning no tag."""
    if language is None:
        return None
    name = str(language).strip().lower()
    if name == AUTO_LANGUAGE:
        return None
    ids = language_ids()
    if name not in ids:
        raise ValueError(
            f"unsupported language {language!r}; this checkpoint speaks {', '.join(supported_languages())}"
        )
    return ids[name]


# ── prompt wrappers ─────────────────────────────────────────────────────────


ROLE_PREFIX = "<|im_start|>assistant\n"


def assistant_prompt(text):
    """The text to speak, followed by an open assistant turn for the model to continue."""
    return f"{ROLE_PREFIX}{text}<|im_end|>\n<|im_start|>assistant\n"


def role_ids():
    """The three ids every prompt opens with, without tokenising any text."""
    return encode(ROLE_PREFIX)


def reference_prompt(text):
    """The transcript of a reference clip: a closed turn, since the model is not continuing it."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def instruction_prompt(instruction):
    """A VoiceDesign instruction, spoken by the user rather than the assistant."""
    return f"<|im_start|>user\n{instruction}<|im_end|>\n"


# ── encoding ────────────────────────────────────────────────────────────────


def encode(text):
    """Raw ids for a string, with no prompt scaffolding."""
    return tokenizer()(text)["input_ids"]


def text_ids(text):
    """Ids for the text the model should speak."""
    return encode(assistant_prompt(text))


def reference_text_ids(text):
    """Ids for a reference clip's transcript."""
    return encode(reference_prompt(text))


def instruction_ids(instruction):
    """Ids for a VoiceDesign instruction."""
    return encode(instruction_prompt(instruction))
