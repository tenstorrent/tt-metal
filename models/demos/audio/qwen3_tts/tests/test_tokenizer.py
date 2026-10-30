# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The host text front-end: tokenizer, prompt scaffolding and language resolution.

No device. The tokenizer is a byte-level BPE that this repository uses as it ships, so
what is worth testing is not the BPE itself but the seams around it: that the special ids
in config.json still name the tokens they should, that the prompt wrappers produce the
structure the model was trained on, and that ids for a fixed corpus have not moved.

Ids below were generated from the pinned revision. They are deliberately literal: a
checkpoint that renumbers its vocabulary should fail here, loudly, rather than produce
quietly wrong speech.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/test_tokenizer.py
"""

import pytest

from models.demos.audio.qwen3_tts import frontend, weights

# One phrase per supported language, with the ids the pinned checkpoint assigns.
# fmt: off  (one id per line is unreadable for a table like this)
CORPUS = {
    "chinese": ("你好，欢迎使用语音合成。", [108386, 3837, 100437, 37029, 105761, 106726, 1773]),
    "english": ("Hello, and welcome to speech synthesis.", [9707, 11, 323, 10565, 311, 8806, 38875, 13]),
    "french": (
        "Bonjour, bienvenue dans la synthèse vocale.",
        [81581, 11, 14370, 7140, 6866, 1187, 42898, 4458, 325, 11984, 1574, 13],
    ),
    "german": (
        "Hallo, willkommen bei der Sprachsynthese.",
        [78078, 11, 686, 42789, 13279, 2694, 15515, 610, 20339, 43910, 13],
    ),
    "italian": (
        "Ciao, benvenuto nella sintesi vocale.",
        [34, 22516, 11, 3318, 1037, 1535, 35922, 42229, 33083, 11984, 1574, 13],
    ),
    "japanese": ("こんにちは、音声合成へようこそ。", [89015, 5373, 78685, 70074, 106726, 126263, 124038, 131741, 1773]),
    "korean": (
        "안녕하세요, 음성 합성에 오신 것을 환영합니다.",
        [
            126246,
            144370,
            91145,
            11,
            16751,
            234,
            32831,
            20136,
            102,
            32831,
            19391,
            73077,
            82528,
            129337,
            46832,
            246,
            125144,
            60838,
            13,
        ],
    ),
    "portuguese": (
        "Olá, bem-vindo à síntese de voz.",
        [42719, 1953, 11, 31915, 8273, 34999, 3784, 44715, 406, 2367, 409, 84090, 13],
    ),
    "russian": (
        "Здравствуйте, добро пожаловать в синтез речи.",
        [
            35451,
            6949,
            26988,
            20200,
            82580,
            50527,
            11,
            140445,
            5063,
            21259,
            15952,
            126915,
            5805,
            5409,
            18943,
            1792,
            31885,
            18108,
            55757,
            1802,
            13,
        ],
    ),
    "spanish": (
        "Hola, bienvenido a la síntesis de voz.",
        [68012, 11, 14370, 1037, 5249, 264, 1187, 44715, 406, 13774, 409, 84090, 13],
    ),
}
# fmt: on

# The token each special id must resolve to. Ids live in config.json; names live in the
# vocabulary, and the two drifting apart is exactly the failure this catches.
SPECIAL_TOKEN_NAMES = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "assistant": "assistant",
    "tts_bos": "<tts_text_bos>",
    "tts_eos": "<tts_text_eod>",
    "tts_pad": "<tts_pad>",
}

# "Hello from Tenstorrent." wrapped for the model to continue.
ASSISTANT_PROMPT_IDS = [151644, 77091, 198, 9707, 504, 17695, 47365, 7976, 13, 151645, 198, 151644, 77091, 198]


def test_the_documented_ten_languages_are_all_supported():
    """The ten the card names must all resolve.

    Asserted as a subset rather than equality: the CustomVoice checkpoint adds
    `beijing_dialect` and `sichuan_dialect` to `codec_language_id` for its dialect speakers,
    so the exact set depends on which release is loaded.
    """
    supported = set(frontend.supported_languages())
    missing = sorted(set(CORPUS) - supported)
    assert not missing, f"unsupported: {missing}"
    for language in CORPUS:
        assert frontend.language_id(language) is not None


def test_special_ids_name_the_tokens_they_should():
    tokenizer = frontend.tokenizer()
    resolved = {name: tokenizer.convert_ids_to_tokens(token_id) for name, token_id in frontend.special_tokens().items()}
    assert resolved == SPECIAL_TOKEN_NAMES


def test_special_ids_survive_a_round_trip_through_the_vocabulary():
    """convert_tokens_to_ids is the inverse, so config and vocabulary agree both ways."""
    tokenizer = frontend.tokenizer()
    for name, token in SPECIAL_TOKEN_NAMES.items():
        assert tokenizer.convert_tokens_to_ids(token) == frontend.special_tokens()[name], name


@pytest.mark.parametrize("language", sorted(CORPUS))
def test_corpus_ids_are_stable(language):
    text, expected = CORPUS[language]
    assert frontend.encode(text) == expected


@pytest.mark.parametrize("language", sorted(CORPUS))
def test_corpus_round_trips_exactly(language):
    """Byte-level BPE should be lossless, CJK included, with no romanisation anywhere."""
    text, _ = CORPUS[language]
    assert frontend.tokenizer().decode(frontend.encode(text)) == text


def test_assistant_prompt_ids_are_stable():
    assert frontend.text_ids("Hello from Tenstorrent.") == ASSISTANT_PROMPT_IDS


def test_assistant_prompt_leaves_a_turn_open():
    """The model continues the final turn, so the prompt must end having opened one."""
    special = frontend.special_tokens()
    ids = frontend.text_ids("Hello from Tenstorrent.")

    assert ids[0] == special["im_start"]
    assert ids.count(special["im_start"]) == 2, "one turn for the text, one for the model to continue"
    assert ids.count(special["im_end"]) == 1, "the opened turn must stay open"
    assert ids[-2] == special["assistant"], "the open turn is the assistant's"


def test_reference_prompt_closes_its_turn():
    """A reference transcript is stated, not continued."""
    special = frontend.special_tokens()
    ids = frontend.reference_text_ids("This is the reference clip.")

    assert ids.count(special["im_start"]) == 1
    assert ids.count(special["im_end"]) == 1
    assert ids[-1] != special["assistant"]


def test_instruction_prompt_speaks_as_the_user():
    """VoiceDesign describes a voice, which is the user's line, not the assistant's."""
    ids = frontend.instruction_ids("A warm, low voice with a slow delivery.")
    assert frontend.special_tokens()["assistant"] not in ids
    assert frontend.tokenizer().decode(ids).startswith("<|im_start|>user")


def test_language_never_enters_the_text_stream():
    """Text ids are the same whatever the language; the tag rides the codec stream.

    The two vocabularies make this concrete: text ids run to 151k, while every language id
    falls inside the talker's 3072-entry codec vocabulary.
    """
    text = "Hello, and welcome to speech synthesis."
    assert frontend.encode(text) == CORPUS["english"][1]

    codec_vocab = weights.model_config()["talker_config"]["vocab_size"]
    for language in frontend.supported_languages():
        identifier = frontend.language_id(language)
        assert identifier < codec_vocab, f"{language} id {identifier} is outside the codec vocabulary"
        assert identifier not in frontend.encode(text), f"{language} id collides with the text stream"


def test_language_resolution_is_case_insensitive():
    assert frontend.language_id("English") == frontend.language_id("english") == 2050
    assert frontend.language_id("  Chinese  ") == 2055


def test_auto_language_resolves_to_no_tag():
    assert frontend.language_id("Auto") is None
    assert frontend.language_id(None) is None


def test_an_unsupported_language_is_refused(expect_error):
    with expect_error(ValueError, "unsupported language"):
        frontend.language_id("Welsh")
