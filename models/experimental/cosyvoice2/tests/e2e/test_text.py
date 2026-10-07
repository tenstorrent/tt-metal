# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt/text.py against upstream's own text path. Host only; no device.

1. `GOLDEN`: upstream's `CosyVoiceFrontEnd.text_normalize` (the real method, run in the reference venv with no
   ttsfrd/wetext, as that venv is built) and `CosyVoice2Tokenizer.encode` on hand-picked inputs covering number
   spelling, a two-segment split with a short tail merged back, punctuation-only pieces, a closing quote after a
   stop, the `<|...|>` bypass, surrounding whitespace with no final stop, and `split=False` (the prompt
   transcript). Generated 2026-09-27 at upstream 074ca6dc9e80, transformers 5.12.1, inflect 7.5.0.
2. Every corpus case `scripts/prepare_inputs.py` wrote (`COSYVOICE2_INPUTS`; skipped when unset): the segments,
   their token ids, and the prompt transcript's normalized text and ids, as upstream produced them.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pytest

from models.experimental.cosyvoice2.tt.text import TextFrontend, is_only_punctuation, split_paragraph

INPUTS_DIR = os.environ.get("COSYVOICE2_INPUTS", "")

# (name, text, split, upstream's text_normalize result, upstream's token ids per segment)
GOLDEN = [
    (
        "plain",
        "Truly this sea is of infinite width.",
        True,
        ["Truly this sea is of infinite width."],
        [[1282, 3901, 419, 9396, 374, 315, 23809, 2374, 13]],
    ),
    (
        "numbers",
        "In 1987, 3 of the 25 boats sailed 100 miles by 7:45 and 2.5 hours later.",
        True,
        [
            "In one thousand, nine hundred and eighty-seven, three of the twenty-five boats sailed one hundred miles by seven:forty-five and two.five hours later."
        ],
        [
            [
                641,
                825,
                16183,
                11,
                11627,
                7739,
                323,
                79579,
                78025,
                11,
                2326,
                315,
                279,
                17073,
                35299,
                31631,
                75744,
                825,
                7739,
                8756,
                553,
                8094,
                25,
                3969,
                88,
                35299,
                323,
                1378,
                833,
                533,
                4115,
                2937,
                13,
            ]
        ],
    ),
    (
        "long_split_merge",
        "It was the white rabbit returning splendidly dressed, with a pair of white kid gloves in one hand and a large fan in the other. He came trotting along in a great hurry, muttering to himself as he came. Oh the duchess, the duchess! Oh won't she be savage if I've kept her waiting! Alice felt so desperate that she was ready to ask help of any one; so, when the rabbit came near her, she began, in a low, timid voice. If you please, sir. The rabbit started violently. Then he dropped the white kid gloves and the fan, and skurried away into the darkness as hard as he could go. Short end.",
        True,
        [
            "It was the white rabbit returning splendidly dressed, with a pair of white kid gloves in one hand and a large fan in the other. He came trotting along in a great hurry, muttering to himself as he came. Oh the duchess, the duchess! Oh won't she be savage if I've kept her waiting!",
            " Alice felt so desperate that she was ready to ask help of any one; so, when the rabbit came near her, she began, in a low, timid voice. If you please, sir. The rabbit started violently. Then he dropped the white kid gloves and the fan, and skurried away into the darkness as hard as he could go. Short end.",
        ],
        [
            [
                2132,
                572,
                279,
                4158,
                38724,
                13451,
                69860,
                398,
                25365,
                11,
                448,
                264,
                6716,
                315,
                4158,
                10369,
                35416,
                304,
                825,
                1424,
                323,
                264,
                3460,
                8405,
                304,
                279,
                1008,
                13,
                1260,
                3697,
                56577,
                1280,
                3156,
                304,
                264,
                2244,
                47235,
                11,
                5206,
                59385,
                311,
                5561,
                438,
                566,
                3697,
                13,
                8670,
                279,
                294,
                1387,
                433,
                11,
                279,
                294,
                1387,
                433,
                0,
                8670,
                2765,
                944,
                1340,
                387,
                72035,
                421,
                358,
                3003,
                8604,
                1059,
                8580,
                0,
            ],
            [
                29405,
                6476,
                773,
                27395,
                429,
                1340,
                572,
                5527,
                311,
                2548,
                1492,
                315,
                894,
                825,
                26,
                773,
                11,
                979,
                279,
                38724,
                3697,
                3143,
                1059,
                11,
                1340,
                6009,
                11,
                304,
                264,
                3347,
                11,
                98049,
                7743,
                13,
                1416,
                498,
                4486,
                11,
                27048,
                13,
                576,
                38724,
                3855,
                64200,
                13,
                5005,
                566,
                12226,
                279,
                4158,
                10369,
                35416,
                323,
                279,
                8405,
                11,
                323,
                1901,
                324,
                4487,
                3123,
                1119,
                279,
                26298,
                438,
                2588,
                438,
                566,
                1410,
                728,
                13,
                10698,
                835,
                13,
            ],
        ],
    ),
    (
        "punctuation_only_dropped",
        "Wait... what?! ... Fine.",
        True,
        ["Wait. what? . Fine."],
        [[14190, 13, 1128, 30, 659, 30153, 13]],
    ),
    (
        "quote_after_stop",
        'He said "stop." Then he left!',
        True,
        ['He said "stop." Then he left!'],
        [[1519, 1053, 330, 9495, 1189, 5005, 566, 2115, 0]],
    ),
    (
        "marker_bypass",
        "<|en|>Hello there 42.",
        True,
        ["<|en|>Hello there 42."],
        [[27, 91, 268, 91, 29, 9707, 1052, 220, 19, 17, 13]],
    ),
    (
        "whitespace_no_final_stop",
        "   no final punctuation here 7   ",
        True,
        ["no final punctuation here seven."],
        [[2152, 1590, 61503, 1588, 8094, 13]],
    ),
    (
        "prompt_unsplit",
        "I have 2 cats; they are 11 years old.",
        False,
        "I have two cats; they are eleven years old.",
        [[40, 614, 1378, 19423, 26, 807, 525, 44214, 1635, 2310, 13]],
    ),
]


@pytest.fixture(scope="module")
def frontend():
    return TextFrontend()


@pytest.mark.parametrize("name,text,split,expected,ids", GOLDEN, ids=[g[0] for g in GOLDEN])
def test_text_normalize_matches_upstream(frontend, name, text, split, expected, ids):
    got = frontend.normalize(text, split=split)
    assert got == expected
    segments = got if split else [got]
    assert [frontend.encode(s) for s in segments] == ids


def test_split_paragraph_is_upstreams_verbatim():
    """Beyond the goldens: a quote closing a sentence stays with it, and a short tail merges into the segment
    before it (lang="en", measured with a whitespace tokenizer so the arithmetic is visible)."""
    words = str.split
    assert split_paragraph('A b c. "D e." F', words, "en", token_max_n=4, token_min_n=2, merge_len=2) == [
        "A b c.",
        ' "D e." F.',
    ]
    assert split_paragraph("One two three. Four five six. Seven.", words, "en", 4, 2, 2) == [
        "One two three.",
        " Four five six. Seven.",
    ]
    assert is_only_punctuation("...") and is_only_punctuation("") and not is_only_punctuation("a.")


def test_chinese_text_is_refused(frontend, expect_error):
    with expect_error(NotImplementedError, "Chinese text path"):
        frontend.normalize("希望你以后能够做的比我还好呦。")


def _inputs():
    return sorted(glob.glob(os.path.join(INPUTS_DIR, "*.npz"))) if INPUTS_DIR else []


@pytest.mark.skipif(not _inputs(), reason="set COSYVOICE2_INPUTS to scripts/prepare_inputs.py's --out-dir")
def test_corpus_text_matches_prepare_inputs(frontend):
    for path in _inputs():
        with np.load(path) as d:
            case = json.loads(str(d["case_json"]))
            segments = json.loads(str(d["segments_json"]))
            segment_ids = json.loads(str(d["segment_text_ids_json"]))
            prompt_norm = str(d["prompt_text_normalized"])
            prompt_ids = d["prompt_text_ids"].reshape(-1).tolist()
        assert frontend.normalize(case["text"]) == segments, case["case_id"]
        assert [frontend.encode(s) for s in segments] == segment_ids, case["case_id"]
        assert frontend.encode(prompt_norm) == prompt_ids, case["case_id"]
        if case["lang"] == "en" and not any("\u4e00" <= ch <= "\u9fff" for ch in case["prompt_text"]):
            assert frontend.normalize(case["prompt_text"], split=False) == prompt_norm, case["case_id"]


def test_checkpoint_revision_is_pinned_and_shared_with_the_reference_side():
    """Both sides read the checkpoint at one pinned revision (a full commit hash, never a branch), so neither a new
    upload to the Hub repo nor a drift between the two constants can change results unnoticed."""
    import importlib.util

    from models.experimental.cosyvoice2.tt.text import MODEL_REVISION

    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "reference_env.py")
    spec = importlib.util.spec_from_file_location("cosyvoice2_reference_env", path)
    reference_env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_env)  # stdlib imports only at module level
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION), MODEL_REVISION
    assert reference_env.MODEL_REVISION == MODEL_REVISION
