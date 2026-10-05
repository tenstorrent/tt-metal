# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CosyVoice2's text frontend, English path, on the host in python_env: normalize, split, tokenize.

Upstream (FunAudioLLM/CosyVoice @ 074ca6dc9e80, `CosyVoiceFrontEnd.text_normalize` in cosyvoice/cli/frontend.py,
and cosyvoice/utils/frontend_utils.py), with neither ttsfrd nor wetext installed (`text_frontend == ''`, the
configuration scripts/requirements-reference.txt builds), does this to text without CJK characters:

  1. return the text untouched (unsplit) if it holds a `<|...|>` marker, or is empty;
  2. strip;
  3. `spell_out_number` (inflect);
  4. `split_paragraph(text, tokenize, "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)`;
  5. drop punctuation-only segments.

`split=False` (how upstream treats the prompt transcript) returns the text after step 3, unsplit.

`contains_chinese`, `spell_out_number`, `split_paragraph` and `is_only_punctuation` are upstream's own, copied
verbatim apart from formatting (Apache-2.0, Copyright (c) 2024 Alibaba Inc, authors Xiang Lyu and Zhihao Du), so
that segment boundaries -- which set each LLM call's min/max token budget -- match upstream's exactly.
`SPECIAL_TOKENS` and `TextFrontend.encode` mirror `CosyVoice2Tokenizer` (cosyvoice/tokenizer/tokenizer.py).

Text with CJK characters takes upstream's Chinese path (a different normalizer and splitter). It is not ported,
and `TextFrontend.normalize` raises on it rather than silently treating it as English.
"""

from __future__ import annotations

import os
import re

import regex

MODEL_REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"
# The checkpoint revision every figure in docs/VALIDATION.md was measured with. Every Hub download of the model
# (weights, tokenizer, config) asks for it, so a new upload to the repo can't change results.
# scripts/reference_env.py pins the same revision for the reference side.
MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
TOKENIZER_SUBDIR = "CosyVoice-BlankEN"
TOKENIZER_FILES = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt")

# upstream's split_paragraph arguments for both languages (frontend.py, text_normalize)
TOKEN_MAX_N, TOKEN_MIN_N, MERGE_LEN = 80, 60, 20

# CosyVoice2Tokenizer's additions to the BlankEN tokenizer, verbatim. The upstream comment: "non-chat model, all
# these special tokens keep randomly initialized."
SPECIAL_TOKENS = {
    "eos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>",
    "additional_special_tokens": [
        "<|im_start|>",
        "<|im_end|>",
        "<|endofprompt|>",
        "[breath]",
        "<strong>",
        "</strong>",
        "[noise]",
        "[laughter]",
        "[cough]",
        "[clucking]",
        "[accent]",
        "[quick_breath]",
        "<laughter>",
        "</laughter>",
        "[hissing]",
        "[sigh]",
        "[vocalized-noise]",
        "[lipsmack]",
        "[mn]",
    ],
}

# ---------------------------------------------------------------------------------------------------------------
# Upstream cosyvoice/utils/frontend_utils.py, verbatim.
# ---------------------------------------------------------------------------------------------------------------
chinese_char_pattern = re.compile(r"[一-鿿]+")


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    def calc_utt_length(_text: str):
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r"^[\p{P}\p{S}]*$"
    return bool(regex.fullmatch(punctuation_pattern, text))


# ---------------------------------------------------------------------------------------------------------------


def tokenizer_dir(repo_id: str = MODEL_REPO_ID, revision: str = MODEL_REVISION) -> str:
    """The checkpoint's `CosyVoice-BlankEN` directory, holding only the tokenizer files (from the Hugging Face
    cache, downloaded on first use). The 1 GB `model.safetensors` beside them is not needed: CosyVoice2's LLM
    weights come from `llm.pt`."""
    from huggingface_hub import hf_hub_download

    paths = [
        hf_hub_download(repo_id=repo_id, filename=f"{TOKENIZER_SUBDIR}/{f}", revision=revision) for f in TOKENIZER_FILES
    ]
    return os.path.dirname(paths[0])


class TextFrontend:
    """Upstream's text path for CosyVoice2 (see the module docstring): `normalize` and `encode`."""

    def __init__(self, tokenizer_path: str | None = None):
        import inflect
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or tokenizer_dir())
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        self.inflect_parser = inflect.engine()

    def encode(self, text: str) -> list[int]:
        """`CosyVoice2Tokenizer.encode`: the text's Qwen2 token ids, no BOS/EOS added."""
        return self.tokenizer([text], return_tensors="pt")["input_ids"][0].tolist()

    def normalize(self, text: str, split: bool = True):
        """`text_normalize(text, split)` with no ttsfrd/wetext: a list of segments if `split`, else one string."""
        if ("<|" in text and "|>" in text) or text == "":
            return [text] if split else text
        text = text.strip()
        if contains_chinese(text):
            raise NotImplementedError("upstream's Chinese text path (contains_chinese) is not ported; see tt/text.py")
        text = spell_out_number(text, self.inflect_parser)
        texts = split_paragraph(
            text,
            self.encode,
            "en",
            token_max_n=TOKEN_MAX_N,
            token_min_n=TOKEN_MIN_N,
            merge_len=MERGE_LEN,
            comma_split=False,
        )
        texts = [t for t in texts if not is_only_punctuation(t)]
        return texts if split else text
