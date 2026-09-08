# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 of 3: turn text into model inputs.

    list[str] of B texts
      -> apply_prompt  -> list[str] of B prefixed texts
      -> tokenize      -> input_ids      (B, S) int64
                          attention_mask (B, S) int64, 1 = real token, 0 = padding

S is the longest tokenized sequence in the batch, capped at MAX_SEQ_LENGTH. Shorter rows are
right-padded with pad_token_id (1), and attention_mask is what marks that padding for every
later stage.

Host-side string work only, no tensors until the tokenizer runs. The prefix is applied to the
text before tokenizing, so it becomes ordinary tokens the encoder attends to rather than a
special embedding.
"""

from __future__ import annotations

from typing import Optional, Sequence

# Verbatim from config_sentence_transformers.json. The trailing space is part of the prefix.
PROMPTS: dict[str, str] = {
    "query": "search_query: ",
    "passage": "search_document: ",
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    "Speed": "search_document: ",
}

# From sentence_bert_config.json.
MAX_SEQ_LENGTH = 512


def apply_prompt(texts: Sequence[str], prompt_name: Optional[str]) -> list[str]:
    """Prepend the task prefix to every text.

    The model was trained with these prefixes, so dropping one measurably moves the embedding.
    The prefix is prepended to the text rather than injected as a token, which is why it shows
    up as ordinary tokens after tokenization.

    Args:
        texts: The input strings, length B.
        prompt_name: A key of PROMPTS ("query", "passage", ...), or None to pass the text
            through unchanged.

    Returns:
        list[str]: Length B, each entry the prefix concatenated with the original text, or the
        original text when prompt_name is None.

    Raises:
        KeyError: If prompt_name is not None and not a key of PROMPTS.
    """
    if prompt_name is None:
        return list(texts)
    if prompt_name not in PROMPTS:
        raise KeyError(f"unknown prompt {prompt_name!r}; known: {sorted(PROMPTS)}")
    return [PROMPTS[prompt_name] + text for text in texts]


def tokenize(tokenizer, texts: Sequence[str], max_length: int = MAX_SEQ_LENGTH):
    """Tokenize a batch of texts into padded tensors.

    padding=True pads to the batch maximum rather than to max_length, so S varies with the
    input rather than being fixed at 512.

    Args:
        tokenizer: The XLMRobertaTokenizerFast loaded from the checkpoint.
        texts: The input strings, length B, already prefixed by apply_prompt.
        max_length: Truncation limit, defaulting to MAX_SEQ_LENGTH (512).

    Returns:
        BatchEncoding: With input_ids (B, S) int64, bos=0 ... eos=2 and right-padded with
        pad=1, and attention_mask (B, S) int64 holding 1 for real tokens and 0 for padding.
    """
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
