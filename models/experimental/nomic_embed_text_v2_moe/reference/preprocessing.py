# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 of 3: turn queries into model inputs.

    list[str] of B queries
      -> apply_prompt  -> list[str] of B prefixed queries
      -> tokenize      -> input_ids      (B, S) int64
                          attention_mask (B, S) int64, 1 = real token, 0 = padding

S is the longest tokenized sequence in the batch, capped at MAX_SEQ_LENGTH. Shorter rows are
right-padded with pad_token_id (1), and attention_mask is what marks that padding for every
later stage.

Host-side string work only, no tensors until the tokenizer runs. The prefix is applied to the
query before tokenizing, so it becomes ordinary tokens the encoder attends to rather than a
special embedding.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence, Union


class NomicPromptPrefix(str, Enum):
    """The task prefixes the model was trained with, verbatim from config_sentence_transformers.json.

    The trailing space is part of the prefix. Upstream names nine MTEB tasks but maps them onto
    only these four prefixes, so the prefix is the member and the repeats are not spelled out:
    an STS or summarization task takes CLASSIFICATION, a speed task takes PASSAGE.
    """

    CLASSIFICATION = "classification: "
    PASSAGE = "search_document: "
    CLUSTERING = "clustering: "
    QUERY = "search_query: "


# From sentence_bert_config.json.
MAX_SEQ_LENGTH = 512


def apply_prompt(
    queries: Sequence[str],
    prompt_prefix: Optional[Union[NomicPromptPrefix, Sequence[Optional[NomicPromptPrefix]]]],
) -> list[str]:
    """Prepend the task prefix to each query.

    The model was trained with these prefixes, so dropping one measurably moves the embedding.
    The prefix is prepended to the query rather than injected as a token, which is why it shows
    up as ordinary tokens after tokenization.

    A single prefix covers the whole batch; a sequence gives each query its own, which is what
    asymmetric retrieval needs, since a search query and the documents it is scored against take
    different prefixes but have to be embedded together to share a batch.

    Args:
        queries: The input strings, length B.
        prompt_prefix: One NomicPromptPrefix for every query, a sequence of B of them for one
            per query, or None to pass the queries through unchanged. A None entry inside the
            sequence leaves that one query unprefixed.

    Returns:
        list[str]: Length B, each entry its prefix concatenated with the original query.

    Raises:
        ValueError: If a prefix is not a NomicPromptPrefix, or if a sequence of prefixes does
            not hold exactly one entry per query.
    """
    if prompt_prefix is None or isinstance(prompt_prefix, str):
        per_query = [prompt_prefix] * len(queries)
    else:
        per_query = list(prompt_prefix)
        if len(per_query) != len(queries):
            raise ValueError(f"got {len(per_query)} prompt prefixes for {len(queries)} queries")

    prefixes = ["" if prefix is None else NomicPromptPrefix(prefix).value for prefix in per_query]
    return [prefix + query for prefix, query in zip(prefixes, queries)]


def tokenize(tokenizer, queries: Sequence[str], max_length: int = MAX_SEQ_LENGTH):
    """Tokenize a batch of queries into padded tensors.

    padding=True pads to the batch maximum rather than to max_length, so S varies with the
    input rather than being fixed at 512.

    Args:
        tokenizer: The XLMRobertaTokenizerFast loaded from the checkpoint.
        queries: The input strings, length B, already prefixed by apply_prompt.
        max_length: Truncation limit, defaulting to MAX_SEQ_LENGTH (512).

    Returns:
        BatchEncoding: With input_ids (B, S) int64, bos=0 ... eos=2 and right-padded with
        pad=1, and attention_mask (B, S) int64 holding 1 for real tokens and 0 for padding.
    """
    return tokenizer(
        list(queries),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
