# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic book-window selection and source binding, using only stdlib."""
import hashlib
import re
import unicodedata
from pathlib import Path

BOOKS = (("1342", "Pride and Prejudice"), ("1400", "Great Expectations"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_window(book_ids, decode, *, context_length, bos_id, vocab_size):
    require(type(context_length) is int and 2 <= context_length <= 131072, "Invalid context length")
    require(type(vocab_size) is int and vocab_size > 1, "Invalid vocabulary")
    require(type(bos_id) is int and 0 <= bos_id < vocab_size, "Invalid BOS")
    require(all(type(i) is int and 0 <= i < vocab_size for i in book_ids), "Invalid book token ID")
    require(bos_id not in book_ids, "Book unexpectedly contains a BOS token")
    first = context_length - 1
    for end in range(first, min(first + 512, len(book_ids) - 1)):
        next_piece = decode([book_ids[end]])
        following_piece = decode([book_ids[end + 1]])
        if not re.fullmatch(r"\s+[A-Za-z]+", next_piece):
            continue
        if not following_piece or not (
            following_piece[0].isspace() or unicodedata.category(following_piece[0]).startswith("P")
        ):
            continue
        start = end - (context_length - 1)
        ids = [bos_id] + list(book_ids[start:end])
        require(len(ids) == context_length and ids.count(bos_id) == 1, "Incorrect prompt length/BOS")
        return dict(
            token_ids=ids,
            book_token_range=[start, end],
            candidate_offset=end - first,
            expected_next_token_id=book_ids[end],
            expected_next_token=next_piece,
            expected_next_word=next_piece.lstrip(),
            following_token=following_piece,
        )
    raise ValueError("No whole-word candidate with a following boundary in the first 512 endpoints")


def validate_source_binding(binding):
    slot = binding["slot"]
    require(type(slot) is int and slot in (0, 1), "Invalid book slot")
    require((binding["book_id"], binding["title"]) == BOOKS[slot], "Book/slot identity mismatch")
    source = Path(binding["source_path"]).read_bytes()
    body_bytes = Path(binding["body_path"]).read_bytes()
    require(hashlib.sha256(source).hexdigest() == binding["source_sha256"], "Downloaded book changed")
    require(hashlib.sha256(body_bytes).hexdigest() == binding["body_sha256"], "Book body changed")
    require(binding["source_encoding"] == "utf-8-sig", "Unexpected source encoding")
    text = source.decode("utf-8-sig")
    body = body_bytes.decode("utf-8")
    start, end = binding["body_start_character"], binding["body_end_character"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "Invalid body range")
    require(
        text[start:end] == body and len(body) == binding["body_characters"], "Body does not match exact source range"
    )
    return body
