# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Word boundaries must survive any kind of whitespace.

The pinned vocab's Whitespace pre-tokenizer discards raw separators, so ``preprocess_text``
substitutes ``[SPACE]`` before BPE. Substituting only literal " " left newlines, tabs, CRLF and
NBSP to be dropped silently: pasted multiline text reached the model with no word gap at all.
"""

import pytest
import torch

from models.experimental.xtts.config import DEMO
from models.experimental.xtts.reference.xtts_text_embedding import _load_tokenizer, preprocess_text


@pytest.fixture(scope="module")
def space_id():
    """The [SPACE] marker the model reads as a word gap."""
    return _load_tokenizer().token_to_id("[SPACE]")


@pytest.mark.parametrize(
    "separator",
    # Escapes, not literals: a raw U+2028 is a Python line terminator (str.splitlines splits
    # on it) so line-oriented scanners mis-count this file -- it broke the SPDX CI check.
    ["\n", "\t", "\r\n", "\xa0", "\n\n", " \n ", "\t\t", "\u2028"],
    ids=["newline", "tab", "crlf", "nbsp", "blank-line", "mixed", "double-tab", "line-sep"],
)
def test_separator_yields_one_space_marker(separator, space_id):
    """Any whitespace run between two words must tokenize exactly like a single space."""
    expected = preprocess_text("hello world")[0].tolist()
    got = preprocess_text(f"hello{separator}world")[0].tolist()
    assert got == expected, f"{separator!r} did not tokenize as a single space"
    assert got.count(space_id) == 1


def test_repeated_spaces_collapse(space_id):
    """A run of spaces is one gap, not several -- each marker costs a text position."""
    assert preprocess_text("hello   world")[0].tolist() == preprocess_text("hello world")[0].tolist()


def test_multiline_paragraph_keeps_every_boundary(space_id):
    """A pasted paragraph must carry one marker per word gap."""
    text = "One two three.\nFour five six.\n\tSeven eight."
    ids = preprocess_text(text)[0].tolist()
    assert ids.count(space_id) == 7, f"expected 7 word gaps, got {ids.count(space_id)}"
    assert ids == preprocess_text("One two three. Four five six. Seven eight.")[0].tolist()


def test_clean_text_is_unchanged():
    """Normalization must not perturb text that was already single-spaced."""
    ids = preprocess_text(DEMO.text)[0].tolist()
    assert ids == preprocess_text(" ".join(DEMO.text.split()))[0].tolist()
    assert torch.tensor(ids).numel() > 0
