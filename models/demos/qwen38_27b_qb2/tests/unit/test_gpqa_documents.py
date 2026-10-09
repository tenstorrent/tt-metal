# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Synthetic GPQA examples: never publish questions from the actual benchmark."""

import random

import pytest
from datasets import Dataset

from models.demos.qwen38_27b_qb2.tests.gpqa_documents import process_gpqa_docs


def document(choices):
    return {
        "Question": "Synthetic question; preserve its [notation].",
        **{f"Incorrect Answer {i + 1}": value for i, value in enumerate(choices[:3])},
        "Correct Answer": choices[3],
    }


@pytest.mark.parametrize(
    "choices",
    [
        ["Vector [1, 2]", "Vector [2, 3]", "Vector [3, 4]", "Vector [4, 5]"],
        ["[H+] = 1 M", "[OH-] = 1 M", "[Na+] = 1 M", "[Cl-] = 1 M"],
        ["[0, 1)", "(0, 1]", "[0, 1]", "(0, 1)"],
        ["a  b", "a [title] b", "a [x[y]z] b", " a [correct] b "],
    ],
)
def test_scientific_notation_and_gold_label_survive_shuffle(choices):
    raw = document(choices)
    random.seed(42)
    doc = process_gpqa_docs(Dataset.from_list([raw]))[0]
    displayed = [doc[f"choice{i}"] for i in range(1, 5)]
    assert sorted(displayed) == sorted(s.strip() for s in choices)
    assert len(set(displayed)) == 4
    assert displayed[ord(doc["answer"][1]) - ord("A")] == choices[3].strip()
    assert doc["Question"] == raw["Question"]
    assert doc["Correct Answer"] == choices[3]


def test_correctness_tracks_original_choice_even_if_text_is_duplicated():
    # Seed 42 places the tagged correct entry after the matching incorrect one;
    # a string index would select that earlier entry with the same text.
    raw = document(["other 1", "duplicate", "other 2", "duplicate"])
    random.seed(42)
    expected = list(enumerate([raw[f"Incorrect Answer {i}"] for i in range(1, 4)] + [raw["Correct Answer"]]))
    random.shuffle(expected)
    random.seed(42)
    doc = process_gpqa_docs(Dataset.from_list([raw]))[0]
    correct_index = next(i for i, (original_index, _) in enumerate(expected) if original_index == 3)
    assert doc["answer"] == f"({chr(65 + correct_index)})"


def test_seeded_permutations_match_legacy_order_without_answer_rewriting():
    docs = [document([f"row {i} choice {j}" for j in range(4)]) for i in range(4)]
    random.seed(42)
    expected = []
    for doc in docs:
        row = [doc[f"Incorrect Answer {i}"] for i in range(1, 4)] + [doc["Correct Answer"]]
        random.shuffle(row)
        expected.append(row)
    random.seed(42)
    actual = process_gpqa_docs(Dataset.from_list(docs))
    assert [[doc[f"choice{i}"] for i in range(1, 5)] for doc in actual] == expected


@pytest.mark.parametrize("invalid", [None, "", "   "])
def test_empty_choices_fail_before_inference(invalid, expect_error):
    raw = document(["one", "two", invalid, "correct"])
    with expect_error(ValueError, "nonempty strings"):
        process_gpqa_docs(Dataset.from_list([raw]))
