# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Preserve scientific answer text when preparing the pinned GPQA dataset."""

import random

CHOICE_PROCESSING = "preserve_scientific_notation_v1"


def process_gpqa_docs(dataset):
    """Shuffle tagged choices without deleting bracketed formulas or vectors.

    The caller seeds the choice permutation. Keep correctness attached to the
    original answer: searching the displayed strings can mislabel duplicates.
    Only surrounding whitespace is removed; internal answer text is unchanged.
    """

    def prepare(doc):
        choices = []
        for key, correct in (
            ("Incorrect Answer 1", False),
            ("Incorrect Answer 2", False),
            ("Incorrect Answer 3", False),
            ("Correct Answer", True),
        ):
            value = doc[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("GPQA choices must be nonempty strings")
            choices.append((value.strip(), correct))
        random.shuffle(choices)
        correct_index = next(i for i, (_, correct) in enumerate(choices) if correct)
        return {
            **{f"choice{i + 1}": value for i, (value, _) in enumerate(choices)},
            "answer": f"({chr(65 + correct_index)})",
        }

    return dataset.map(prepare, load_from_cache_file=False)
