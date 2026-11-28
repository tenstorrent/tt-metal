# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the upsampler's layout mode. No device needed. Expected values are the outputs of BFL's
flux3/bbox.py (plan, reconcile) on the same inputs."""

from __future__ import annotations

import pytest

from ....pipelines.flux3_klein import bbox

LOCKED_NOTE = (
    "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at the top-left). "
    "This is an authoritative JSON list of dictionaries; preserve every supplied identity, provenance value, null, "
    "and coordinate exactly:\n"
)
BALLOON = '{"id": "balloon_1", "bbox": [80, 60, 480, 380], "desc": "a red balloon"}'
KNIGHT = (
    '{"id": "knight_1", "from": "ref_image_0", "src_bbox": [100, 100, 500, 400], "tgt_bbox": [100, 500, 500, 800], '
    '"desc": "the knight"}'
)


@pytest.mark.parametrize("task", ["t2i", "i2i"])
@pytest.mark.parametrize("prompt", ["A man walking his dog in the snow", "A poster that says [SALE]", "A list [1, 2]"])
def test_plan_without_boxes_keeps_the_prompt(task, prompt):
    assert bbox.plan(prompt, task) == (task, prompt, None)


@pytest.mark.parametrize(
    "task, prompt, request_, row",
    [
        ("t2i", "A red balloon over a field <balloon_1>", "A red balloon over a field <balloon_1>", BALLOON),
        ("i2i", "Move the knight <knight_1>", "Move the knight <knight_1>", KNIGHT),
    ],
)
def test_plan_with_boxes_selects_the_bbox_prompt(task, prompt, request_, row):
    key, text, locked = bbox.plan(f"{prompt} [{row}]", task)
    assert key == f"{task}_bbox"
    assert text == f"ORIGINAL PROMPT:\n{request_}\n\n{LOCKED_NOTE}[{row}]"
    assert len(locked) == 1 and locked[0]["id"] in prompt


def test_reconcile_keeps_the_users_boxes():
    locked = bbox.plan(f"A red balloon over a field <balloon_1> [{BALLOON}]", "t2i")[2]
    answer = (
        "A wide field under a blue sky, a red balloon <obj_1> floats high and a dog <dog_1> runs below, a tree "
        '<tree_1>. [{"id": "obj_1", "bbox": [82, 61, 478, 381], "desc": "a glossy red balloon"}, {"id": "dog_1", '
        '"bbox": [700, 400, 950, 600], "desc": "a brown dog"}, {"id": "cat_1", "bbox": [700, 100, 900, 300], "desc": '
        '"a cat"}, {"id": "tree_1", "bbox": [0.1, 0.1, 0.5, 0.5], "desc": "a tree"}]'
    )
    assert bbox.reconcile(answer, locked) == (
        "A wide field under a blue sky, a red balloon <balloon_1> floats high and a dog <dog_1> runs below, a tree "
        '<tree_1>. [{"id": "balloon_1", "bbox": [80, 60, 480, 380], "desc": "a glossy red balloon"}, {"id": "dog_1", '
        '"bbox": [700, 400, 950, 600], "desc": "a brown dog"}]'
    )


def test_reconcile_restores_a_missing_table():
    locked = bbox.plan(f"A red balloon over a field <balloon_1> [{BALLOON}]", "t2i")[2]
    assert bbox.reconcile("Just a caption with no table.", locked) == f"Just a caption with no table. [{BALLOON}]"


@pytest.mark.parametrize(
    "prompt, finished",
    [
        ('A red balloon over a field <balloon_1> [{"id": "balloon_1", "bbox": [100, 200, 400, 500]}]', True),
        ('A red balloon over a field [{"id": "balloon_1", "bbox": [100, 200, 400, 500]}]', False),
        ("A red balloon over a field", False),
    ],
)
def test_is_finished_needs_every_row_cited(prompt, finished):
    assert bbox.is_finished(prompt) is finished
