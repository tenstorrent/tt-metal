# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prove partition remapping preserves every state tile and output segment."""

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_bank_placement import MODES, assignments, bank_pair, cost, placement
from models.demos.qwen38_27b_qb2.tt.gdn_step.op import work_items

READ = ((0, 9), (0, 0), (0, 7), (0, 3), (6, 9), (6, 1), (6, 6), (6, 4))
WRITE = tuple((11 - x, 9 - y) for x, y in READ)


@pytest.mark.parametrize("heads", (192, 384))
@pytest.mark.parametrize("mode", MODES)
def test_complete_disjoint_state_and_output_ownership(heads, mode):
    rows = assignments(heads, 12, 10, 4, read=READ, write=WRITE, mode=mode)
    native = work_items(heads, 12, 10, 4)
    assert [(first, stride, count) for _, _, first, stride, count in rows] == [row[2:] for row in native]
    assert len({(x, y) for x, y, *_ in rows}) == 120
    tiles, outputs = [], []
    for _, _, first, stride, count in rows:
        for i in range(count):
            work = first + i * stride
            head, partition = divmod(work, 4)
            owned = [head * 16 + key_row * 4 + partition for key_row in range(4)]
            assert set(tile % 8 for tile in owned) == set(bank_pair(first))
            tiles.extend(owned)
            outputs.append(head * 4 + partition)
    assert sorted(tiles) == list(range(heads * 16))
    assert sorted(outputs) == list(range(heads * 4))
    if mode == "native":
        assert rows == native


@pytest.mark.parametrize("mode", MODES[1:])
def test_heuristic_reduces_its_declared_distance_on_fixture(mode):
    native = placement(READ, WRITE, "native")
    candidate = placement(READ, WRITE, mode)
    total = lambda cores: sum(cost(core, i % 4, READ, WRITE, mode) for i, core in enumerate(cores))
    assert total(candidate) <= total(native)
    assert candidate == placement(READ, WRITE, mode)


@pytest.mark.parametrize("anchors", ((), ((0, 0),) * 7, ((12, 0),) * 8, ((0.0, 0),) * 8))
def test_invalid_anchor_geometry_is_rejected(anchors, expect_error):
    with expect_error(ValueError, "eight logical bank anchors"):
        placement(anchors, WRITE, "bank_read")


def test_unsupported_geometry_is_rejected(expect_error):
    with expect_error(ValueError, "requires B16/B32"):
        assignments(96, 12, 10, 4, read=READ, write=WRITE, mode="bank_read")


def test_unknown_mode_is_rejected(expect_error):
    with expect_error(ValueError, "Unknown bank placement mode"):
        placement(READ, WRITE, "other")
