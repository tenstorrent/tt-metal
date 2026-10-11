# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ownership and asynchronous mailbox checks for the experimental fused op."""

import random

import pytest

from models.demos.qwen38_27b_qb2.tt.gdn_fused.op import consumer_source, replace_once, schedule


@pytest.mark.parametrize("heads", [1, 23, 24, 25, 192, 384])
def test_groups_cover_every_head_and_partition_once(heads):
    plan = schedule(heads, 12, 10)
    owned, cores = [], []
    for group in plan:
        cores += [*group["producers"], group["consumer"]]
        for step in range(group["count"]):
            head = group["first"] + step * group["stride"]
            owned += [(head, p) for p in range(4)]
    assert sorted(owned) == [(h, p) for h in range(heads) for p in range(4)]
    assert len(cores) == len(set(cores))
    assert all(0 <= x < 12 and 0 <= y < 10 for x, y in cores)


@pytest.mark.parametrize("groups", [1, 6, 12, 24])
def test_skewed_producers_cannot_overwrite_or_mix_heads(groups):
    # Arbitrarily interleave producer progress and consumer work over many wraps.
    for group in schedule(384, 12, 10, groups=groups):
        rng = random.Random(724 + group["first"])
        ready = [0] * 4
        consumed = 0
        mailbox = [[None] * 4 for _ in range(2)]
        while consumed < group["count"]:
            possible = [p for p in range(4) if ready[p] < group["count"] and ready[p] < consumed + 2]
            if min(ready) > consumed:
                possible.append("consume")
            assert possible, "Protocol deadlocked"
            action = rng.choice(possible)
            if action == "consume":
                head = group["first"] + consumed * group["stride"]
                assert mailbox[consumed % 2] == [(head, p) for p in range(4)]
                consumed += 1
            else:
                sequence = ready[action]
                head = group["first"] + sequence * group["stride"]
                mailbox[sequence % 2][action] = (head, action)
                ready[action] += 1
        assert ready == [group["count"]] * 4


def test_a_single_total_ready_count_is_insufficient():
    ready = [2, 2, 0, 0]
    assert sum(ready) == 4 and min(ready) == 0


@pytest.mark.parametrize("args", [(0, 12, 10), (1, 1, 4), (192, 0, 10), (True, 12, 10)])
def test_invalid_schedules_fail_before_device_open(args, expect_error):
    with expect_error(ValueError, "required|Require"):
        schedule(*args)


def test_source_transform_preserves_math_and_fails_closed(expect_error):
    source = consumer_source()
    assert "raw.get_noc_addr" not in source
    assert "noc_semaphore_wait_min(ready, item + 1)" in source
    assert "noc_semaphore_inc(credits[p], 1)" in source
    assert "mailbox + (item % 2) * 512" in source
    with expect_error(ValueError, "anchor changed"):
        replace_once("ambiguous ambiguous", "ambiguous", "changed")
