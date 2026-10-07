# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Detect serial completion attribution without hardware or timing thresholds."""

import threading

import pytest

from models.demos.qwen38_27b_qb2.tests.replica_timing import completion_times


@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_fast_replica_timestamp_is_not_charged_slow_wait(order):
    fast_timestamp_recorded = threading.Event()
    state = threading.local()

    def synchronize(replica):
        state.replica = replica
        if replica == "slow":
            assert fast_timestamp_recorded.wait(2), "Waiters must run independently"

    def clock():
        if state.replica == "fast":
            fast_timestamp_recorded.set()
            return 101.0
        return 205.0

    # The first submitted future can finish last. Timestamps must be captured
    # inside the worker, not while results are joined in submission order.
    assert completion_times(["slow", "fast"], synchronize, order, clock=clock) == [205.0, 101.0]


def test_completion_propagates_sync_failure(expect_error):
    def synchronize(_):
        raise RuntimeError("device synchronization failed")

    with expect_error(RuntimeError, "device synchronization failed"):
        completion_times([0], synchronize, [0])


@pytest.mark.parametrize("items,order", [([], []), ([0, 1], [0, 0]), ([0, 1], [0]), ([0], [1])])
def test_completion_rejects_invalid_order(items, order, expect_error):
    with expect_error(ValueError, "permutation of nonempty replicas"):
        completion_times(items, lambda _: None, order)
