# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prove the diagnostic exercises continuations, ownership and page identity."""

import pytest

from models.demos.qwen38_27b_qb2.tests.chunked_prefill_scenario import run_scenario


def execute(*, lose_continuation=False, skip_remap=False):
    state = []
    pages = {}

    def reset():
        state[:] = [[] for _ in range(8)]
        pages.clear()

    def prefill(rows):
        result = []
        for row, (request, start, end, slot) in enumerate(rows):
            if start == 0:
                state[slot] = []
                pages[request] = []
            if lose_continuation and start:
                slot = row
            state[slot].extend((request, pos) for pos in range(start, end))
            pages[request].extend((request, pos) for pos in range(start, end))
            result.append((tuple(state[slot]), tuple(pages[request])))
        return result

    def decode(rows, remap):
        if remap is not None and not skip_remap:
            old = state[:]
            state[:] = [old[index] for index in remap]
        result = []
        for slot, (request, position) in enumerate(rows):
            state[slot].append((request, position))
            pages[request].append((request, position))
            result.append((tuple(state[slot]), tuple(pages[request])))
        return result

    return run_scenario(reset, prefill, decode)


def test_independent_and_interleaved_requests_have_identical_history():
    reference, actual = execute()
    assert len(reference) == 11
    assert actual == reference
    for (request, position), (state, pages) in actual.items():
        assert state == pages == tuple((request, pos) for pos in range(position + 1))


@pytest.mark.parametrize("defect", ["lose_continuation", "skip_remap"])
def test_scenario_exposes_wrong_slot_or_missing_physical_gather(defect):
    reference, actual = execute(**{defect: True})
    assert actual != reference
    assert actual[("A", 96)] != reference[("A", 96)]


def test_missing_output_is_rejected(expect_error):
    with expect_error(ValueError, "One output row"):
        run_scenario(lambda: None, lambda rows: [], lambda rows, remap: [])
