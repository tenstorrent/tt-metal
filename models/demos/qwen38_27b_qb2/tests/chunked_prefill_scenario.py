# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent and interleaved executions of the same teacher-forced requests.

The callbacks return per-row host logits. They are deliberately independent
of a scheduler's ownership map: explicit permutations describe physical state
moves, and attention pages stay attached to their requests throughout.
"""


def run_scenario(reset, prefill, decode):
    reference, interleaved = {}, {}

    def record(target, rows, outputs):
        if len(rows) != len(outputs):
            raise ValueError("One output row is required for each scheduled request")
        for (request, position), output in zip(rows, outputs):
            key = (request, position)
            if key in target:
                raise ValueError("Duplicate request position in scenario")
            target[key] = output

    # Equal chunk boundaries on each arm separate ownership errors from changes
    # in floating-point reduction order caused by changing the chunk sizes.
    for request, ends, decode_positions in (
        ("A", (31, 64, 97), (97, 98)),
        ("B", (65,), (65, 66)),
        ("C", (32, 33), (33,)),
    ):
        reset()
        start = 0
        for end in ends:
            record(reference, [(request, end - 1)], prefill([(request, start, end, 0)]))
            start = end
        for position in decode_positions:
            record(reference, [(request, position)], decode([(request, position)], None))

    reset()
    record(interleaved, [("A", 30)], prefill([("A", 0, 31, 3)]))
    record(interleaved, [("B", 64)], prefill([("B", 0, 65, 1)]))
    # new_state[dst] = old_state[remap[dst]]: B -> 0, unfinished A -> 1.
    record(interleaved, [("B", 65)], decode([("B", 65)], [1, 3, 0, 2, 4, 5, 6, 7]))
    record(interleaved, [("C", 31), ("A", 63)], prefill([("C", 0, 32, 3), ("A", 31, 64, 1)]))
    record(interleaved, [("B", 66)], decode([("B", 66)], None))
    record(interleaved, [("A", 96), ("C", 32)], prefill([("A", 64, 97, 1), ("C", 32, 33, 3)]))
    record(interleaved, [("C", 33), ("A", 97)], decode([("C", 33), ("A", 97)], [3, 1, 0, 2, 4, 5, 6, 7]))
    record(interleaved, [("A", 98)], decode([("A", 98)], [1, 0, 2, 3, 4, 5, 6, 7]))
    if reference.keys() != interleaved.keys():
        raise ValueError("Both schedules must evaluate the same request positions")
    return reference, interleaved
