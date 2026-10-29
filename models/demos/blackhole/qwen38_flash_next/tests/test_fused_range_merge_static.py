# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Core range merging (``ttnn/fused/range_merge.py``, ``QWEN38_RANGE_MERGE=1``) without a device: the rectangles cover exactly the
same cores, disjointly; the switch is off by default; program.py applies it at the three descriptor builders and at run_program."""

import inspect
import random

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import range_merge as rm


def _covered(rects):
    return [(x, y) for x0, y0, x1, y1 in rects for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def test_rectangles_cover_exactly_the_cores():
    rng = random.Random(7)
    for _ in range(400):
        cores = {(rng.randrange(13), rng.randrange(10)) for _ in range(rng.randrange(1, 90))}
        cover = _covered(rm.rectangles(cores))
        assert len(cover) == len(set(cover)) == len(cores) and set(cover) == cores


def test_known_shapes_merge_to_few_ranges():
    block = {(x, y) for x in range(5) for y in range(2)}
    assert rm.rectangles(block) == [(0, 0, 4, 1)]
    assert rm.rectangles(block | {(6, 0), (7, 0)}) == [(0, 0, 4, 1), (6, 0, 7, 0)]
    assert len(rm.rectangles({(x, 3) for x in range(0, 12, 2)})) == 6  # nothing adjacent: nothing to merge
    grid = {(x, y) for x in range(8) for y in range(10)}
    assert rm.rectangles(grid) == [(0, 0, 7, 9)]
    assert len(rm.rectangles(grid - {(3, 4)})) <= 4


def test_off_by_default_and_wired_in_program_py():
    assert not rm.enabled({}) and rm.enabled({rm.ENV: "1"}) and not rm.stats_enabled({})
    source = inspect.getsource(fp)
    for needle in (
        'range_merge.apply(cores, "cb")',
        'range_merge.apply(cores, "kernel")',
        'range_merge.apply(cores, "semaphore")',
        "range_merge.merge_program(descriptor)",
        "range_merge.flush(",
    ):
        assert needle in source, needle
