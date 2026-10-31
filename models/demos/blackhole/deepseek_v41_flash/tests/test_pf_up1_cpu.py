# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the one-slot short-prompt context selection (tt/pf_up1.py)."""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import pf_up1

C, ROWS, U = 512, 4, 2


def it(b, e, st=0):
    return (b, torch.zeros(max(e, 1), dtype=torch.long), st, e)


def ok(items, final=None, **kw):
    final = {b: True for b, _, _, _ in items} if final is None else final
    return pf_up1.eligible(items, C, ROWS, U, U, final, **kw)


def test_lone_short_prompt():
    assert ok([it(3, 18)])
    assert ok([it(0, 512)])  # exactly one chunk, final


def test_one_user_per_mesh_row_only():
    assert ok([it(0, 18), it(2, 40), it(5, 100), it(6, 7)])  # rows 0, 1, 2, 3
    assert not ok([it(0, 18), it(1, 40)])  # both in row 0


def test_not_from_scratch_or_not_whole_prompt():
    assert not ok([it(0, 600)])  # needs two windows
    assert not ok([it(0, 18, st=512)])  # a continuation
    assert not ok([it(0, 18)], final={0: False})  # not the last piece of the prompt
    assert not pf_up1.eligible([it(0, 18)], C, ROWS, U, U, None)


def test_images_hits_and_owned():
    assert not ok([it(0, 18)], vis={0: object()})
    assert not ok([it(0, 18), it(2, 30)], has_hit=lambda b, t, e: b == 2)
    assert ok([it(0, 18), it(2, 30)], has_hit=lambda b, t, e: False)
    assert not ok([it(0, 18)], owned=lambda b: True)


def test_modes_without_two_slots():
    assert not pf_up1.eligible([it(0, 18)], C, ROWS, 1, 1, {0: True})  # U = 1: the trace has one slot already
    assert not pf_up1.eligible([it(0, 18)], C, ROWS, 4, 1, {0: True})  # Up < U (general mode already)
    assert not ok([])


def test_small_spad(monkeypatch):
    monkeypatch.delenv("DSV41_PF_UP1_SPAD", raising=False)
    assert pf_up1.small_spad(1 << 20, 512) == 16384
    assert pf_up1.small_spad(16384, 512) == 0  # no room below the main context
    monkeypatch.setenv("DSV41_PF_UP1_SPAD", "20000")
    assert pf_up1.small_spad(1 << 20, 512) == 20480
