# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ResidentDFlash._prompt with the pipeline-parallel prefill on fakes (CPU, no device): every chip finishes the
drafter's context fill before the prompt's verify step is enqueued (dflash.FILL_SYNC). Without it a resident step's hubs
pushed their first round into the hub slots of a chip still running the fill (2026-10-30: first token 0 and no draft
accepted, after long-decoded requests); tests/dfl_long_predecessor.py is the device test.

usage: python -m pytest --noconftest -c /dev/null qwen36_resident/tests/test_dflash_fill_sync.py
"""
from types import SimpleNamespace
from unittest.mock import patch

import torch

from qwen36_resident import dflash

R, TAPS = 8, 5


def prompt_events(fill_sync, prompt_len=50):
    """the device calls of one _prompt, in order"""
    ev = []

    def rec(name, ret=None):
        return lambda *a, **k: ev.append(name) or ret

    s = object.__new__(dflash.ResidentDFlash)
    s.R, s.window, s.fill_rows, s.use_prefill = R, 2048, 2048 + 64, True
    s.mesh, s.prompt_times, s.tap_log, s.emb, s.tap_masks = None, None, None, torch.zeros(prompt_len, 4), [None] * TAPS
    s.m = SimpleNamespace(
        reset=rec("reset"), token_state=lambda *a, **k: None, step=rec("step"), argmax=rec("argmax", [0] * R)
    )
    s.pp = SimpleNamespace(
        reset=rec("pp.reset"), run=rec("pp.run"), handoff=rec("pp.handoff"), tap_rows=lambda s0, rows: [None] * TAPS
    )
    s.drafter = SimpleNamespace(
        tt_ccl=None, topo=None, fill_context=rec("fill_context"), extend_context=rec("extend_context")
    )
    fake_ttnn = SimpleNamespace(
        mul=lambda x, m: None,
        deallocate=lambda t: None,
        concat=lambda xs, dim: None,
        slice=lambda *a: None,
        synchronize_device=rec("synchronize_device"),
        DRAM_MEMORY_CONFIG=None,
    )
    reduced = SimpleNamespace(shape=(1, 1, TAPS * s.fill_rows, 64))
    with patch.object(dflash, "ttnn", fake_ttnn), patch.object(dflash, "FILL_SYNC", fill_sync), patch.object(
        dflash, "tt_all_reduce", lambda *a, **k: reduced
    ):
        s._prompt(list(range(prompt_len)))
    return ev


def test_the_fill_is_synchronized_before_the_verify_step():
    ev = prompt_events(fill_sync=True)
    fill, step = ev.index("fill_context"), ev.index("step")
    assert fill < step and "synchronize_device" in ev[fill:step], ev


def test_fill_sync_0_enqueues_the_step_right_after_the_fill():
    """RESIDENT_DFLASH_FILL_SYNC=0 (A/B only): the order that raced"""
    ev = prompt_events(fill_sync=False)
    assert ev[ev.index("fill_context") + 1] == "step", ev
