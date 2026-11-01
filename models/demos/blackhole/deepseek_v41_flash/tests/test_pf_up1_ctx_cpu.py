# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests (no device) of the context-switch state of the one-slot (Up=1) prefill context next to the Up=U one.

Regression for the stale vision-buffer bug found in the rc4b validation (a call on the one-slot context cleared the 'vision buffers dirty' flag while the Up=U context kept a stale image mask /
embedding override: a later text-only window of the Up=U context saw image rows). The device is replaced by a model of the vision buffers: every ``copy_host_to_device_tensor`` that EXECUTES updates the
simulated content of its destination buffer; recorded-but-unreplayed ops do not.
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import pf_up1
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_model as PMOD
from models.demos.blackhole.deepseek_v41_flash.tt.h2d import recording, replay

ROWS, COLS, R = 4, 8, 1024


class Dev:
    def __init__(self, name):
        self.name, self.has_image = name, False


class Host:
    def __init__(self, nonzero):
        self.nonzero = nonzero


@pytest.fixture
def env(monkeypatch):
    def fake_from_torch(t, *a, **k):
        return Host(bool(t.abs().sum() > 0))

    def fake_copy(host, dev):
        if dev.name.endswith(".ov"):
            dev.has_image = host.nonzero

    monkeypatch.setattr(ttnn, "from_torch", fake_from_torch)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", fake_copy)
    pm = PMOD.DSV41PrefillModel.__new__(PMOD.DSV41PrefillModel)
    pm.vis, pm.cs, pm.cols, pm.rows = True, False, COLS, ROWS
    pm.timing, pm._vis_dirty = {}, {}
    pm.shard_rc = None
    pm._host = lambda t, *a, **k: Host(bool(t.abs().sum() > 0))
    bufs = {}
    for ctx in ("up2", "up1"):  # one set of persistent buffers per context, as ``_bufs`` of the bundle
        vb = (Dev(ctx + ".ov"), Dev(ctx + ".vl"))
        bufs[ctx] = (Dev(ctx + ".tok"), {}, vb)
        pm._vis_dirty[id(vb[0])] = False  # (``alloc_inputs``)
    return pm, bufs


def prepped(image):
    n = ROWS * R
    out = {"tok": torch.zeros(n, 1, dtype=torch.int32)}
    if image:
        out["ov"] = torch.ones(n // COLS, COLS * PMOD.DIM, dtype=torch.bfloat16)
        out["vl"] = torch.ones(n // COLS, COLS)
    return out


def upload(pm, bufs, ctx, image, record=None):
    if record is None:
        pm.upload_inputs(prepped(image), bufs[ctx])
        return None
    with recording() as ops:
        pm.upload_inputs(prepped(image), bufs[ctx])
    return ops


def stale(bufs, ctx):
    return bufs[ctx][2][0].has_image


def test_image_then_short_text_on_up1_then_text_on_up2(env):
    pm, bufs = env
    upload(pm, bufs, "up2", True)  # image window on the Up=U context
    assert stale(bufs, "up2")
    upload(pm, bufs, "up1", False)  # one-slot call: must not clear the Up=U buffers' flag
    upload(pm, bufs, "up2", False)  # long text window on the Up=U context
    assert not stale(bufs, "up2"), "stale image rows left in the Up=U vision buffers"


def test_text_image_text_both_orders(env):
    pm, bufs = env
    for first, second in (("up2", "up1"), ("up1", "up2")):
        upload(pm, bufs, first, False)
        upload(pm, bufs, first, True)
        upload(pm, bufs, second, False)
        upload(pm, bufs, first, False)
        assert not stale(bufs, first) and not stale(bufs, second)


def test_clean_buffer_is_not_cleared_again(env):
    pm, bufs = env
    copies = []
    orig = ttnn.copy_host_to_device_tensor
    ttnn.copy_host_to_device_tensor = lambda h, d: (copies.append(d.name), orig(h, d))[1]
    try:
        upload(pm, bufs, "up2", False)
        upload(pm, bufs, "up2", False)
    finally:
        ttnn.copy_host_to_device_tensor = orig
    assert not [
        c for c in copies if c.endswith(".ov")
    ], "text windows must not pay the vision clear when the buffer is clean"


def test_recorded_then_discarded_clear_is_redone(env):
    """a lookahead upload recorded ahead (the clear is IN the recorded ops) and then discarded: the next upload must clear again"""
    pm, bufs = env
    upload(pm, bufs, "up2", True)
    ops = upload(pm, bufs, "up2", False, record=True)  # records the clear (flag says clean now)
    assert stale(bufs, "up2")  # recorded, not executed
    del ops  # discarded without replay
    pm.vis_mark_dirty()  # what the call site does for a discarded lookahead
    upload(pm, bufs, "up2", False)
    assert not stale(bufs, "up2")


def test_recorded_and_replayed_in_order(env):
    pm, bufs = env
    a = upload(pm, bufs, "up2", True, record=True)  # window k (image) recorded ahead
    b = upload(pm, bufs, "up2", False, record=True)  # window k+1 (text) recorded ahead, before k ran
    replay(a)
    assert stale(bufs, "up2")
    replay(b)
    assert not stale(bufs, "up2")


class FakeDyn:
    def __init__(self, S):
        self.S_pad, self.calls = S, []

    def prebuild(self, s0):
        self.calls.append(("prebuild", list(s0)))

    def enqueue_prebuilt(self, epoch):
        self.calls.append(("enqueue", epoch))


def _setup(active_S, up1_S, main_S):
    d = {up1_S: FakeDyn(up1_S), main_S: FakeDyn(main_S)}
    pm = type("PM", (), {})()
    pm.dyn = d[active_S]
    pm.dyn_stash = {S: {"pm": {"dyn": x}} for S, x in d.items() if S != active_S}
    pm._win_count = 41
    return pm, {"S": up1_S, "main": main_S, "C": 512}, d


def test_prebuild_other_context_after_up1_call():
    pm, up1, d = _setup(16384, 16384, 1048576)
    pf_up1.prebuild_other_ctx(pm, up1, ROWS, 2, True)
    assert d[1048576].calls == [
        ("prebuild", [0] * (ROWS * 2)),
        ("enqueue", 41),
    ]  # the Up=U tables of a fresh prompt, in place for the next call
    assert d[16384].calls == []  # the active context is handled by the call's own prebuild


def test_prebuild_other_context_after_up2_call():
    pm, up1, d = _setup(1048576, 16384, 1048576)
    pf_up1.prebuild_other_ctx(pm, up1, ROWS, 2, False)
    assert d[16384].calls == [("prebuild", [0] * ROWS)]  # one slot per mesh row; no enqueue without speculative enqueue


def test_no_other_context_without_up1():
    pm, up1, d = _setup(1048576, 16384, 1048576)
    pf_up1.prebuild_other_ctx(pm, None, ROWS, 2, True)
    assert all(not x.calls for x in d.values())
