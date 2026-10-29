# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of DSV41_PREFILL_UP_BATCH (several prefill slots per mesh row, every slot at its own chunk start, ONE trace replay): the host values of the per-slot tensors.
No device: the DynCtx / SingleCtx objects are built without ``__init__`` and the host uploads are replaced by identity functions.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_prefill_up_batch_cpu.py -q
"""


import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_dyn import HEAD_DIM, WINDOW, DynCtx

ROWS, COLS, C = 4, 8, 512


class FakeRope:
    """rope tables as a deterministic function of the absolute position (what attention._rope_inputs gives, up to values)"""

    def _rope_inputs(self, pos):
        pos = torch.as_tensor(pos).float()
        c = pos.view(-1, 1) * torch.ones(1, HEAD_DIM) * 0.001
        return c.cos(), c.sin()


def make_ctx(U, per_row=True, ratios=(0, 4, 128)):
    ctx = object.__new__(DynCtx)
    ctx.C, ctx.S_pad, ctx.rows, ctx.cols = C, 1 << 20, ROWS, COLS
    ctx.rope_src = {False: FakeRope(), True: FakeRope()}
    ctx.ratios = sorted(r for r in ratios if r)
    ctx.L = {r: ctx.S_pad // r for r in ctx.ratios}
    ctx.per_row, ctx.U = per_row, (U if per_row else 1)
    ctx.nslot = ROWS * ctx.U
    ctx.lat_tabs = {r: object() for r in ctx.ratios if r > 1}
    ctx._host = lambda t, dtype: t  # (no mesh mapper: the host tensor itself)
    ctx.tabs = {k: tuple(f"tab{k}{i}" for i in range(3)) for k in ctx.rope_src}
    ctx.lat_tabs = {r: tuple(f"lat{r}{i}" for i in range(3)) for r in ctx.ratios if r > 1}
    ctx.win_mask, ctx.lim, ctx.off = "win", {r: f"lim{r}" for r in ctx.ratios}, {r: f"off{r}" for r in ctx.ratios}
    return ctx


def by_dev(ctx, s0):
    return {dev: host for host, dev in ctx._host_tensors(s0)}


@pytest.mark.parametrize("U", [1, 2, 4])
def test_per_slot_tables_equal_single_start_tables(U):
    """slot k = row * U + u of the concatenated tensor is exactly the table of a replay at the start of that slot"""
    ctx = make_ctx(U)
    s0s = [
        512 * ((7 * k) % 13) + (0 if k % 3 else 4096) for k in range(ROWS * U)
    ]  # mixed starts, several slots at 0 / equal
    assert len(set(s0s)) > 3
    full = by_dev(ctx, s0s)
    for k, x in enumerate(s0s):
        one = make_ctx(1, per_row=False)
        ref = by_dev(one, x)
        for dev, host in full.items():
            assert host.shape[0] == ROWS * U, (dev, host.shape)
            assert torch.equal(host[k : k + 1], ref[dev]), (dev, k, x)


def test_one_slot_per_row_is_unchanged():
    """U == 1 per_row: dim 0 = rows, the values of the existing per-row path (bit-identical when the switch is off)"""
    ctx = make_ctx(1)
    s0s = [0, 512, 1024, 8192]
    full = by_dev(ctx, s0s)
    for dev, host in full.items():
        assert host.shape[0] == ROWS


def test_non_per_row_rejects_mixed_starts():
    ctx = make_ctx(2, per_row=False)
    try:
        ctx._host_tensors([0, 512] * 4)
        raised = False
    except AssertionError:
        raised = True
    assert raised
    same = by_dev(ctx, 1024)
    assert all(h.shape[0] == 1 for h in same.values())


def test_first_b_per_slot():
    ctx = make_ctx(2)
    ctx.p0 = None
    assert ctx.first_b([0, 512, 0, 1024]) == [True, False, True, False]
    ctx.p0 = [1024, 0, 2048, 512]  # per slot zone (len == slots)
    assert ctx.first_b([512, 512, 2048, 1024]) == [True, False, True, False]
    ctx.p0 = [0] * ROWS  # per row zone with per-slot starts: the min rule (a mismatch of lengths never raises)
    assert len(ctx.first_b([0] * (2 * ROWS))) == 2 * ROWS


class FakeMD:
    shape = (ROWS, COLS)


def test_single_ctx_values_per_slot():
    U = 2
    sg = object.__new__(PS.SingleCtx)
    sg.md, sg.U, sg.rows, sg.cols = FakeMD(), U, ROWS, COLS
    CS = (
        2048  # a chunk whose key entries split into whole 32-row tiles per column (tt/prefill_single.shard_cols_single)
    )
    sg.ctx = type("c", (), {"C": CS})()
    s0s = [
        1024 * k for k in range(ROWS * U)
    ]  # (the chunk geometry of the single trace: key entries per column are whole 32-row tiles)
    act = torch.tensor([k % 3 != 0 for k in range(ROWS * U)])
    for r in (4,):
        nloc, ko = sg.values(r, s0s, act)
        assert len(nloc) == len(ko) == ROWS * U
        for k, x in enumerate(s0s):
            assert nloc[k] == (CS // r) // COLS + (x // r) // COLS
            assert int(ko[k]) == (nloc[k] // 32 if bool(act[k]) else 0)
        # the column of slot u over the rows (what SingleCtx.update uploads per slot tensor): entries u, u + U, ...
        for u in range(U):
            assert nloc[u::U] == [nloc[row * U + u] for row in range(ROWS)]


def test_switch_gate(monkeypatch):
    monkeypatch.setenv("DSV41_PF_SINGLE", "1")
    monkeypatch.delenv("DSV41_PF_PERROW", raising=False)
    monkeypatch.delenv("DSV41_PREFILL_UP_BATCH", raising=False)
    assert PS.per_row_enabled(1) and not PS.per_row_enabled(2)
    monkeypatch.setenv("DSV41_PREFILL_UP_BATCH", "1")
    assert PS.per_row_enabled(1) and PS.per_row_enabled(2) and PS.per_row_enabled(4)
    monkeypatch.setenv("DSV41_PF_PERROW", "0")
    assert not PS.per_row_enabled(2)
    monkeypatch.setenv("DSV41_PF_PERROW", "1")
    monkeypatch.setenv("DSV41_PF_SINGLE", "0")
    assert not PS.per_row_enabled(2)


def test_win_mask_values():
    """the window band mask of slot k: the first chunk (s0 == 0) hides the halo, later chunks see 128 previous positions"""
    ctx = make_ctx(2)
    s0s = [0, 512] * ROWS
    w = by_dev(ctx, s0s)["win"]
    assert w.shape == (ROWS * 2, 1, C, WINDOW + C)
    assert bool((w[0, 0, 0, :WINDOW] < 0).all())  # slot at 0: the halo columns are masked
    assert bool(
        (w[1, 0, 0, :WINDOW][-1] == 0)
    )  # slot at 512: position s0 - 1 is visible to the first query? (kp > t - 128: yes)
