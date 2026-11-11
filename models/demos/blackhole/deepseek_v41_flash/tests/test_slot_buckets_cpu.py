# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests (no device) of the slot-bucketed prefill windows (DSV41_PREFILL_SLOT_BUCKETS): selection, context keys / stash, the call wrapper, the prefix view, and the placement that keeps prompts in low slots."""

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import dsv41_model as DM
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_model as PMOD
from models.demos.blackhole.deepseek_v41_flash.tt import slot_buckets as SB
from models.demos.blackhole.deepseek_v41_flash.tt.vllm_state import SlotTable

U, ROWS = 4, 4


def test_parse_need_pick():
    assert SB.parse("1,2", 4) == [1, 2]
    assert SB.parse("2;1;2;4;0;9", 4) == [1, 2]  # k >= U and k < 1 dropped, sorted, unique
    assert SB.parse("", 4) == [] and SB.parse(None, 4) == []
    assert SB.parse("1", 1) == []  # U = 1: nothing below
    assert SB.need_slots([0, 4, 8, 12], 4) == 1  # one prompt per row, all in slot 0
    assert SB.need_slots([0, 5], 4) == 2
    assert SB.need_slots([3], 4) == 4
    assert SB.need_slots([], 4) == 0
    assert SB.pick([1, 2], {1, 2}, 1) == 1
    assert SB.pick([1, 2], {1, 2}, 2) == 2
    assert SB.pick([1, 2], {1, 2}, 3) == 0  # the full trace
    assert SB.pick([1, 2], {2}, 1) == 2  # bucket 1 not captured: the next one that covers
    assert SB.pick([1, 2], set(), 1) == 0


class FakePM:
    def __init__(self, key, slot_keys):
        self.dyn = object()
        self._key, self._slots = key, slot_keys
        self.used = []

    def dyn_key(self):
        return self._key

    def slot_ctx_keys(self):
        return sorted(self._slots, reverse=True)

    def use_dyn(self, k):
        self.used.append(k)
        self._key = k


def model(key=1048576, slots=(-1, -2), ks=(1, 2), up=U):
    m = DM.Model.__new__(DM.Model)
    m.U, m.Up, m._slot_ks, m._slot_chunk = U, up, list(ks), 512
    m.prefill_model = FakePM(key, slots)
    return m


def it(b, e=512, st=0):
    return (b, torch.zeros(e, dtype=torch.long), st, e)


def test_slot_pick(monkeypatch):
    monkeypatch.delenv("DSV41_PREFILL_SLOT_BUCKETS_FILE", raising=False)
    m = model()
    assert m._slot_pick([it(0)], 512, None) == 1  # a lone prompt: slot 0 of row 0
    assert m._slot_pick([it(0), it(4), it(8), it(12)], 512, None) == 1  # four rows, one prompt each
    assert m._slot_pick([it(0), it(1)], 512, None) == 2  # two prompts in one row
    assert m._slot_pick([it(0), it(6)], 512, None) == 0  # slot 2 of row 1: three slots needed, no 3-slot trace
    assert m._slot_pick([it(3)], 512, None) == 0  # slot 3: the full trace
    assert m._slot_pick([it(0)], 256, None) == 0  # another chunk size: no captured bucket
    assert m._slot_pick([it(0)], 512, {0: object()}) == 0  # image prompt: the full trace
    assert m._slot_pick([], 512, None) == 0


def test_slot_pick_off_states(monkeypatch, tmp_path):
    monkeypatch.delenv("DSV41_PREFILL_SLOT_BUCKETS_FILE", raising=False)
    assert model(ks=())._slot_pick([it(0)], 512, None) == 0
    assert model(up=2)._slot_pick([it(0)], 512, None) == 0  # Up != U: general mode, no buckets
    assert model(key=-1)._slot_pick([it(0)], 512, None) == 0  # a bucket context is active: never nested
    assert model(slots=())._slot_pick([it(0)], 512, None) == 0  # nothing captured (re-capture dropped them)
    assert (
        model(slots=(-2,))._slot_pick([it(0)], 512, None) == 2
    )  # only the 2-slot trace exists: it covers a lone prompt
    f = tmp_path / "sw"
    monkeypatch.setenv("DSV41_PREFILL_SLOT_BUCKETS_FILE", str(f))
    f.write_text("0")
    assert model()._slot_pick([it(0)], 512, None) == 0
    f.write_text("1")
    assert model()._slot_pick([it(0)], 512, None) == 1
    monkeypatch.setenv("DSV41_PREFILL_SLOT_BUCKETS_FILE", str(tmp_path / "missing"))
    assert model()._slot_pick([it(0)], 512, None) == 1


def test_prefill_slots_wrapper_restores_state_even_on_error():
    m = model()
    seen = {}

    def one(items, chunk, s_pad_max, want_logits, vis, final, totals, rids):
        seen.update(up=m.Up, k=m.__dict__["_slot_k"], key=m.prefill_model.dyn_key())
        return {0: (1, None)}

    m._prefill_interleaved_one = one
    m.timing = {}
    m.log = lambda *a, **k: None
    res = m._prefill_slots(1, [it(0)], 512, 1048576, False, None, None, None, None)
    assert res == {0: (1, None)}
    assert seen == dict(up=1, k=1, key=-1)
    assert (m.Up, m.__dict__["_slot_k"], m.prefill_model.dyn_key()) == (U, None, 1048576)
    assert m.prefill_model.used == [-1, 1048576]
    assert m.timing["slot_k"] == 1.0

    def boom(*a, **k):
        raise RuntimeError("x")

    m._prefill_interleaved_one = boom
    try:
        m._prefill_slots(2, [it(0)], 512, 1048576, False, None, None, None, None)
        raise AssertionError("the error was swallowed")
    except RuntimeError:
        pass
    assert (m.Up, m.__dict__["_slot_k"], m.prefill_model.dyn_key()) == (U, None, 1048576)


class Ctx:
    def __init__(self, S, key=None):
        self.S_pad = S
        if key is not None:
            self.ctx_key = key


def test_context_keys_and_stash():
    pm = PMOD.DSV41PrefillModel.__new__(PMOD.DSV41PrefillModel)
    pm.dyn, pm.dyn_trace = Ctx(1 << 20), object()
    pm._dyn_stash = {
        -1: {"pm": {"dyn_trace": object()}},
        -2: {"pm": {"dyn_trace": object()}},
        16384: {"pm": {"dyn_trace": object()}},
        -3: {"pm": {"dyn_trace": None}},
    }
    assert pm.dyn_key() == 1 << 20
    assert pm.dyn_buckets() == [16384, 1 << 20]  # the slot contexts are not S_pad buckets
    assert pm.slot_ctx_keys() == [-1, -2]  # captured ones only, ascending k
    pm.dyn = Ctx(1 << 20, -2)
    del pm._dyn_stash[-2]  # (the active context is not in the stash)
    assert pm.dyn_key() == -2
    assert pm.dyn_buckets() == [16384]
    assert pm.slot_ctx_keys() == [-1, -2]
    # the main context is found again by use_dyn(its S_pad) while a slot context is active
    pm._dyn_stash[1 << 20] = {"pm": {"dyn_trace": object()}}
    calls = []
    pm.stash_dyn = lambda: calls.append("stash")
    pm._dyn_load = lambda st: calls.append("load")
    pm.use_dyn(-2)  # already active: no-op
    assert calls == []
    pm.use_dyn(1 << 20)
    assert calls == ["stash", "load"] or calls == ["load", "stash"] or len(calls) == 2


def test_slot_prefix_view_shape(monkeypatch):
    got = {}

    class T:
        shape = (4, 1, 288, 128)

    def fake_reshape(t, shape, padded=None):
        got["a"] = (list(shape), list(padded))
        return "view"

    monkeypatch.setattr(ttnn, "reshape", fake_reshape)
    assert PMOD.DSV41PrefillModel.slot_prefix_view(T(), 2) == "view"
    assert got["a"] == ([2, 1, 288, 128], [2, 1, 288, 128])


def test_share_snapshot_collects_halos_and_keys():
    pm = PMOD.DSV41PrefillModel.__new__(PMOD.DSV41PrefillModel)

    class O:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    pa1, pa2, ix1, ix2 = O(halo="h1"), O(halo=None), O(keys="k1"), O(keys=None)
    pm._dyn_objs = lambda: ([pa1, pa2], [], [ix1, ix2])
    sh = pm._share_snapshot()
    assert sh == {"halo": {id(pa1): "h1"}, "keys": {id(ix1): "k1"}}


def test_placement_keeps_prefilling_users_in_low_slots():
    """claim_packed (the placement of the decode buckets) hands out the lowest free in-row index first: a lone prompt needs 1 slot, 4 concurrent prompts one per row still 1, 8 prompts 2, 16 prompts 4."""
    for n_prompts, want in ((1, 1), (2, 1), (4, 1), (5, 2), (8, 2), (9, 3), (16, 4)):
        t = SlotTable(16, 16)
        load, users = {}, []
        for lg in range(n_prompts):
            users.append(t.claim_packed(lg, load, U))
        assert SB.need_slots(users, U) == want, (n_prompts, users)
