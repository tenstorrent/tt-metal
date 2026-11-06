# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the paged APC with fewer prefill slots than users per mesh row (DSV41_APC_PAGED_UP=1, B=16: Up 2, U 4): ``supported`` gate, ``lane_index`` (user -> row of the slot tables
of the stores), the slot tables written by ``PagedApc.commit`` / ``restore`` with a fake model and fake stores (lanes = Up). Run (from the worktree root):
  python_env/bin/python -m pytest --noconftest -c /dev/null --rootdir=/tmp tests/test_apc_paged_up_cpu.py -q"""


from models.demos.blackhole.deepseek_v41_flash.tests import test_apc_paged_model_cpu as T0
from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_model as APM

ROWS, U, UP = 2, 4, 2


class FakeStore:
    def __init__(self):
        self.save, self.load, self.rest = None, None, None
        self.store = {}

    def set_save(self, *a):
        self.save = a[-1]

    def set_load(self, v):
        self.load = v

    def set_restore(self, part, v):
        self.rest = (part, v)

    def save_(self, *a):
        pass


def test_lane_index_identity_when_every_user_has_a_slot():
    assert APM.lane_index(5, 4, 4, {}) == 5
    assert APM.lane_index(5, 4, 4, None) == 5


def test_lane_index_slot_of_user():
    slot_of = {1: 0, 3: 3, 6: 5}
    assert APM.lane_index(1, 4, 2, slot_of) == 0
    assert APM.lane_index(3, 4, 2, slot_of) == 3
    assert APM.lane_index(6, 4, 2, slot_of) == 5
    assert APM.lane_index(2, 4, 2, slot_of) is None  # owns no slot: nothing to save / restore
    assert APM.lane_index(2, 4, 2, None) is None


def test_slot_tables_size_is_rows_times_lanes():
    # rows * Up entries (not rows * U): the tables index the lanes of the prefill slabs
    slot_of = {b: b // U * UP + (b % U) % UP for b in range(ROWS * U)}
    idx = [APM.lane_index(b, U, UP, slot_of) for b in range(ROWS * U)]
    assert max(idx) < ROWS * UP
    # two users of a row share a lane only because their slots were handed over (slot_of is a function of the owner, the allocator keeps owners unique)
    assert idx[0] == idx[2]


class UpModel(T0.FakeModel):
    rows, U, Up = ROWS, U, UP


def test_supported_gate(monkeypatch):
    from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PSG

    monkeypatch.setattr(PSG, "enabled", lambda: True)
    m = UpModel.__new__(UpModel)
    monkeypatch.delenv("DSV41_APC_PAGED_UP", raising=False)
    ok, why = APM.supported(m, 512)
    assert not ok and "DSV41_APC_PAGED_UP" in why
    monkeypatch.setenv("DSV41_APC_PAGED_UP", "1")
    assert APM.supported(m, 512)[0]
    monkeypatch.setenv("DSV41_APC_PAGED_XROW", "1")
    ok, why = APM.supported(m, 512)
    assert not ok and "XROW" in why


# ---- per-wave restore of the hit users (Up < U): the slot assignment the wave loop of prefill_interleaved performs ---------------------------------------
def _slot_alloc_fn():
    """the REAL ``Model._slot_alloc`` (source of tt/dsv41_model.py, no device import)"""
    import ast
    import textwrap
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "tt" / "dsv41_model.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Model")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_slot_alloc")
    ns = {}
    exec(textwrap.dedent(ast.get_source_segment(src, fn)), ns)
    return ns["_slot_alloc"]


class SlotModel:
    rows, U, Up = ROWS, U, UP

    def flush_key_exports(self, users=None, why=None):
        pass


def test_alloc_view_hits_look_like_new_prompts():
    wave = [(0, None, 1024, 2048), (1, None, 0, 600), (2, None, 4096, 9000)]
    v = APM.alloc_view(wave, {2})
    assert v == [(0, None, 1024, 2048), (1, None, 0, 600), (2, None, 0, 9000)]
    assert APM.alloc_view(wave, set()) == wave


def test_three_hit_users_of_one_row_restore_in_two_waves():
    """B=16 shape: one mesh row with 3 hit users and Up 2 slots: wave 1 takes 2 slots, wave 2 the third (evicting a finished owner); every restore lane is the slot of its user"""
    alloc = _slot_alloc_fn()
    m = SlotModel()
    resume = {0: 2048, 1: 1024, 2: 3072}  # the hits' resume positions (set before the waves)
    items = [(0, None, 2048, 4000), (1, None, 1024, 3000), (2, None, 3072, 5000)]
    hit_todo = {0, 1, 2}
    # the wave loop: continuations first, at most Up users of a mesh row per wave
    waves = [items[0:UP], items[UP : 2 * UP]]
    lanes = {}
    for wave in waves:
        slot_of = alloc(m, APM.alloc_view(wave, hit_todo), resume)
        wh = [b for b, *_ in wave if b in hit_todo]
        hit_todo -= set(wh)
        for b in wh:
            lanes[b] = APM.lane_index(b, U, UP, m.pf_slot_of)
            assert lanes[b] == slot_of[b]
            assert 0 <= lanes[b] < ROWS * UP
        owners = m.pf_slot_owner
        assert all(owners[slot_of[b]] == b for b, *_ in wave)
    assert not hit_todo
    assert lanes[0] != lanes[1]  # wave 1: two different slots of the row
    assert lanes[2] in (lanes[0], lanes[1])  # wave 2: reuses a slot (the owner of it lost its carried state)
    assert m.pf_slot_of.get(2) == lanes[2]


def test_hit_user_keeps_its_own_slot_when_it_has_one():
    alloc = _slot_alloc_fn()
    m = SlotModel()
    resume = {}
    s1 = alloc(m, [(1, None, 0, 700)], resume)  # a cold prompt of user 1 took a slot earlier
    s2 = alloc(m, APM.alloc_view([(1, None, 1024, 3000)], {1}), resume)
    assert s1 == s2


def test_store_sizes_per_bank():
    # B=8 default stores (the measured ~66 MiB/bank) and the B=16 sizing recommendation: stores are per mesh row, independent of U
    assert abs(APM.store_mib_per_bank(128, 48, 2048) - 66.25) < 0.1
    assert APM.store_mib_per_bank(80, 24, 1024) < 40.0
    assert APM.idle_pages_mib_per_bank(4) < 0.2  # the idle pages of 4 users are negligible
