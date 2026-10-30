# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device) tests of the vLLM adapter's bookkeeping: the slot / column placement
(``tt/vllm_config.py::SlotTable``), the logical block table, the prefill plan, the memory-planner gate, and
``tt/generator_vllm.py`` driven end to end with a fake mesh model that mimics ``GLM53MeshModel``'s API on torch
tensors (token / position routing onto the physical user slots, output un-permutation, chunked-prefill
contiguity, slot remaps, the v1 reload commands).

    PYTHONPATH=<worktree>:<tt-metal>/ttnn:<tt-metal> <tt-metal>/python_env/bin/pytest \\
        models/demos/glm53_flash/tests/test_generator_vllm_host.py -p no:cacheprovider -q
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.generator_vllm import GLM53DecodeOutput, GLM53FlashForCausalLM
from models.demos.glm53_flash.tt.memory_plan import MemoryPlan
from models.demos.glm53_flash.tt.packed_inputs import RESIDENT_ENV
from models.demos.glm53_flash.tt.vllm_config import (
    DEFAULT_WAVE_ROWS,
    REPLAY_BLOCKING_ENV,
    LogicalBlockTable,
    PrefillPlanner,
    ServingGeometry,
    SlotTable,
    WaveCall,
    WaveMember,
    WavePlanner,
    greedy_rows,
    max_tokens_all_users,
    prefill_buckets,
)

VOCAB = 64
PAD = 3


@contextlib.contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


def test_module_imports_without_ttnn():
    """The adapter module must be importable on a host without ttnn (every ttnn use is lazy)."""
    assert "models.demos.glm53_flash.tt.generator_vllm" in sys.modules
    assert "ttnn" not in sys.modules, "importing the adapter must not import ttnn"
    # the env-knob readers only the device path calls (a missing local import in one of them killed the 2026-10-30
    # served boot at the decode warm-up while every host suite was green): called here with and without an env
    from models.demos.glm53_flash.tt.packed_inputs import gate_page, resident_decode_enabled, slot_mask_enabled

    assert slot_mask_enabled({"GLM53_DECODE_SLOT_MASK": "1"}) and not slot_mask_enabled({})
    assert resident_decode_enabled({"GLM53_RESIDENT_DECODE": "on"}) and not resident_decode_enabled({})
    assert isinstance(slot_mask_enabled(), bool) and isinstance(resident_decode_enabled(), bool)
    # the KDA write-gate pages: one page per mesh row (its users in the first words), two pages per row past 32 users
    page = gate_page([True, False, True, True, False, False, True, True], rows=2, users_per_row=4)
    assert (
        tuple(page.shape) == (2, 32) and page[0, :4].tolist() == [1, 0, 1, 1] and page[1, :4].tolist() == [0, 0, 1, 1]
    )
    assert int(page[:, 4:].sum()) == 0 and tuple(gate_page([True] * 80, rows=2, users_per_row=40).shape) == (4, 32)
    assert GLM53FlashForCausalLM.decode_input_update_contract == 1
    caps = GLM53FlashForCausalLM.model_capabilities
    assert caps["supports_async_decode"] is False and caps["supports_prefix_caching"] is False
    assert caps["supports_sample_on_device"] is True and caps["max_device_top_k"] == 1
    assert caps["supports_chunked_prefill"] is True


# ----------------------------------------------------------------------------------------- geometry
def test_geometry_columns_and_capacity():
    g = ServingGeometry(rows=8, cols=4, users_per_row=8, max_seq_len=131072)
    assert g.users_per_chip == 2 and g.physical_slots == 64 and g.request_capacity == 8
    assert [g.column_of(u) for u in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert list(g.users_of_column(2)) == [4, 5]
    assert g.flat(3, 5) == 3 * 8 + 5
    assert g.blocks_per_seq == 4096 and g.chunk_unit == 256
    g.check_max_num_seqs(8)
    with expect_error(ValueError, "request capacity"):
        g.check_max_num_seqs(64)
    gi = ServingGeometry(rows=8, cols=4, users_per_row=8, max_seq_len=131072, rows_independent=True)
    assert gi.request_capacity == 64
    with expect_error(ValueError, "users_per_row"):
        ServingGeometry(rows=8, cols=4, users_per_row=6, max_seq_len=1024)
    rep = ServingGeometry(rows=2, cols=4, users_per_row=6, max_seq_len=1024, partitioned=False)
    assert rep.users_per_chip == 6 and rep.column_of(5) == 0
    with expect_error(ValueError, "device_prefill_chunk"):
        ServingGeometry(rows=8, cols=4, users_per_row=4, max_seq_len=1024, device_prefill_chunk=100)


def test_memory_plan_gate_matches_the_envelope():
    """The planner's partitioned / no-scratch envelope (tests/test_memory_plan.py): 4 users per row fit 1 M, 8 do not."""
    cfg = GLM53FlashConfig()  # config defaults: 11 DSA + 34 KDA layers, 288 experts
    plan = MemoryPlan.estimate(cfg, 8, 4, partition=True, scratch_slot=False)
    ServingGeometry(rows=8, cols=4, users_per_row=4, max_seq_len=1048576).check_memory_plan(plan)
    with expect_error(ValueError, "do not fit"):
        ServingGeometry(rows=8, cols=4, users_per_row=8, max_seq_len=1048576).check_memory_plan(plan)
    ServingGeometry(rows=8, cols=4, users_per_row=28, max_seq_len=131072).check_memory_plan(plan)  # 28 since 2026-10-27
    assert max_tokens_all_users(32, 131072) == 32 * 131072


# ---------------------------------------------------------------------------------------- slot table
def test_slot_table_balances_columns():
    g = ServingGeometry(rows=8, cols=4, users_per_row=8, max_seq_len=4096)
    t = SlotTable(g)
    users = [t.allocate(i).user for i in range(5)]
    # one user per column first (lowest free user of the least-loaded column), then the second user of column 0
    assert users == [0, 2, 4, 6, 1]
    assert t.column_loads() == [2, 1, 1, 1]
    assert all(t.physical(i).row == 0 for i in range(5))  # the rows replicate: row 0 is the read copy
    t.release(1)  # frees user 2 (column 1)
    assert t.column_loads() == [2, 0, 1, 1]
    assert t.allocate(5).user == 2  # column 1 is the least loaded again
    assert [t.allocate(i).user for i in (6, 7)] == [3, 5]  # the second users of columns 1 and 2
    assert t.column_loads() == [2, 2, 2, 1]
    assert t.allocate(1).user == 7 and t.column_loads() == [2, 2, 2, 2]
    with expect_error(ValueError, "outside"):
        t.allocate(8)
    # a leaked physical slot (the table and the device disagree) fails loudly instead of placing a prompt on a live user
    small = SlotTable(ServingGeometry(rows=1, cols=2, users_per_row=2, max_seq_len=64))
    small.allocate(0)
    small._occupied.add((0, 1))
    with expect_error(RuntimeError, "no free physical slot"):
        small.allocate(1)


def test_slot_table_rows_independent_placement():
    # phase 2 affinity ("row"): column balance first (user 1 lives on column 1), then row balance, then the lowest indices
    g = ServingGeometry(rows=2, cols=2, users_per_row=2, max_seq_len=64, rows_independent=True, wave_affinity="row")
    t = SlotTable(g)
    placed = [(s.row, s.user) for s in (t.allocate(i) for i in range(4))]
    assert placed == [(0, 0), (1, 1), (1, 0), (0, 1)]
    assert t.flat_indices(2) == [g.flat(1, 0)] and t.read_index(2) == 2
    assert t.column_loads() == [2, 2] and t.row_loads() == [2, 2]
    # phase 1 affinity ("user", the default): one slot index across every row before the next, so a burst of
    # `rows` arrivals shares the slot index (one wave); consecutive indices alternate the columns
    gu = ServingGeometry(rows=2, cols=2, users_per_row=4, max_seq_len=64, rows_independent=True)
    assert gu.wave_user_order() == [0, 2, 1, 3]  # column 0's first index, column 1's first, then their seconds
    tu = SlotTable(gu)
    placed = [(s.row, s.user) for s in (tu.allocate(i) for i in range(6))]
    assert placed == [(0, 0), (1, 0), (0, 2), (1, 2), (0, 1), (1, 1)]
    assert tu.column_loads() == [4, 2] and tu.row_loads() == [3, 3]
    tu.release(1)  # (1, 0) frees: the next arrival takes it (the lowest index with a free row)
    s6 = tu.allocate(6)
    assert (s6.row, s6.user) == (1, 0)
    # a prompt no traced wave holds (prefer_sp: it prefills eagerly) takes a slot index live on no row first, on the
    # lowest-load row, and stays SP-eligible (a lone eager 16 K chunk costs what one 2048-row wave costs); once every
    # index is live somewhere the wave-affine order resumes
    ts = SlotTable(gu)
    placed = [(s.row, s.user) for s in (ts.allocate(i, prefer_sp=True) for i in range(6))]
    assert placed == [(0, 0), (1, 2), (0, 1), (1, 3), (1, 0), (0, 2)]
    assert all(ts.sp_eligible(i) for i in range(2, 4)) and not any(ts.sp_eligible(i) for i in (0, 1, 4, 5))
    assert ts.column_loads() == [3, 3] and ts.row_loads() == [3, 3]
    ts.release(1)  # (1, 2) frees but index 2 stays live on row 0 (request 5): the wave-affine order -> (1, 2), waves
    s6 = ts.allocate(6, prefer_sp=True)
    assert (s6.row, s6.user) == (1, 2) and not ts.sp_eligible(6)
    ts.release(5)  # (0, 2) frees too: index 2 is live on no row, so the next eager arrival is SP-eligible there
    ts.release(6)
    s7 = ts.allocate(7, prefer_sp=True)
    assert (s7.row, s7.user) == (0, 2) and ts.sp_eligible(7)  # rows tie at 2 live each: the lowest row
    # the 8x4 at 1 M: 4 users per row, one per column -> the first 8 arrivals land on slot index 0 of rows 0..7; eager
    # prompts land on the diagonal first (four SP-eligible long prompts), then on slot index 0 of the rows free of it
    g8 = ServingGeometry(rows=8, cols=4, users_per_row=4, max_seq_len=1048576, rows_independent=True)
    assert g8.wave_user_order() == [0, 1, 2, 3] and g8.request_capacity == 32 and g8.chunk_unit == 128
    t8 = SlotTable(g8)
    assert [(s.row, s.user) for s in (t8.allocate(i) for i in range(9))] == [(r, 0) for r in range(8)] + [(0, 1)]
    t8 = SlotTable(g8)
    assert [(s.row, s.user) for s in (t8.allocate(i, prefer_sp=True) for i in range(9))] == [
        (r, r) for r in range(4)
    ] + [(r, 0) for r in range(1, 6)]
    # without the SP fallback the diagonal buys nothing, and sp_first=False (GLM53_VLLM_SP_FIRST=0, the policy A/B)
    # switches it off: the wave-affine order from the first eager arrival
    for g_off in (dataclasses.replace(g8, sp_fallback=False), dataclasses.replace(g8, sp_first=False)):
        t8w = SlotTable(g_off)
        assert [(s.row, s.user) for s in (t8w.allocate(i, prefer_sp=True) for i in range(9))] == [
            (r, 0) for r in range(8)
        ] + [(0, 1)]
    # the non-partitioned layout: plain index order
    assert ServingGeometry(rows=2, cols=4, users_per_row=3, max_seq_len=64, partitioned=False).wave_user_order() == [
        0,
        1,
        2,
    ]
    with expect_error(ValueError, "wave_affinity"):
        ServingGeometry(rows=2, cols=2, users_per_row=2, max_seq_len=64, wave_affinity="column")
    with expect_error(ValueError, "wave_rows"):
        ServingGeometry(rows=2, cols=2, users_per_row=2, max_seq_len=64, wave_rows=1024)


def test_sp_eligibility_is_exact_only_while_the_other_rows_are_free():
    g = ServingGeometry(rows=4, cols=2, users_per_row=2, max_seq_len=4096, rows_independent=True)
    t = SlotTable(g)
    a = t.allocate(0)  # (0, 0)
    assert (a.row, a.user) == (0, 0) and t.sp_eligible(0)
    b = t.allocate(1)  # (1, 0): slot index 0 is live on another row -> neither may replicate into it
    assert (b.row, b.user) == (1, 0) and not t.sp_eligible(0) and not t.sp_eligible(1)
    c = t.allocate(2)  # (2, 0)
    t.release(1)
    t.release(2)
    assert t.sp_eligible(0)  # free again on every other row
    a.sp_only = False  # a wave chunk ran: the prefix exists on row 0 only, SP is never exact again
    assert not t.sp_eligible(0)
    # policy off / replicated rows
    tp = SlotTable(dataclasses.replace(g, sp_fallback=False))
    tp.allocate(0)
    assert not tp.sp_eligible(0)
    tr = SlotTable(ServingGeometry(rows=4, cols=2, users_per_row=2, max_seq_len=4096))
    tr.allocate(0)
    tr.allocate(1)
    assert tr.sp_eligible(0) and tr.sp_eligible(1)
    with expect_error(RuntimeError, "holds no request"):
        t.sp_eligible(3)
    del c


def test_wave_planner_groups_walks_and_traces():
    g = ServingGeometry(
        rows=4,
        cols=2,
        users_per_row=2,
        max_seq_len=65536,
        rows_independent=True,
        traced_prefill_lengths=(256, 2048),
        traced_wave_lengths=(256, 2048),  # the per-row wave lengths decide a wave's trace, not the SP lengths
    )
    t = SlotTable(g)
    pp = PrefillPlanner(g)
    wp = WavePlanner(g, t, pp)
    assert [(s.row, s.user) for s in (t.allocate(i) for i in range(4))] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    e = t.allocate(6)  # slot index 0 is full on every row -> (0, 1): alone on its index -> SP-eligible
    assert (e.row, e.user) == (0, 1) and t.sp_eligible(6) and not t.sp_eligible(0)
    # logical 3 (row 3) is live but has no chunk this step: row 3 idles in the waves of slot index 0
    steps = wp.plan([(0, 0, 5000), (1, 0, 1000), (6, 2048, 3000), (2, 2048, 2100)])
    kinds = [type(s).__name__ for s in steps]
    assert kinds == ["SPStep", "WaveStep", "WaveStep", "WaveStep"]
    sp = steps[0]
    assert sp.logical == 6 and sp.plan.kind == "eager" and sp.plan.chunks == ((2048, 3000),)
    w1, w2, w3 = (s.call for s in steps[1:])
    # wave 1: the first wave_rows chunk of every member of slot index 0 (rows 0, 1, 2), padded to the longest
    assert [(m.logical, m.row, m.start, m.end) for m in w1.members] == [
        (0, 0, 0, 2048),
        (1, 1, 0, 1000),
        (2, 2, 2048, 2100),
    ]
    assert w1.user == 0 and w1.t_pad == 2048 and w1.traced_length is None  # member 2 starts at 2048: eager
    assert w1.idle_rows == [3] and w1.member_of_row(1).logical == 1 and w1.member_of_row(3) is None
    # waves 2 and 3: only request 0 has tokens left; the last chunk is padded to a tile multiple
    assert [(m.logical, m.start, m.end) for m in w2.members] == [(0, 2048, 4096)] and w2.t_pad == 2048
    assert [(m.logical, m.start, m.end) for m in w3.members] == [(0, 4096, 5000)] and w3.t_pad == 928
    # a wave of fresh prompts within a captured per-row length is traced at that length
    assert WavePlanner(g, t, pp, wave_captured=lambda T: False).plan([(0, 0, 200)])[0].call.traced_length is None
    # an SP-eligible short fresh prompt without an SP trace takes the traced wave; a long one the eager SP chunks
    g_nosp = dataclasses.replace(g, traced_prefill_lengths=())
    t3 = SlotTable(g_nosp)
    p3 = WavePlanner(g_nosp, t3, PrefillPlanner(g_nosp))
    for i in range(4):
        t3.allocate(i)
    t3.allocate(6)  # (0, 1): alone on its index
    assert t3.sp_eligible(6)
    alone = p3.plan([(6, 0, 200)])
    assert type(alone[0]).__name__ == "WaveStep" and alone[0].call.traced_length == 256 and alone[0].call.user == 1
    assert type(p3.plan([(6, 0, 3000)])[0]).__name__ == "SPStep"  # beyond the wave traces: the 4 rows share the chunks
    assert type(p3.plan([(6, 0, 200)], enable_trace=False)[0]).__name__ == "SPStep"
    tr = wp.plan([(0, 0, 300), (1, 0, 100)])
    assert len(tr) == 1 and tr[0].call.traced_length == 2048 and tr[0].call.t_pad == 320
    assert wp.plan([(0, 0, 300), (1, 0, 100)], enable_trace=False)[0].call.traced_length is None
    assert wp.plan([(0, 0, 200)])[0].call.traced_length == 256
    # phase 2 ("row" affinity): one group, a row serves its requests one after the other
    gr = dataclasses.replace(g, wave_affinity="row", traced_wave_lengths=())
    t2 = SlotTable(gr)
    p2 = WavePlanner(gr, t2, PrefillPlanner(gr))
    placed = [(s.row, s.user) for s in (t2.allocate(i) for i in range(6))]
    assert placed == [(0, 0), (1, 1), (2, 0), (3, 1), (1, 0), (0, 1)]  # column balance, then row balance
    for logical in range(6):
        t2.physical(logical).sp_only = False
    # rows 0 and 1 hold two pending requests each: a row serves them one after the other (the k-th wave takes the
    # head of every row's queue), members may sit on different slot indices
    waves = [s.call for s in p2.plan([(0, 0, 3000), (1, 0, 100), (5, 0, 2048), (4, 0, 64)])]
    assert all(c.user is None for c in waves)
    assert [[(m.logical, m.row, m.user) for m in c.members] for c in waves] == [
        [(0, 0, 0), (1, 1, 1)],
        [(0, 0, 0), (4, 1, 0)],
        [(5, 0, 1)],
    ]
    assert [c.t_pad for c in waves] == [2048, 960, 2048]
    with expect_error(ValueError, "off the 128-token grid"):  # r3's '(2, 0, 224)': a KV-block end short of the grid
        wp.plan([(0, 224, 4000), (1, 0, 4000)])
    # a multimodal prompt (sp_required) never takes a wave: alone on its index it is an SP step even within a captured
    # wave length; with another request on its index it is refused with the cause
    t.release(6)
    mm = t.allocate(7, prefer_sp=True)
    mm.sp_required = True
    (mm_step,) = wp.plan([(7, 0, 200)])
    assert t.sp_eligible(7) and type(mm_step).__name__ == "SPStep"
    assert mm_step.plan.kind == "eager"  # the vision features are scattered into an eager chunk; a text SP step traces
    mm.sp_required = False
    assert wp.plan([(7, 0, 200)])[0].plan.kind == "traced"
    t.release(7)
    t.allocate(5)  # the wave-affine order: index 0 is full, so (0, 1) ...
    t.allocate(7)  # ... and (1, 1): index 1 now lives on another row, no SP for a multimodal prompt here
    t.physical(7).sp_required = True
    assert not t.sp_eligible(7)
    with expect_error(NotImplementedError, "sequence-parallel prefill"):
        wp.plan([(7, 0, 200)])
    with expect_error(ValueError, "appears twice"):
        p2.plan([(0, 0, 64), (0, 64, 128)])
    with expect_error(RuntimeError, "holds no request"):
        p2.plan([(7, 0, 64)])
    with expect_error(ValueError, "at most one per row"):
        WaveCall(rows=2, t_pad=64, members=(WaveMember(0, 0, 0, 0, 64), WaveMember(1, 0, 1, 0, 64)))
    with expect_error(ValueError, "exceeds t_pad"):
        WaveCall(rows=2, t_pad=32, members=(WaveMember(0, 0, 0, 0, 64),))
    with expect_error(ValueError, "slot index"):
        WaveCall(rows=2, t_pad=64, members=(WaveMember(0, 0, 1, 0, 64),), user=0)


def test_slot_table_remap_and_moves():
    g = ServingGeometry(rows=2, cols=4, users_per_row=4, max_seq_len=64)
    t = SlotTable(g)
    phys = [t.allocate(i).user for i in range(3)]  # users 0, 1, 2 (one per column)
    assert phys == [0, 1, 2]
    t.release(1)
    # the plugin condenses: row 1 now reads the state of slot 2 -> remap [0, 2, 1, 3]
    moves = t.apply_remap([0, 2, 1, 3])
    assert moves == {2: 1, 1: 2}
    assert t.physical(1).user == 2 and t.physical(2) is None
    # the same permutation in the plugin's old -> new form reproduces the table
    t2 = SlotTable(g)
    for i in range(3):
        t2.allocate(i)
    t2.release(1)
    t2.apply_moves({2: 1, 1: 2})
    assert [None if s is None else s.user for s in t2._phys] == [None if s is None else s.user for s in t._phys]
    with expect_error(ValueError, "permutation"):
        t.apply_remap([0, 0, 1, 3])
    # device feed: the rows replicate, so a request's token lands at its user index of EVERY row
    ids, pos = t.device_feed({0: 11, 1: 22}, {0: 5, 1: 9})
    assert ids == [11, 0, 22, 0, 11, 0, 22, 0] and pos == [5, 0, 9, 0, 5, 0, 9, 0]
    with expect_error(ValueError, "outside"):
        t.device_feed({0: 1}, {0: 64})


def test_logical_block_table():
    g = ServingGeometry(rows=1, cols=1, users_per_row=1, max_seq_len=256, partitioned=False)
    bt = LogicalBlockTable(g)
    assert bt.blocks_needed(0) == 0 and bt.blocks_needed(1) == 1 and bt.blocks_needed(33) == 2
    assert bt.check_row(torch.tensor([7, 3, 9, 0, 0, 0, 0, 0]), 70) == 3
    with expect_error(ValueError, "cannot hold"):
        bt.check_row([1, 2], 100)
    with expect_error(ValueError, "repeats"):
        bt.check_row([4, 4, 5], 70)
    with expect_error(ValueError, "negative"):
        bt.check_row([4, -1, 5], 70)
    with expect_error(ValueError, "outside"):
        bt.check_row([1] * 8, 257)


def test_prefill_planner_traced_vs_eager():
    g = ServingGeometry(
        rows=8, cols=4, users_per_row=4, max_seq_len=65536, traced_prefill_lengths=(256, 512, 1024, 2048)
    )
    p = PrefillPlanner(g)
    traced = p.plan(0, 300)
    assert traced.kind == "traced" and traced.padded_length == 512 and traced.chunks == ((0, 300),)
    assert p.plan(0, 2048).padded_length == 2048
    assert p.plan(0, 2049).kind == "eager"
    assert p.plan(0, 300, enable_trace=False).kind == "eager"
    long = p.plan(0, 40000)
    assert long.kind == "eager" and long.chunks == ((0, 16384), (16384, 32768), (32768, 40000))
    cont = p.plan(16384, 20000)
    assert cont.kind == "eager" and cont.chunks == ((16384, 20000),)
    with expect_error(ValueError, "multiple of 32"):
        p.plan(100, 200)
    with expect_error(ValueError, "empty"):
        p.plan(256, 256)
    with expect_error(ValueError, "max_seq_len"):
        p.plan(0, 65537)
    # a captured-length check that says "not captured" falls back to eager
    assert PrefillPlanner(g, traced_available=lambda T: False).plan(0, 300).kind == "eager"


def test_greedy_rows_admission():
    greedy_rows(SimpleNamespace(temperature=[0.0, 0.7], top_k=[-1, 1]), [0, 1])
    with expect_error(ValueError, "greedy"):
        greedy_rows(SimpleNamespace(temperature=[0.0, 0.7], top_k=[-1, 40]), [0, 1])


# ------------------------------------------------------------------------------------- fake model
def _next_token(tok: int, pos: int) -> int:
    return (tok * 7 + pos) % VOCAB


def _first_token(user: int, ids: torch.Tensor) -> int:
    return (int(ids.sum()) + user) % VOCAB


class _FakeMesh:
    def __init__(self, model):
        self.m = model
        self.rows, self.cols = model.rows, model.cols

    def device_tensors(self, handle):
        kind = handle[0]
        if kind in ("logits_tr", "logits_eager"):
            B = self.m.max_batch
            out = []
            for r in range(self.rows):
                for _c in range(self.cols):
                    out.append(self.m._last_logits[r * B : (r + 1) * B].reshape(1, 1, B, VOCAB).to(torch.bfloat16))
            return out
        if kind == "logits_last":  # eager prefill: [1,1,1,V] replicated on every chip
            t = torch.zeros(1, 1, 1, VOCAB)
            t[0, 0, 0, handle[1]] = 1.0
            return [t.clone() for _ in range(self.rows * self.cols)]
        if kind == "logits_tile":  # traced prefill: [1,1,32,V], the prompt's token at row handle[2]
            t = torch.zeros(1, 1, 32, VOCAB)
            t[0, 0, handle[2], handle[1]] = 1.0
            return [t.clone() for _ in range(self.rows * self.cols)]
        raise KeyError(handle)


class _FakeMeshModel:
    """``GLM53MeshModel``'s serving surface on torch tensors with deterministic token functions."""

    wave_per_row_users = False  # phase 1: every member of a wave shares the slot index

    def __init__(self, rows: int, cols: int, max_batch: int, max_seq_len: int, traced=(256, 512, 1024, 2048)):
        self.rows, self.cols = rows, cols
        self.max_batch = max_batch
        self.total_users = rows * max_batch
        self.max_seq_len = max_seq_len
        self.cfg = SimpleNamespace(vocab_size=VOCAB, pad_token_id=PAD)
        self.device = None
        self.mesh = _FakeMesh(self)
        self.layers = []
        self.ids_dev, self.positions = ("ids_dev",), ("positions",)
        self.calls: list[tuple] = []
        self._traced = tuple(traced)
        self._captured: list[int] = []
        self._ready_waves: list[int] = []  # prepared (allocated + compiled) wave lengths
        self._captured_waves: list[int] = []  # captured per-row wave lengths
        self.trace_id = None
        self._fed_ids = None
        self._fed_pos = None
        self._last_ids = torch.zeros(self.total_users, dtype=torch.long)
        self._unread: list[torch.Tensor] = []  # the ids of every stepped-but-unread step, oldest first (the CQ order)
        self._last_logits = torch.zeros(self.total_users, VOCAB)
        self.next_start = [0] * max_batch  # the chunked-prefill cursor per slot (``rewind_user``)
        # GLM53_RESIDENT_DECODE: the step feeds its own ids back and advances the live positions (the device side)
        self.resident = os.environ.get(RESIDENT_ENV, "0") == "1"
        self.supports_slot_mask = False  # GLM53MeshModel: the live mask gates the KDA state writes (parked slots)
        self._live: list[bool] | None = None

    # trace management
    def trace_prefill_lengths(self):
        return list(self._traced)

    def has_prefill_trace(self, T):
        return T in self._captured

    def capture_prefill(self, lengths=None):
        self.calls.append(("capture_prefill",))
        new = [T for T in (lengths or self._traced) if T not in self._captured]
        self._captured += new
        return new

    # waves: the per-row lengths within the dense range, prepared (before any capture) then captured
    def wave_trace_lengths(self):
        return [T for T in self._traced if T <= 2048]

    def has_wave_trace(self, T):
        return T in self._captured_waves

    def prepare_waves(self, lengths=None):
        self.calls.append(("prepare_waves",))
        new = [T for T in (lengths or self.wave_trace_lengths()) if T not in self._ready_waves + self._captured_waves]
        self._ready_waves += new
        return new

    def capture_waves(self, lengths=None):
        self.calls.append(("capture_waves",))
        new = [T for T in (lengths or self.wave_trace_lengths()) if T not in self._captured_waves]
        assert all(T in self._ready_waves for T in new), "the served warm-up prepares every wave before the captures"
        self._ready_waves = [T for T in self._ready_waves if T not in new]
        self._captured_waves += new
        return new

    def warmup_decode(self, ids, positions):
        self.calls.append(("warmup_decode", len(positions)))

    def capture(self, ids, positions):
        self.calls.append(("capture", len(positions)))
        self.trace_id = 7
        return 7, (("streams",), ("logits_tr",), ("ids_tr",))

    def reset_kda_states(self):
        self.calls.append(("reset_kda_states",))

    def release_prefill_traces(self):
        self._captured = []

    # decode
    def load_ids(self, ids):
        assert ids.numel() == self.total_users
        self._fed_ids = ids.clone()

    def set_positions(self, positions):
        assert len(positions) == self.total_users
        self._fed_pos = list(positions)

    def load_decode_inputs(self, ids, positions, live=None):
        """The one-call input load (``GLM53MeshModel.load_decode_inputs``: one packed upload with the knob; the
        resident chain's transition carries the live mask)."""
        self.calls.append(("load_decode_inputs", len(positions)))
        self._unread.clear()  # a reload follows the plugin's drain: every earlier read landed
        self.load_ids(ids)
        self.set_positions(positions)
        self._live = None if live is None else [bool(v) for v in live]

    def advance_positions_host(self, live=None):
        self.calls.append(("advance_positions_host", tuple(bool(v) for v in (live or ()))))

    def _step(self):
        self.calls.append(("decode", self._fed_ids.tolist(), list(self._fed_pos)))
        for s in range(self.total_users):
            nxt = _next_token(int(self._fed_ids[s]), int(self._fed_pos[s]))
            self._last_ids[s] = nxt
            self._last_logits[s] = 0.0
            self._last_logits[s, nxt] = 1.0
        self._unread.append(self._last_ids.clone())
        if self.resident:  # the in-graph feedback: the next step embeds these ids at the advanced live positions
            self._fed_ids = self._last_ids.clone()
            live = [True] * self.total_users if self._live is None else self._live
            self._fed_pos = [p + (1 if v else 0) for p, v in zip(self._fed_pos, live)]

    def replay(self, trace_id, blocking=True):
        assert trace_id == self.trace_id
        self.calls.append(("replay", blocking))
        self._step()

    def decode_step(self, ids_dev, positions):
        self._step()
        return ("streams",), ("logits_eager",), ("ids_eager",)

    def ids_to_host(self, ids):
        """A read drains the oldest unread step (the command queue's order): a deferred read lands the ids of ITS
        step, whatever stepped after it."""
        return self._unread.pop(0) if self._unread else self._last_ids.clone()

    # prefill
    def rewind_user(self, user, position):
        """``GLM53MeshModel.rewind_user``: the served warm-up walks continued chunks into every slot at every prefix
        bucket and rewinds the slot's cursor around each (``_warmup_prefix_starts``)."""
        if position < 0 or position % 32:
            raise ValueError(f"rewind position {position} must be a non-negative multiple of 32")
        self.calls.append(("rewind_user", user, int(position)))
        self.next_start[user] = int(position)

    def prefill(self, ids, user=0, start=0):
        S = int(ids.numel())
        self.calls.append(("prefill", user, start, S))
        self.next_start[user] = start + S
        first = _first_token(user, ids)
        unit = 32 * self.rows
        T_loc = (-(-S // unit) * unit) // self.rows
        owner_row = (S - 1) // T_loc
        return ("streams",), ("logits_last", first), owner_row

    def prefill_traced(self, ids, user=0):
        S = int(ids.numel())
        T = min(t for t in self._captured if t >= S)
        self.calls.append(("prefill_traced", user, S, T))
        first = _first_token(user, ids)
        return SimpleNamespace(
            streams=("streams",),
            logits=("logits_tile", first, (S - 1) % 32),
            ids=None,
            owner_row=0,
            row=(S - 1) % 32,
            S=S,
            T=T,
        )

    def first_token(self, logits_or_res, owner_row=0, row=0):
        if isinstance(logits_or_res, tuple):
            return logits_or_res[1]
        return logits_or_res.logits[1]

    # waves (per-row prefill ownership): one chunk of a different prompt per row, the model answers per row
    def _wave(self, kind: str, call: WaveCall):
        assert len(call.members) >= 1 and all(r in call.ids for r in (m.row for m in call.members))
        assert call.user is None or all(m.user == call.user for m in call.members)
        self.calls.append(
            (kind, call.user, call.t_pad, call.traced_length, [(m.row, m.user, m.start, m.end) for m in call.members])
        )
        out = {}
        for m in call.members:
            ids = call.ids[m.row]
            assert int(ids.numel()) == m.end - m.start <= call.t_pad
            first = _first_token(m.user, ids)
            logits = torch.zeros(VOCAB)
            logits[first] = 1.0
            out[m.row] = (first, logits)
        return out

    def prefill_wave(self, call: WaveCall):
        return self._wave("prefill_wave", call)

    def prefill_traced_wave(self, call: WaveCall):
        assert call.traced_length in self._captured_waves and all(m.start == 0 for m in call.members)
        return self._wave("prefill_traced_wave", call)


class _FakeMeshModelNoWaves(_FakeMeshModel):
    """A mesh model without the row-local wave prefill (today's device side)."""

    prefill_wave = None
    prefill_traced_wave = None


def _adapter(
    rows=2,
    cols=4,
    users_per_row=4,
    max_seq_len=65536,
    warm=True,
    independent=False,
    model_cls=_FakeMeshModel,
    wave_rows=DEFAULT_WAVE_ROWS,
):
    model = model_cls(rows, cols, users_per_row, max_seq_len)
    g = ServingGeometry(
        rows=rows,
        cols=cols,
        users_per_row=users_per_row,
        max_seq_len=max_seq_len,
        traced_prefill_lengths=tuple(model.trace_prefill_lengths()),
        pad_token_id=PAD,
        rows_independent=independent,
        wave_affinity="row" if model.wave_per_row_users else "user",
        wave_rows=wave_rows,
    )
    a = GLM53FlashForCausalLM(model, g)
    if warm:  # the plugin's order: eager prefill, eager decode, traced prefill, traced decode
        n = g.request_capacity
        a.warmup_model_prefill(enable_trace=False, can_sample_on_device=True)
        a.warmup_model_decode(enable_trace=False, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
        a.warmup_model_prefill(enable_trace=True, can_sample_on_device=True)
        a.warmup_model_decode(enable_trace=True, max_batch_size=n, num_blocks=1, can_sample_on_device=True)
    return a, model


def _greedy(n):
    return SimpleNamespace(temperature=[0.0] * n, top_k=[1] * n)


# ------------------------------------------------------------------------------------- adapter e2e
def test_warmup_order_and_decode_bucket():
    a, m = _adapter()
    kinds = [c[0] for c in m.calls]
    # persistent decode inputs + eager step first, one eager prefill, the prefill traces, then the decode trace
    assert (
        kinds.index("warmup_decode") < kinds.index("prefill") < kinds.index("capture_prefill") < kinds.index("capture")
    )
    assert a.tt_supported_decode_batch_sizes == (4,) and a._decode_warmup_complete and a.trace_id == 7
    assert a.already_warmed_up_prefill and m.calls.count("capture") == 0  # tuples, not strings
    with expect_error(ValueError, "request capacity"):
        a.warmup_model_decode(enable_trace=True, max_batch_size=8, num_blocks=1, can_sample_on_device=True)


def test_prefill_routes_prompts_to_columns_and_scores_them():
    a, m = _adapter()
    toks = torch.zeros(2, 300, dtype=torch.int32)
    toks[0, :300] = torch.arange(300) % 50
    toks[1, :100] = torch.arange(100) % 20 + 1
    logits = a.prefill_forward(
        toks,
        page_table=torch.arange(2 * 2048).reshape(2, 2048),
        prompt_lens=[300, 100],
        start_pos=torch.tensor([0, 0]),
        empty_slots=[0, 1],
        enable_trace=True,
    )
    assert logits.shape == (2, 1, VOCAB)
    # the traced path for both (<= 2048 tokens), distinct columns, the prompt's token out of the tile row
    tr = [c for c in m.calls if c[0] == "prefill_traced"]
    assert [(c[1], c[2], c[3]) for c in tr] == [(0, 300, 512), (1, 100, 256)]
    assert int(logits[0, 0].argmax()) == _first_token(0, toks[0, :300].long())
    assert int(logits[1, 0].argmax()) == _first_token(1, toks[1, :100].long())
    assert a.slots.physical(0).user == 0 and a.slots.physical(1).user == 1 and a.slots.column_loads() == [1, 1, 0, 0]
    assert a.slots.physical(0).prefilled_to == 300 and a.slots.physical(0).phase == "prefill"
    # device sampling returns the greedy ids
    toks2 = torch.full((1, 40), 5, dtype=torch.int32)
    out, lp = a.prefill_forward(
        toks2, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[2], sampling_params=_greedy(1)
    )
    assert lp is None and out.shape == (1, 1) and int(out[0, 0]) == _first_token(2, toks2[0].long())
    assert a.slots.physical(2).user == 2
    with expect_error(ValueError, "greedy"):
        a.prefill_forward(
            toks2,
            prompt_lens=[40],
            start_pos=torch.tensor([0]),
            empty_slots=[3],
            sampling_params=SimpleNamespace(temperature=[0.9], top_k=[50]),
        )


def test_decode_permutes_rows_onto_physical_slots_and_back():
    a, m = _adapter()
    for slot, n in ((0, 300), (1, 100), (2, 40)):
        t = torch.full((1, n), 2 + slot, dtype=torch.int32)
        a.prefill_forward(
            t, prompt_lens=[n], start_pos=torch.tensor([0]), empty_slots=[slot], sampling_params=_greedy(1)
        )
    # a padded decode: 3 live rows, row 3 padding (position -1)
    tokens = torch.tensor([[11], [22], [33], [0]], dtype=torch.int32)
    positions = torch.tensor([300, 100, 40, -1])
    out = a.decode_forward(
        tokens,
        positions,
        sampling_params=_greedy(4),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    assert out.shape == (4, 1) and out.dtype == torch.int32
    step = [c for c in m.calls if c[0] == "decode"][-1]
    fed_ids, fed_pos = step[1], step[2]
    B = 4
    for slot, tok, pos in ((0, 11, 300), (1, 22, 100), (2, 33, 40)):
        user = a.slots.physical(slot).user
        for r in range(2):  # the rows replicate the prompt: every row's copy of the user slot gets the token
            assert fed_ids[r * B + user] == tok and fed_pos[r * B + user] == pos
        assert int(out[slot, 0]) == _next_token(tok, pos)
    idle = [i for i in range(2 * B) if i % B == 3]
    assert all(fed_ids[i] == PAD and fed_pos[i] == 0 for i in idle)
    assert int(out[3, 0]) == 0
    assert a.slots.physical(0).position == 301 and a.slots.physical(0).phase == "decode"
    # host sampling: logits in row order, argmax = the same next token
    logits, lp = a.decode_forward(
        torch.tensor([[1], [2], [3], [0]], dtype=torch.int32),
        torch.tensor([301, 101, 41, -1]),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert lp is None and logits.shape == (4, 1, VOCAB)
    assert [int(logits[i, 0].argmax()) for i in range(3)] == [
        _next_token(1, 301),
        _next_token(2, 101),
        _next_token(3, 41),
    ]
    assert float(logits[3].abs().sum()) == 0.0
    # the deferred-read form the plugin's sync path uses: a device handle, then process_decode_output_host
    handle = a.decode_forward(
        torch.tensor([[4], [5], [6], [0]], dtype=torch.int32),
        torch.tensor([302, 102, 42, -1]),
        read_from_device=False,
        sampling_params=_greedy(4),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert isinstance(handle, GLM53DecodeOutput) and not handle.on_host
    host, events = a.read_decode_output(handle, async_read=True)
    assert events == [] and host.on_host
    toks = a.process_decode_output_host(host, is_tokens=True)
    assert [int(toks[i, 0]) for i in range(3)] == [_next_token(4, 302), _next_token(5, 102), _next_token(6, 42)]


def test_decode_loads_inputs_with_one_call_per_step():
    """Every decode step stages its inputs through the model's ``load_decode_inputs`` (one packed upload with
    ``GLM53_DECODE_PACKED_INPUTS=1``); a model without it gets the ``load_ids`` + ``set_positions`` pair."""
    a, m = _adapter()
    t = torch.full((1, 40), 5, dtype=torch.int32)
    a.prefill_forward(t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    before = len([c for c in m.calls if c[0] == "load_decode_inputs"])
    kw = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False)
    for step in range(3):
        a.decode_forward(
            torch.tensor([[1], [0], [0], [0]], dtype=torch.int32),
            torch.tensor([40 + step, -1, -1, -1]),
            sampling_params=_greedy(4),
            **kw,
        )
    loads = [c for c in m.calls if c[0] == "load_decode_inputs"]
    assert len(loads) - before == 3 and all(c[1] == m.total_users for c in loads)
    assert m._fed_pos[a.slots.physical(0).user] == 42

    class _NoSingleLoad(_FakeMeshModel):
        load_decode_inputs = None  # an older model surface: the adapter falls back to the pair

    b, mb = _adapter(model_cls=_NoSingleLoad)
    b.prefill_forward(t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    out = b.decode_forward(
        torch.tensor([[1], [0], [0], [0]], dtype=torch.int32),
        torch.tensor([40, -1, -1, -1]),
        sampling_params=_greedy(4),
        **kw,
    )
    assert int(out[0, 0]) == _next_token(1, 40) and not [c for c in mb.calls if c[0] == "load_decode_inputs"]


def test_decode_reload_contract_is_enforced(monkeypatch):
    a, _ = _adapter()
    t = torch.full((1, 40), 5, dtype=torch.int32)
    a.prefill_forward(t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    tokens, positions = torch.tensor([[1], [0], [0], [0]], dtype=torch.int32), torch.tensor([40, -1, -1, -1])
    with expect_error(ValueError, "reload_inputs=False"):
        a.decode_forward(
            tokens,
            positions,
            reload_inputs=False,
            reload_page_table=True,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    with expect_error(TypeError, "reset_batch"):
        a.decode_forward(
            tokens,
            positions,
            reset_batch=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    with expect_error(ValueError, "greedy"):
        a.decode_forward(
            tokens,
            positions,
            sampling_params=SimpleNamespace(temperature=[0.5] * 4, top_k=[10] * 4),
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    with expect_error(RuntimeError, "never prefilled"):
        a.decode_forward(
            tokens,
            torch.tensor([40, 7, -1, -1]),
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    with expect_error(RuntimeError, "prefilled to 40"):
        a.decode_forward(
            tokens,
            torch.tensor([41, -1, -1, -1]),
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    assert a.slots.physical(0).phase == "prefill"  # a refused step leaves the table untouched
    # the resident chain (GLM53_RESIDENT_DECODE=1, doc/mesh/resident_decode.md): the reload step stages the inputs
    # with the live mask, a chain step uploads nothing (the device fed its sampled ids back and advanced the live
    # positions), a disagreeing position or a changed layout without reload_inputs is refused
    monkeypatch.setenv(RESIDENT_ENV, "1")
    a, m = _adapter()
    assert a.resident and m.resident
    a.prefill_forward(t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    kw = dict(
        sampling_params=_greedy(4), reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )
    out1 = a.decode_forward(tokens, positions, reload_inputs=True, **kw)
    t1 = _next_token(1, 40)
    assert int(out1[0, 0]) == t1 and a.slots.physical(0).position == 41
    assert m._live is not None and sum(m._live) == len(a.slots.flat_indices(0))
    loads = len([c for c in m.calls if c[0] == "load_decode_inputs"])
    out2 = a.decode_forward(torch.zeros_like(tokens), torch.tensor([41, -1, -1, -1]), reload_inputs=False, **kw)
    assert int(out2[0, 0]) == _next_token(t1, 41) and a.slots.physical(0).position == 42
    assert len([c for c in m.calls if c[0] == "load_decode_inputs"]) == loads  # nothing uploaded on the chain step
    assert m.calls[-1][0] == "advance_positions_host" and sum(m.calls[-1][1]) == len(a.slots.flat_indices(0))
    # async scheduling builds step N + 1 before step N's token lands: a plugin position one behind the device is
    # accepted (the device's is the truth), any other disagreement is refused
    out2b = a.decode_forward(torch.zeros_like(tokens), torch.tensor([41, -1, -1, -1]), reload_inputs=False, **kw)
    assert int(out2b[0, 0]) == _next_token(_next_token(t1, 41), 42) and a.slots.physical(0).position == 43
    assert a._perf_stale_positions == 1
    assert ("replay", False) in m.calls and ("replay", True) not in m.calls  # the default: non-blocking
    # GLM53_DECODE_REPLAY_BLOCKING=1: the blocking replay (the A/B control)
    monkeypatch.setenv(REPLAY_BLOCKING_ENV, "1")
    b, mb = _adapter()
    b.prefill_forward(t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    b.decode_forward(tokens, positions, reload_inputs=True, **kw)
    assert b._replay_blocking and ("replay", True) in mb.calls
    monkeypatch.delenv(REPLAY_BLOCKING_ENV)
    with expect_error(RuntimeError, "diverged"):
        a.decode_forward(tokens, torch.tensor([50, -1, -1, -1]), reload_inputs=False, **kw)
    t2 = torch.full((1, 40), 6, dtype=torch.int32)
    a.prefill_forward(t2, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[1], sampling_params=_greedy(1))
    with expect_error(RuntimeError, "layout changed"):
        a.decode_forward(tokens, torch.tensor([43, 40, -1, -1]), reload_inputs=False, **kw)
    out3 = a.decode_forward(
        torch.tensor([[t1], [1], [0], [0]], dtype=torch.int32), torch.tensor([43, 40, -1, -1]), reload_inputs=True, **kw
    )
    assert int(out3[1, 0]) == _next_token(1, 40) and a.slots.physical(1).position == 41
    with expect_error(ValueError, "not a legal command"):  # reload_page_table rides a chain step only
        a.decode_forward(tokens, torch.tensor([44, 41, -1, -1]), reload_inputs=True, reload_page_table=True)


def test_chunked_prefill_contiguity_and_the_mid_prefill_guard():
    a, m = _adapter()
    L = 16384 * 2 + 100
    ids = (torch.arange(L) % 60).to(torch.int32).reshape(1, L)
    # vLLM chunk 1: [0, 16384) -> one eager device chunk into user 0
    a.prefill_forward(
        ids, prompt_lens=[16384], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1)
    )
    assert [c for c in m.calls if c[0] == "prefill"][-1] == ("prefill", 0, 0, 16384)
    s = a.slots.physical(0)
    assert s.phase == "prefill" and s.prefilled_to == 16384
    # another request decodes while slot 0 is mid-prefill: refused (the decode step would advance slot 0's KDA state)
    t = torch.full((1, 40), 5, dtype=torch.int32)
    a.prefill_forward(t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[1], sampling_params=_greedy(1))
    with expect_error(RuntimeError, "mid-prefill"):
        a.decode_forward(
            torch.tensor([[0], [9], [0], [0]], dtype=torch.int32),
            torch.tensor([-1, 40, -1, -1]),
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
    # with the slot mask (GLM53MeshModel.supports_slot_mask: the decode step leaves a parked slot's KDA state alone)
    # the step runs: slot 0 is parked at its prefilled end with the pad id, out of the live set, its table untouched
    m.supports_slot_mask = True
    a.decode_forward(
        torch.tensor([[0], [9], [0], [0]], dtype=torch.int32),
        torch.tensor([-1, 40, -1, -1]),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    f0, f1 = a.slots.flat_indices(0)[0], a.slots.flat_indices(1)[0]
    assert m._fed_pos[f0] == 16384 and int(m._fed_ids[f0]) == a.slots.g.pad_token_id and m._fed_pos[f1] == 40
    assert s.phase == "prefill" and s.prefilled_to == 16384 and a.slots.physical(1).position == 41
    m.supports_slot_mask = False
    # a non-contiguous continuation is refused; the contiguous one runs as one eager chunk of 16484 tokens
    with expect_error(RuntimeError, "not contiguous"):
        a.prefill_forward(ids, prompt_lens=[L], start_pos=torch.tensor([16000]), empty_slots=[0])
    # a 16484-token vLLM chunk is longer than the device chunk: the adapter walks it in two contiguous device calls
    a.prefill_forward(
        ids, prompt_lens=[L], start_pos=torch.tensor([16384]), empty_slots=[0], sampling_params=_greedy(1)
    )
    assert [c for c in m.calls if c[0] == "prefill"][-2:] == [("prefill", 0, 16384, 16384), ("prefill", 0, 32768, 100)]
    assert a.slots.physical(0).prefilled_to == L
    # a whole long prompt in one vLLM step walks the device chunks in order (the warm-up's own calls into that
    # slot, the ladder and the continued-chunk walk, precede n0)
    n0 = len(m.calls)
    a.prefill_forward(ids, prompt_lens=[L], start_pos=torch.tensor([0]), empty_slots=[2], sampling_params=_greedy(1))
    chunks = [c for c in m.calls[n0:] if c[0] == "prefill" and c[1] == a.slots.physical(2).user]
    assert [(c[2], c[3]) for c in chunks] == [(0, 16384), (16384, 16384), (32768, 100)]
    # both prompts complete: the decode runs for all three slots
    out = a.decode_forward(
        torch.tensor([[1], [2], [3], [0]], dtype=torch.int32),
        torch.tensor([L, 40, L, -1]),
        sampling_params=_greedy(4),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert [int(out[i, 0]) for i in range(3)] == [_next_token(1, L), _next_token(2, 40), _next_token(3, L)]


def test_slot_remap_release_and_move_notification():
    a, m = _adapter()
    for slot in range(3):
        t = torch.full((1, 40), 2 + slot, dtype=torch.int32)
        a.prefill_forward(
            t, prompt_lens=[40], start_pos=torch.tensor([0]), empty_slots=[slot], sampling_params=_greedy(1)
        )
    users = {slot: a.slots.physical(slot).user for slot in range(3)}
    assert users == {0: 0, 1: 1, 2: 2}
    # slot 1 finishes; the plugin condenses row 2 into row 1 and sends remap [0, 2, 1, 3]
    a.release_request(1)
    out = a.decode_forward(
        torch.tensor([[1], [2], [0], [0]], dtype=torch.int32),
        torch.tensor([40, 40, -1, -1]),
        sampling_params=_greedy(4),
        slot_remap=torch.tensor([0, 2, 1, 3], dtype=torch.int32),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    step = [c for c in m.calls if c[0] == "decode"][-1]
    assert step[1][2] == 2 and step[2][2] == 40  # row 1's token went to physical user 2 (the former slot 2)
    assert step[1][1] == PAD  # the released user 1 idles
    assert a.slots.physical(1).user == 2 and a.slots.physical(2) is None
    assert [int(out[i, 0]) for i in range(2)] == [_next_token(1, 40), _next_token(2, 40)]
    # the plugin then reports the same gather: consumed, not applied twice
    a.note_state_slots_moved({2: 1, 1: 2})
    assert a.slots.physical(1).user == 2
    # a gather applied through slot_remap must be reported with the same moves
    a.decode_forward(
        torch.tensor([[1], [2], [0], [0]], dtype=torch.int32),
        torch.tensor([41, 41, -1, -1]),
        sampling_params=_greedy(4),
        slot_remap=torch.tensor([1, 0, 2, 3], dtype=torch.int32),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert a.slots.physical(0).user == 2 and a.slots.physical(1).user == 0
    with expect_error(RuntimeError, "disagrees"):
        a.note_state_slots_moved({3: 0})
    a.note_state_slots_moved({1: 0, 0: 1})
    # a plugin that never passes slot_remap: the notification is applied once
    b, _ = _adapter()
    for slot in range(2):
        b.prefill_forward(
            torch.full((1, 40), 9, dtype=torch.int32),
            prompt_lens=[40],
            start_pos=torch.tensor([0]),
            empty_slots=[slot],
            sampling_params=_greedy(1),
        )
    b.note_state_slots_moved({1: 0, 0: 1})
    assert b.slots.physical(0).user == 1 and b.slots.physical(1).user == 0
    # a new prompt handed a reused logical slot replaces the previous occupant
    b.prefill_forward(
        torch.full((1, 40), 4, dtype=torch.int32),
        prompt_lens=[40],
        start_pos=torch.tensor([0]),
        empty_slots=[0],
        sampling_params=_greedy(1),
    )
    assert b.slots.physical(0).user == 1 and b.slots.column_loads() == [1, 1, 0, 0]
    b.release_persistent_capture()
    assert b.trace_id is None and b.tt_supported_decode_batch_sizes == ()


# ----------------------------------------------------------------------------- per-row prefill ownership
def _decode_kw(**kw):
    base = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False)
    base.update(kw)
    return base


def test_rows_independent_needs_the_wave_prefill():
    with expect_error(NotImplementedError, "prefill_wave"):
        _adapter(independent=True, warm=False, model_cls=_FakeMeshModelNoWaves)
    a, m = _adapter(independent=True)  # the fake has it: capacity = rows x users_per_row, the wave ladder ran
    assert a.geometry.request_capacity == 8 and a.tt_supported_decode_batch_sizes == (8,)
    assert a.geometry.traced_wave_lengths == (256, 512, 1024, 2048)  # the model's per-row wave lengths
    g = a.geometry
    waves = [c for c in m.calls if c[0] == "prefill_wave"]
    assert all(c[3] is None and [r for r, *_ in c[4]] == [0, 1] for c in waves)  # eager, every row a member
    # every per-row bucket up to the wave width into slot 0 (the shortest into every slot) at start 0, at the
    # wave width (a second chunk) and at every prefix bucket (the chunk here: no GLM53_PREFIX_BUCKET)
    buckets = [b for b in prefill_buckets() if b <= g.wave_rows]
    starts = [0, g.wave_rows] + [j * g.device_prefill_chunk for j in range(1, g.max_seq_len // g.device_prefill_chunk)]
    shapes = {(c[1], c[2], c[4][0][2]) for c in waves}  # (slot index, t_pad, start)
    assert shapes == {(0, b, s) for s in starts for b in buckets} | {
        (u, buckets[0], s) for s in starts for u in (1, 2, 3)
    }
    full = [c for c in waves if c[4][0][3] - c[4][0][2] == c[2]]
    ragged = [c for c in waves if c not in full]
    assert len(full) + len(ragged) == len(waves) and len(full) >= len(ragged) == len(buckets)
    # every bucket once ragged: row 0 one token short (the pad-row mask program), the other rows full, start 0, slot 0
    assert sorted(c[2] for c in ragged) == buckets
    assert all(c[1] == 0 and c[4][0][2:] == (0, c[2] - 1) and c[4][1][2:] == (0, c[2]) for c in ragged)
    # the traced phase: the wave inputs + compiles before ANY capture, the waves captured after the prefill traces
    order = [c[0] for c in m.calls if c[0] in ("prepare_waves", "capture_prefill", "capture_waves", "capture")]
    assert order == ["prepare_waves", "capture_waves", "capture"]  # no SP prefill traces under per-row ownership
    assert a.geometry.traced_prefill_lengths == () and m._captured == []
    assert m._captured_waves == [256, 512] and not m._ready_waves  # the default 128,256,512 within the model's lengths
    os.environ["GLM53_VLLM_WAVE_TRACE_LENGTHS"] = "all"
    os.environ["GLM53_VLLM_SP_TRACES"] = "1"
    try:
        a_all, m_all = _adapter(independent=True)
        assert m_all._captured_waves == [256, 512, 1024, 2048]
        assert a_all.geometry.traced_prefill_lengths == (256, 512, 1024, 2048) and m_all._captured == [
            256,
            512,
            1024,
            2048,
        ]
        os.environ.pop("GLM53_VLLM_SP_TRACES")
        os.environ["GLM53_VLLM_WAVE_TRACE_LENGTHS"] = "0"
        _, m_none = _adapter(independent=True)
        assert m_none._captured_waves == [] and not any(c[0] == "prepare_waves" for c in m_none.calls)
        os.environ["GLM53_VLLM_WAVE_TRACE_LENGTHS"] = "96"
        with expect_error(ValueError, "wave trace lengths"):
            _adapter(independent=True)
    finally:
        os.environ.pop("GLM53_VLLM_WAVE_TRACE_LENGTHS", None)
        os.environ.pop("GLM53_VLLM_SP_TRACES", None)
    # the wave width must be a prefill bucket: a wave's chunk pads to the next bucket
    with expect_error(ValueError, "not a prefill bucket"):
        _adapter(independent=True, warm=False, wave_rows=2304)


def test_wave_prefill_routes_rows_and_decodes_per_row():
    """2 rows x 4 users per row (one per column): the first two arrivals share slot index 0 (one per row) and form
    one traced wave; the third sits alone on index 1 (SP-eligible) and, being short and fresh, takes a traced wave of
    its own (no SP prefill traces under per-row ownership); decode feeds each
    request's token to its own (row, user) only."""
    a, m = _adapter(independent=True)
    toks = torch.zeros(3, 300, dtype=torch.int32)
    toks[0, :300] = torch.arange(300) % 50
    toks[1, :100] = torch.arange(100) % 20 + 1
    toks[2, :40] = 5
    logits = a.prefill_forward(
        toks, prompt_lens=[300, 100, 40], start_pos=torch.tensor([0, 0, 0]), empty_slots=[0, 1, 2], enable_trace=True
    )
    assert logits.shape == (3, 1, VOCAB)
    placed = {i: (a.slots.physical(i).row, a.slots.physical(i).user) for i in range(3)}
    assert placed == {0: (0, 0), 1: (1, 0), 2: (0, 1)}
    waves = [c for c in m.calls if c[0] == "prefill_traced_wave"]
    assert waves[0][1:] == (0, 320, 512, [(0, 0, 0, 300), (1, 0, 0, 100)])
    # the third request is alone on index 1 (SP-eligible) but short and fresh: the traced wave, not an eager SP chunk
    assert waves[1][1:] == (1, 64, 256, [(0, 1, 0, 40)])
    assert not [c for c in m.calls if c[0] == "prefill_traced"]
    assert int(logits[0, 0].argmax()) == _first_token(0, toks[0, :300].long())
    assert int(logits[1, 0].argmax()) == _first_token(0, toks[1, :100].long())
    assert int(logits[2, 0].argmax()) == _first_token(1, toks[2, :40].long())
    assert not any(a.slots.physical(i).sp_only for i in range(3))  # every prefix exists on its own row only (waves)
    assert [a.slots.physical(i).prefilled_to for i in range(3)] == [300, 100, 40]
    # decode: every request's token at its own (row, user) flat index, nothing replicated
    out = a.decode_forward(
        torch.tensor([[11], [22], [33], [0]], dtype=torch.int32),
        torch.tensor([300, 100, 40, -1]),
        sampling_params=_greedy(4),
        **_decode_kw(reload_sampling_params=True, reset_sampling_state=True),
    )
    step = [c for c in m.calls if c[0] == "decode"][-1]
    fed_ids, fed_pos = step[1], step[2]
    B = 4
    assert fed_ids[0 * B + 0] == 11 and fed_pos[0] == 300  # (row 0, user 0)
    assert fed_ids[1 * B + 0] == 22 and fed_pos[1 * B] == 100  # (row 1, user 0)
    assert fed_ids[0 * B + 1] == 33 and fed_pos[1] == 40  # (row 0, user 1)
    assert fed_ids[1 * B + 1] == PAD and fed_pos[1 * B + 1] == 0  # (row 1, user 1) idle
    assert [int(out[i, 0]) for i in range(3)] == [_next_token(11, 300), _next_token(22, 100), _next_token(33, 40)]
    assert int(out[3, 0]) == 0


def test_wave_prefill_walks_chunks_and_switches_sp_to_waves_one_way():
    a, m = _adapter(independent=True, max_seq_len=65536)
    L = 2048 * 2 + 100
    ids = (torch.arange(L) % 60).to(torch.int32).reshape(1, L)
    # a lone prompt beyond the captured wave lengths (256 / 512 by default): the eager SP chunks (no SP prefill
    # traces under per-row ownership) while slot index 0 is free on row 1
    n0 = len(m.calls)
    a.prefill_forward(ids, prompt_lens=[2048], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1))
    kinds = [c[0] for c in m.calls[n0:] if c[0].startswith("prefill")]
    assert kinds == ["prefill"] and m.calls[n0:][0][1:3] == (0, 0), kinds  # one eager SP chunk into user 0 at start 0
    assert a.slots.physical(0).sp_only and a.slots.physical(0).prefilled_to == 2048
    # three eager arrivals take the other slot indices first (SlotTable.allocate(prefer_sp): each stays SP-eligible),
    # so the next eager prompt must share index 0: it lands on (1, 0), slot index 0 is no longer free on the other
    # row, and request 0's next chunk and the new prompt go through waves; the 2148-token continuation is walked in
    # 2048-row device chunks
    for k in (2, 3, 4):
        a.slots.allocate(k, prefer_sp=True)
    assert [(a.slots.physical(k).row, a.slots.physical(k).user) for k in (2, 3, 4)] == [(1, 1), (0, 2), (1, 3)]
    assert a.slots.sp_eligible(0)
    both = torch.zeros(2, L, dtype=torch.int32)
    both[0] = ids[0]
    both[1, :3000] = torch.arange(3000) % 7 + 1
    out, _ = a.prefill_forward(
        both, prompt_lens=[L, 3000], start_pos=torch.tensor([2048, 0]), empty_slots=[0, 1], sampling_params=_greedy(2)
    )
    waves = [c for c in m.calls if c[0] == "prefill_wave"]
    assert [w[4] for w in waves[-2:]] == [[(0, 0, 2048, 4096), (1, 0, 0, 2048)], [(0, 0, 4096, L), (1, 0, 2048, 3000)]]
    assert [w[2] for w in waves[-2:]] == [2048, 960] and all(w[3] is None for w in waves[-2:])
    assert int(out[0, 0]) == _first_token(0, ids[0, 4096:L].long()) and int(out[1, 0]) == _first_token(
        0, both[1, 2048:3000].long()
    )
    assert [a.slots.physical(i).prefilled_to for i in range(2)] == [L, 3000]
    assert not a.slots.physical(0).sp_only and not a.slots.physical(1).sp_only
    # the other row frees up again: request 0 stays on waves (its prefix lives on row 0 only)
    a.release_request(1)
    assert not a.slots.sp_eligible(0)
    # the mid-prefill guard still holds under waves
    a.prefill_forward(
        torch.full((1, 64), 3, dtype=torch.int32),
        prompt_lens=[32],
        start_pos=torch.tensor([0]),
        empty_slots=[2],
        sampling_params=_greedy(1),
    )
    with expect_error(RuntimeError, "mid-prefill"):
        a.decode_forward(
            torch.tensor([[1], [0], [0], [0]], dtype=torch.int32), torch.tensor([L, -1, -1, -1]), **_decode_kw()
        )

    # a burst of fresh long prompts that fills a wave (``rows`` of them in one step) takes the wave-affine order even
    # though slot indices are free on every row: both land on slot index 0 of rows 0 and 1 and prefill as ONE wave of
    # two members (the 32 x 1 M capstone r2 vs r3: SP-taken slot indices fragmented the waves at 1.15 K tok/s, the
    # wave-affine order ran waves of 8 at 2.28 K tok/s); a lone long prompt still takes the sequence-parallel path
    c, mc = _adapter(independent=True, max_seq_len=65536)
    pair = torch.zeros(2, 2048, dtype=torch.int32)
    pair[0] = torch.arange(2048) % 60
    pair[1] = torch.arange(2048) % 7 + 1
    c.prefill_forward(
        pair, prompt_lens=[2048, 2048], start_pos=torch.tensor([0, 0]), empty_slots=[0, 1], sampling_params=_greedy(2)
    )
    assert [(c.slots.physical(k).row, c.slots.physical(k).user) for k in (0, 1)] == [(0, 0), (1, 0)]
    assert [w[4] for w in mc.calls if w[0] == "prefill_wave"][-1] == [(0, 0, 0, 2048), (1, 0, 0, 2048)]
    assert not any(c.slots.physical(k).sp_only for k in (0, 1))
    c.prefill_forward(
        pair[:1], prompt_lens=[2048], start_pos=torch.tensor([0]), empty_slots=[2], sampling_params=_greedy(1)
    )
    assert c.slots.physical(2).user == 1 and c.slots.physical(2).sp_only  # alone: SP on the next free index
    # arrivals straggle across steps (four rows here): A arrives alone (SP on slot index 0 of row 0), then B and C
    # one step apart (fresh 1 + young 1 or 2 < 4 rows: SP on index 1 of row 1, index 2 of row 2), then D arrives with
    # A's, B's and C's continuations: fresh 1 + young 3 fill a wave of 4, so the young prompts off the wave's index
    # (B, C) restart at 0 on index 0, A (already on index 0) continues as a member, and the step prefills ONE wave of
    # four (no SP chunk) instead of fragmenting into a wave of two + two SP chunks (needle64k_8_adaptive 2026-10-30:
    # waves of 5 + 3 SP chunks per step)
    d, md = _adapter(rows=4, independent=True, max_seq_len=65536)
    ids4 = [(torch.arange(4096) % m + 1).to(torch.int32).reshape(1, 4096) for m in (60, 7, 11, 13)]
    for k in range(3):
        d.prefill_forward(
            ids4[k], prompt_lens=[2048], start_pos=torch.tensor([0]), empty_slots=[k], sampling_params=_greedy(1)
        )
        assert (d.slots.physical(k).row, d.slots.physical(k).user) == (k, k) and d.slots.physical(k).sp_only
    assert [d.slots.physical(k).prefilled_to for k in range(3)] == [2048] * 3  # young SP prompts kept: 1 + 2 < 4
    quad = torch.zeros(4, 4096, dtype=torch.int32)
    for k in range(4):
        quad[k] = ids4[k][0]
    n0 = len(md.calls)
    d.prefill_forward(
        quad,
        prompt_lens=[4096, 4096, 4096, 2048],
        start_pos=torch.tensor([2048, 2048, 2048, 0]),
        empty_slots=[0, 1, 2, 3],
        sampling_params=_greedy(4),
    )
    waves = [w for w in md.calls[n0:] if w[0] == "prefill_wave"]
    assert [w[4] for w in waves] == [
        [(0, 0, 2048, 4096), (1, 0, 0, 2048), (2, 0, 0, 2048), (3, 0, 0, 2048)],
        [(1, 0, 2048, 4096), (2, 0, 2048, 4096)],
    ], waves
    assert not any(c0[0] == "prefill" for c0 in md.calls[n0:])  # no SP chunk in the step
    assert [
        (d.slots.physical(k).row, d.slots.physical(k).user, d.slots.physical(k).prefilled_to) for k in range(4)
    ] == [
        (0, 0, 4096),
        (1, 0, 4096),
        (2, 0, 4096),
        (3, 0, 2048),
    ]
    assert not any(d.slots.physical(k).sp_only for k in range(4))
    # the served shape (needle64k_8_adaptive_r2: the first step saw ONE arrival, the second its continuation + seven
    # fresh): the young prompt already sits on the wave's index, so nothing restarts, but it counts in the burst --
    # without it the seven fresh prompts read as a burst of 7 < 8 and the first three took SP indices (waves of 5)
    e, me = _adapter(rows=8, independent=True, max_seq_len=65536)
    ids8 = [(torch.arange(4096) % m + 1).to(torch.int32).reshape(1, 4096) for m in (60, 7, 11, 13, 17, 19, 23, 29)]
    e.prefill_forward(
        ids8[0], prompt_lens=[2048], start_pos=torch.tensor([0]), empty_slots=[0], sampling_params=_greedy(1)
    )
    assert (e.slots.physical(0).row, e.slots.physical(0).user) == (0, 0) and e.slots.physical(0).sp_only
    octet = torch.zeros(8, 4096, dtype=torch.int32)
    for k in range(8):
        octet[k] = ids8[k][0]
    n0 = len(me.calls)
    e.prefill_forward(
        octet,
        prompt_lens=[4096] + [2048] * 7,
        start_pos=torch.tensor([2048] + [0] * 7),
        empty_slots=list(range(8)),
        sampling_params=_greedy(8),
    )
    waves = [w for w in me.calls[n0:] if w[0] == "prefill_wave"]
    assert [w[4] for w in waves] == [[(0, 0, 2048, 4096)] + [(r, 0, 0, 2048) for r in range(1, 8)]], waves
    assert not any(c0[0] == "prefill" for c0 in me.calls[n0:])  # one wave of eight, no SP chunk, no restart
    assert [(e.slots.physical(k).row, e.slots.physical(k).user) for k in range(8)] == [(k, 0) for k in range(8)]


def test_kv_accounting_hooks():
    a, _ = _adapter()
    cache = a.allocate_kv_cache((1234, 1, 32, 512), torch.bfloat16, 11)
    assert cache.num_blocks == 1234 and cache.num_layers == 11
    with expect_error(ValueError, "block size"):
        a.allocate_kv_cache((10, 1, 64, 512), torch.bfloat16, 11)
    assert GLM53FlashForCausalLM.get_max_tokens_all_users(max_model_len=1048576, max_num_seqs=4) == 4 * 1048576
    with expect_error(ValueError, "needs max_model_len"):
        GLM53FlashForCausalLM.get_max_tokens_all_users()


def test_zz_no_ttnn_was_imported_by_the_suite():
    """Runs last (file order): every adapter path the fake model exercised stayed off ttnn."""
    assert "ttnn" not in sys.modules
