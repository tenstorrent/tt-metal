# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the resident AdaLN schedule tables (`MINIMAX_H3_ADALN_PRECOMPUTE`).

The device ops are replaced by torch stand-ins, so these check the row layout and bookkeeping: the rows a
forward selects from the resident tables are exactly the rows the block's own `_modulation_tables` returns
for that step, on #58676's layout (`slot * MODALITY_NUM + modality` rows, the norm weight folded into the
`1 + scale` rows), and the tables are allocated only while no trace is live and never reallocated.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.tt_dit.models.transformers.minimax_h3 import adaln_schedule_minimax_h3 as schedule_module
from models.tt_dit.models.transformers.minimax_h3 import transformer_minimax_h3 as tm
from models.tt_dit.models.transformers.minimax_h3.adaln_schedule_minimax_h3 import AdalnRowPlan, AdalnSchedule
from models.tt_dit.models.transformers.minimax_h3.transformer_block_minimax_h3 import (
    _SCALE_MLP,
    _SCALE_MSA,
    MODALITY_NUM,
    NUM_MODULATION_PARAMS,
)
from models.tt_dit.pipelines.minimax_h3 import pipeline_minimax_h3 as pm
from models.tt_dit.utils.tracing import Tracer

ROLES = ("video", "audio", "condition_video")
HIDDEN = 8
TEMB_DIM = 5


def _schedule(levels, endpoints=None, roles=ROLES):
    return AdalnSchedule(
        roles=roles,
        levels=tuple(torch.tensor(step, dtype=torch.float32) for step in levels),
        endpoints=None if endpoints is None else tuple(torch.tensor(e, dtype=torch.float32) for e in endpoints),
    )


class _FakeBlock:
    """A block whose `_modulation_tables` follows #58676's layout in torch: project `silu(temb)`, fold the
    modality axis into rows (`slot * MODALITY_NUM + modality`), split the six parameters, add 1 to the scale
    rows and fold the norm weight into them, round to bf16."""

    def __init__(self, generator):
        self.hidden_local = HIDDEN
        self.weight = torch.randn(MODALITY_NUM * NUM_MODULATION_PARAMS * HIDDEN, TEMB_DIM, generator=generator)
        self.norm_weight = {p: torch.rand(HIDDEN, generator=generator) + 0.5 for p in (_SCALE_MSA, _SCALE_MLP)}
        self.calls = 0

    def _modulation_tables(self, temb):
        self.calls += 1
        slots = temb.shape[2]
        # Row by row, so a slot's rows cannot depend on how many slots share the call (CPU GEMMs pick kernels by M).
        projected = torch.stack(
            [torch.nn.functional.silu(row) @ self.weight.T for row in temb.reshape(slots, TEMB_DIM)]
        )
        projected = projected.reshape(slots * MODALITY_NUM, NUM_MODULATION_PARAMS * HIDDEN)
        tables = []
        for p in range(NUM_MODULATION_PARAMS):
            table = projected[:, p * HIDDEN : (p + 1) * HIDDEN]
            if p in (_SCALE_MSA, _SCALE_MLP):
                table = (table + 1.0) * self.norm_weight[p]
            tables.append(table.to(torch.bfloat16).float())
        return tables

    def _gather_rows(self, table, selector):
        return selector @ table

    select_modulation_tables = tm.MiniMaxH3TransformerBlock.select_modulation_tables


def _torch_ttnn(monkeypatch):
    """Torch stand-ins for the ttnn calls the resident-table code makes."""
    copies = []

    def copy(src, dst):
        assert src.shape == dst.shape, (src.shape, dst.shape)
        dst.copy_(src)
        copies.append(dst.data_ptr())

    monkeypatch.setattr(ttnn, "zeros", lambda shape, **kwargs: torch.zeros(shape))
    monkeypatch.setattr(ttnn, "add", torch.add)
    monkeypatch.setattr(ttnn, "copy", copy)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None)
    return copies


def _fake_transformer(num_blocks=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    fake = SimpleNamespace(
        transformer_blocks=[_FakeBlock(generator) for _ in range(num_blocks)],
        mesh_device=None,
        _adaln_tables=None,
        embed=torch.randn(TEMB_DIM, generator=generator),
    )
    # temb per slot from its level (and endpoint): any deterministic per-row function stands in for the embedder.
    fake.time_embedding = lambda timestep, endpoint=None: (
        timestep.reshape(1, 1, -1, 1) * fake.embed + (0 if endpoint is None else 0.5 * endpoint.reshape(1, 1, -1, 1))
    )
    for name in ("allocate_adaln_tables", "write_adaln_tables"):
        method = getattr(tm.MiniMaxH3Transformer3DModel, name)
        setattr(fake, name, lambda *args, _m=method, **kwargs: _m(fake, *args, **kwargs))
    return fake


def _on_device_tables(transformer, block_index, levels, endpoints=None):
    """What the on-device path computes for one forward: the block projects that step's `temb`."""
    temb = transformer.time_embedding(levels, endpoints)
    return transformer.transformer_blocks[block_index]._modulation_tables(temb)


# ---------------------------------------------------------------------------- row plan


def test_schedule_equality_is_bitwise(expect_error):
    a = _schedule([[1.0, 0.5, 0.25], [0.5, 0.25, 0.25]])
    assert a.same_as(_schedule([[1.0, 0.5, 0.25], [0.5, 0.25, 0.25]]))
    nudged = torch.nextafter(torch.tensor(0.5), torch.tensor(1.0)).item()
    assert not a.same_as(_schedule([[1.0, 0.5, 0.25], [nudged, 0.25, 0.25]]))
    assert not a.same_as(_schedule([[1.0, 0.5, 0.25]]))
    assert not a.same_as(_schedule([[1.0, 0.5, 0.25], [0.5, 0.25, 0.25]], endpoints=[[0.5] * 3, [0.0] * 3]))
    with expect_error(ValueError, "expected"):
        _schedule([[1.0, 0.5]])
    with expect_error(ValueError, "endpoints"):
        _schedule([[1.0, 0.5, 0.25]], endpoints=[[0.5] * 3, [0.0] * 3])


def test_row_plan_appends_then_restarts_then_refuses(expect_error):
    plan = AdalnRowPlan(64, ROLES)
    warm = _schedule([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])  # 18 rows
    serve = _schedule([[0.9, 0.8, 0.9]] * 4)  # 36 rows
    plan.segments = plan.placed(warm)
    plan.segments = plan.placed(serve)
    assert [(s.offset, s.rows) for s in plan.segments] == [(0, 18), (18, 36)]
    assert plan.find(serve).offset == 18 and plan.find(warm).offset == 0
    # 54 rows used; a 2-step schedule (18) no longer fits behind them, so it starts over alone.
    other = _schedule([[0.7, 0.7, 0.7]] * 2)
    assert [(s.offset, s.rows) for s in plan.placed(other)] == [(0, 18)]
    assert plan.placed(_schedule([[0.1, 0.1, 0.1]] * 8)) is None  # 72 rows > 64
    with expect_error(ValueError, "roles"):
        plan.placed(_schedule([[1.0, 1.0]], roles=("video", "audio")))


def test_selector_routes_each_slot_to_its_role_row(expect_error):
    plan = AdalnRowPlan(64, ROLES)
    plan.segments = plan.placed(_schedule([[1.0, 1.0, 1.0]]))
    plan.segments = plan.placed(_schedule([[0.9, 0.8, 0.9]] * 3))
    segment = plan.segments[1]
    selector = plan.selector(segment, 2, ("audio", "video"))
    assert selector.shape == (2 * MODALITY_NUM, 64)
    assert torch.equal(selector.sum(dim=1), torch.ones(2 * MODALITY_NUM))
    base = 9 + 2 * 9
    for modality in range(MODALITY_NUM):
        assert selector[0 * MODALITY_NUM + modality, base + 1 * MODALITY_NUM + modality] == 1  # audio
        assert selector[1 * MODALITY_NUM + modality, base + 0 * MODALITY_NUM + modality] == 1  # video
    with expect_error(ValueError, "outside"):
        plan.selector(segment, 3, ROLES)
    with expect_error(ValueError, "not in"):
        plan.selector(segment, 0, ("condition_audio",))


def test_capacity_env(monkeypatch, expect_error):
    monkeypatch.delenv(schedule_module.ADALN_PRECOMPUTE_ROWS_ENV, raising=False)
    assert schedule_module.adaln_capacity_rows() == schedule_module.DEFAULT_CAPACITY_ROWS
    for bad in ("0", "48", "-32"):
        monkeypatch.setenv(schedule_module.ADALN_PRECOMPUTE_ROWS_ENV, bad)
        with expect_error(ValueError, "multiple of 32"):
            schedule_module.adaln_capacity_rows()
    monkeypatch.setenv(schedule_module.ADALN_PRECOMPUTE_ENV, "0")
    assert not schedule_module.adaln_precompute_enabled()
    monkeypatch.delenv(schedule_module.ADALN_PRECOMPUTE_ENV)
    assert schedule_module.adaln_precompute_enabled()


# ---------------------------------------------------------------------------- resident tables


@pytest.mark.parametrize("two_time", [False, True], ids=["one_time", "two_time"])
def test_selected_rows_equal_the_on_device_tables(monkeypatch, two_time):
    """Every step of every resident schedule: `selector @ resident` is bit-identical to `_modulation_tables`."""
    copies = _torch_ttnn(monkeypatch)
    transformer = _fake_transformer()
    plan = AdalnRowPlan(64, ROLES)
    transformer.allocate_adaln_tables(plan.capacity)
    allocated = [t.data_ptr() for tables in transformer._adaln_tables for t in tables]

    warm = _schedule([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]], [[0.5, 0.5, 1.0], [0.0, 0.0, 0.5]] if two_time else None)
    serve = _schedule(
        [[0.95, 0.9, 0.95], [0.7, 0.6, 0.7], [0.4, 0.3, 0.4], [0.1, 0.05, 0.1]],
        [[0.7, 0.6, 0.95], [0.4, 0.3, 0.7], [0.1, 0.05, 0.4], [0.0, 0.0, 0.1]] if two_time else None,
    )
    for schedule in (warm, serve):
        segments = plan.placed(schedule)
        transformer.write_adaln_tables([(t, e, plan.placement(row)) for t, e, row in plan.step_inputs(segments)])
        plan.segments = segments

    # Rewritten in place: the same buffers, every table copied into each write.
    assert [t.data_ptr() for tables in transformer._adaln_tables for t in tables] == allocated
    assert len(copies) == 2 * len(allocated)

    for schedule in (warm, serve):
        segment = plan.find(schedule)
        for step in range(schedule.num_steps):
            endpoints = None if schedule.endpoints is None else schedule.endpoints[step]
            for slot_roles in (ROLES, ("video", "audio")):
                index = [ROLES.index(role) for role in slot_roles]
                selector = plan.selector(segment, step, slot_roles)
                for b, block in enumerate(transformer.transformer_blocks):
                    selected = block.select_modulation_tables(selector, transformer._adaln_tables[b])
                    expected = _on_device_tables(
                        transformer,
                        b,
                        schedule.levels[step][index],
                        None if endpoints is None else endpoints[index],
                    )
                    for p in range(NUM_MODULATION_PARAMS):
                        assert torch.equal(selected[p], expected[p]), (step, slot_roles, b, p)
    # Rows past the last schedule stay zero.
    used = sum(s.rows for s in plan.segments)
    assert all(not t[used:].any() for tables in transformer._adaln_tables for t in tables)


def test_tables_refuse_misuse(monkeypatch, expect_error):
    _torch_ttnn(monkeypatch)
    transformer = _fake_transformer()
    plan = AdalnRowPlan(32, ROLES)
    with expect_error(RuntimeError, "allocate_adaln_tables"):
        transformer.write_adaln_tables([(torch.ones(3), None, plan.placement(0))])
    transformer.allocate_adaln_tables(32)
    with expect_error(RuntimeError, "already allocated"):
        transformer.allocate_adaln_tables(32)
    with expect_error(ValueError, "at least one step"):
        transformer.write_adaln_tables([])
    # A step placed past the end of the tables: 9 rows from row 24 overrun 32.
    assert plan.placement(23).shape == (32, 9)
    for row in (24, -1):
        with expect_error(ValueError, "leaves no room"):
            plan.placement(row)


# ---------------------------------------------------------------------------- pipeline


def _fake_pipeline(monkeypatch, *, coresident=True, enabled=True):
    monkeypatch.setattr(pm, "from_torch", lambda x, **kwargs: x)
    mesh = SimpleNamespace(id=lambda: 7)
    logs = []
    fake = SimpleNamespace(
        coresident=coresident,
        mesh_device=mesh,
        adaln_slot_roles=ROLES,
        _adaln_precompute=enabled,
        _adaln_plan=None,
        _adaln_plan_owner=None,
        _adaln_t_table=None,
        _log=logs.append,
        _host_log=logs.append,
    )
    fake._timestep_inputs = lambda levels: pm.MiniMaxH3Pipeline._timestep_inputs(fake, levels)
    return fake, logs


class _CountingTransformer:
    def __init__(self):
        self.allocated = []
        self.writes = []
        self._allocated = False

    @property
    def adaln_tables_allocated(self):
        return self._allocated

    def allocate_adaln_tables(self, capacity):
        self.allocated.append(capacity)
        self._allocated = True

    def write_adaln_tables(self, step_inputs):
        self.writes.append(len(step_inputs))
        self.rows = [int(placement.argmax(dim=0)[0]) for _, _, placement in step_inputs]


def _levels(rows, endpoints=False):
    return [
        (torch.tensor(r, dtype=torch.float32), torch.tensor(r, dtype=torch.float32) * 0.5 if endpoints else None)
        for r in rows
    ]


def test_pipeline_allocates_once_and_rewrites_per_new_schedule(monkeypatch):
    monkeypatch.setattr(Tracer, "_traces_live", {})
    fake, _ = _fake_pipeline(monkeypatch)
    transformer = _CountingTransformer()
    warm, serve = _levels([[1.0] * 3, [0.5] * 3]), _levels([[0.9] * 3] * 4)

    first = pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, warm)
    assert transformer.allocated == [schedule_module.DEFAULT_CAPACITY_ROWS] and transformer.writes == [2]
    # Warmup's captures are live from here on: a new schedule is rewritten in place, never reallocated.
    monkeypatch.setattr(Tracer, "_traces_live", {7: 2})
    second = pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, serve)
    assert transformer.allocated == [schedule_module.DEFAULT_CAPACITY_ROWS] and transformer.writes == [2, 6]
    assert (first.offset, second.offset) == (0, 18)
    # The rewrite keeps the warm schedule's rows and places the new steps behind them.
    assert transformer.rows == [0, 9, 18, 27, 36, 45]
    # A repeated schedule reuses its rows without a write.
    assert pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, serve) == second
    assert pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, warm) == first
    assert transformer.writes == [2, 6]


def test_pipeline_stays_on_device_when_tables_cannot_be_used(monkeypatch):
    serve = _levels([[0.9] * 3] * 4)
    for kwargs in ({"enabled": False}, {"coresident": False}):
        monkeypatch.setattr(Tracer, "_traces_live", {})
        fake, _ = _fake_pipeline(monkeypatch, **kwargs)
        transformer = _CountingTransformer()
        assert pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, serve) is None
        assert transformer.allocated == []

    # A trace already live before the first allocation: never allocate, and stop trying.
    monkeypatch.setattr(Tracer, "_traces_live", {7: 1})
    fake, logs = _fake_pipeline(monkeypatch)
    transformer = _CountingTransformer()
    assert pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, serve) is None
    assert transformer.allocated == [] and fake._adaln_precompute is False and "trace is already live" in logs[0]

    # A schedule larger than the tables.
    monkeypatch.setattr(Tracer, "_traces_live", {})
    monkeypatch.setenv(schedule_module.ADALN_PRECOMPUTE_ROWS_ENV, "32")
    fake, logs = _fake_pipeline(monkeypatch)
    transformer = _CountingTransformer()
    assert pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, _levels([[0.9] * 3] * 4)) is None
    assert transformer.writes == [] and "more rows" in logs[-1]


def test_pipeline_routes_a_slot_subset_onto_the_full_roles(monkeypatch):
    monkeypatch.setattr(Tracer, "_traces_live", {})
    fake, _ = _fake_pipeline(monkeypatch)
    transformer = _CountingTransformer()
    full = pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ROLES, _levels([[0.9, 0.8, 0.7]], endpoints=True))
    subset = pm.MiniMaxH3Pipeline._adaln_segment(
        fake, transformer, ("audio", "condition_video", "video"), _levels([[0.8, 0.7, 0.9]], endpoints=True)
    )
    assert subset == full and transformer.writes == [1]
    # A request without levels for every pinned role stays on device.
    assert pm.MiniMaxH3Pipeline._adaln_segment(fake, transformer, ("video", "audio"), _levels([[0.9, 0.8]])) is None
