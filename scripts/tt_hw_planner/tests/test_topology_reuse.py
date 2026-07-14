"""Both emit-e2e and optimize reshape topology through ONE shared planner (parallelism.plan_parallelism
-> select_parallelism). These tests pin: chip-count derivation from --devices/--mesh, the exported
TT_PERF_MESH_ROWS/COLS shape (rows=dp, cols=tp), the 1D fallback when the model can't be probed, and
that emit-e2e's _planned_parallelism now delegates to the same planner.
"""

from __future__ import annotations

import os
from argparse import Namespace

from scripts.tt_hw_planner.commands import emit_e2e as E
from scripts.tt_hw_planner.commands import optimize as O
from scripts.tt_hw_planner.parallelism import ParallelConfig, plan_parallelism, select_parallelism


def _args(**kw):
    kw.setdefault("mesh", None)
    kw.setdefault("devices", "")
    kw.setdefault("target", "some/model")
    return Namespace(**kw)


def test_chip_count_from_devices_and_mesh():
    assert O._optimize_chip_count(_args(devices="single")) == 1
    assert O._optimize_chip_count(_args(devices="0,1,2,3")) == 4
    assert O._optimize_chip_count(_args(mesh="1x4")) == 4
    assert O._optimize_chip_count(_args(mesh="2x4")) == 8
    assert O._optimize_chip_count(_args(devices="all")) is None
    assert O._optimize_chip_count(_args(devices="")) is None


def test_plan_parallelism_guards():
    assert plan_parallelism("m/x", 1) is None  # single chip -> no split
    assert plan_parallelism("", 4) is None  # no model to probe
    pc = select_parallelism(4, type("KR", (), {"tp_grid": [1, 2, 4], "has_blockers": lambda s, tp: False})())
    assert (pc.tp, pc.dp, pc.chips) == (4, 1, 4)


def _clear_env():
    os.environ.pop("TT_PERF_MESH_ROWS", None)
    os.environ.pop("TT_PERF_MESH_COLS", None)


def test_single_chip_exports_1x1():
    _clear_env()
    O._derive_topology_env(_args(devices="single"), model_dir=None)
    assert os.environ["TT_PERF_MESH_ROWS"] == "1" and os.environ["TT_PERF_MESH_COLS"] == "1"


def test_multichip_no_model_falls_back_to_1d(monkeypatch):
    _clear_env()
    O._derive_topology_env(_args(devices="0,1,2,3"), model_dir="/some/existing/dir")
    assert (os.environ["TT_PERF_MESH_ROWS"], os.environ["TT_PERF_MESH_COLS"]) == ("1", "4")


def test_multichip_kernel_viable_uses_shared_planner(monkeypatch):
    _clear_env()
    monkeypatch.setattr(
        "scripts.tt_hw_planner.parallelism.plan_parallelism", lambda mid, chips: ParallelConfig(tp=2, dp=2)
    )
    O._derive_topology_env(_args(devices="0,1,2,3"), model_dir=None)
    assert (os.environ["TT_PERF_MESH_ROWS"], os.environ["TT_PERF_MESH_COLS"]) == ("2", "2")  # rows=dp, cols=tp


def test_all_devices_does_not_force_a_shape():
    _clear_env()
    O._derive_topology_env(_args(devices="all"), model_dir=None)
    assert "TT_PERF_MESH_ROWS" not in os.environ


def test_emit_e2e_delegates_to_shared_planner():
    assert E._planned_parallelism("some/model", _args(mesh=None)) is None  # <=1 chip
    assert E._mesh_chip_count("2x4") == 8
