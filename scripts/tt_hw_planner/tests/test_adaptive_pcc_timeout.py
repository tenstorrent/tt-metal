# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The per-run PCC pytest wall must scale with the mesh a run spans, not be one flat value.

A sharded run compiles kernels for and runs collectives across every chip it opens, so it
legitimately outlasts a single-device run. Under one flat wall it was hard-killed with rc=124
mid-compile, which parses as no report at all -> classified ``OTHER`` -> the component never
graduates and is re-queued forever. These pin that the wall grows with the chip count, that a
single-device run is left exactly as it was, and that the scaling shares ONE formula with the
agent budget (`cli._scaled_timeout`) instead of a second copy."""
import importlib

import pytest

from scripts.tt_hw_planner import cli


@pytest.fixture()
def bmcp(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    monkeypatch.delenv("BRINGUP_MCP_TIMEOUT_MODE", raising=False)
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    (tmp_path / "_stubs").mkdir(parents=True, exist_ok=True)
    return m, tmp_path


# --- the shared formula ---------------------------------------------------------------


def test_scaled_timeout_is_the_single_formula() -> None:
    """`_scaled_timeout` is base + step*bonus capped at base + max_extra, and a
    non-positive base or bonus passes through untouched (a bonus only ever ADDS)."""
    f = cli._scaled_timeout
    assert f(1800, 0, step_s=900, max_extra_s=3600) == 1800
    assert f(1800, 1, step_s=900, max_extra_s=3600) == 2700
    assert f(1800, 4, step_s=900, max_extra_s=3600) == 1800 + 3600
    assert f(1800, 99, step_s=900, max_extra_s=3600) == 1800 + 3600, "must clamp at max_extra_s"
    assert f(0, 5, step_s=900, max_extra_s=3600) == 0, "unbounded base must stay unbounded"
    assert f(1800, -1, step_s=900, max_extra_s=3600) == 1800


def test_agent_budget_still_uses_its_own_step_and_cap() -> None:
    """Extracting the shared helper must not have changed the agent budget's contract
    (+5 min/unit, +20 min cap) — the two budgets are independent quantities."""
    f = cli._agent_complexity_timeout
    assert f(900, 0) == 900
    assert f(900, 1) == 900 + 300
    assert f(900, 4) == 2100
    assert f(900, 99) == 2100
    assert f(0, 0) == 0


# --- the per-run PCC wall ------------------------------------------------------------


def test_single_device_keeps_base_exactly(bmcp, monkeypatch):
    """A single-device run gets bonus 0, so the flat base is preserved and nothing that
    already passes within it is affected."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    assert m._mesh_degree_bonus(shard=False) == 0
    assert m._adaptive_pcc_timeout(shard=False) == 1800


def test_shard_adds_one_unit_per_extra_chip(bmcp, monkeypatch):
    """TP=2 x DP=1 spans 2 chips -> 1 extra chip -> one step of extra wall."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 2)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    assert m._mesh_degree_bonus(shard=True) == 1
    assert m._adaptive_pcc_timeout(shard=True) == 1800 + m._PCC_TIMEOUT_STEP_S


def test_wider_mesh_scales_further(bmcp, monkeypatch):
    """TP=2 x DP=2 spans 4 chips -> 3 extra chips -> three steps."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 2)
    monkeypatch.setattr(m, "_SHARD_DP", 2)
    assert m._mesh_degree_bonus(shard=True) == 3
    assert m._adaptive_pcc_timeout(shard=True) == 1800 + 3 * m._PCC_TIMEOUT_STEP_S


def test_hard_cap_bounds_a_hung_run(bmcp, monkeypatch):
    """An absurdly wide mesh is still capped, so a genuinely hung run dies in bounded
    time instead of holding the device indefinitely."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 8)
    monkeypatch.setattr(m, "_SHARD_DP", 4)
    assert m._adaptive_pcc_timeout(shard=True) == 1800 + m._PCC_TIMEOUT_MAX_EXTRA_S


def test_fixed_mode_restores_flat_wall(bmcp, monkeypatch):
    """The escape hatch returns the exact pre-change behaviour."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 4)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    monkeypatch.setenv("BRINGUP_MCP_TIMEOUT_MODE", "fixed")
    assert m._adaptive_pcc_timeout(shard=True) == 1800


def test_unbounded_base_stays_unbounded(bmcp, monkeypatch):
    """timeout_s=0 means "no wall"; the mesh bonus must not invent one."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 0)
    monkeypatch.setattr(m, "_SHARD_TP", 4)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    assert m._adaptive_pcc_timeout(shard=True) == 0


def test_single_chip_shard_keeps_base(bmcp, monkeypatch):
    """Declared TP=1 x DP=1 is still one chip: bonus 0, wall unchanged."""
    m, _ = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 1)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    assert m._mesh_degree_bonus(shard=True) == 0
    assert m._adaptive_pcc_timeout(shard=True) == 1800


def test_no_hardcoded_model_vocabulary_in_the_scaling(bmcp):
    """The wall must be derived from declared parallelism only — never from a component
    or stage name, or a config shape field. Pins constraint 3."""
    import inspect

    m, _ = bmcp
    src = inspect.getsource(m._mesh_degree_bonus) + inspect.getsource(m._adaptive_pcc_timeout)
    for forbidden in ("hidden_size", "num_hidden_layers", "decode", "prefill", "encoder", "decoder", "attention"):
        assert forbidden not in src, f"scaling must not reference {forbidden!r}"


# --- wiring + the timeout error path --------------------------------------------------


def _stub_focused_pytest(m, monkeypatch, tmp, *, rc: int, shard: bool):
    """Point `_run_pcc` at a fake test file and a fake pytest so we can pin the wall
    it asks for and the report it produces, without touching a device."""
    captured: dict = {}
    monkeypatch.setattr(m, "_test_file_for", lambda _c: str(tmp / "test_comp.py"))
    monkeypatch.setattr(m, "_ensure_shard_test", lambda _c: str(tmp / "test_comp_sharded.py"))
    if shard:
        monkeypatch.setenv("TT_HW_PLANNER_SHARD_RUN", "1")
    else:
        monkeypatch.delenv("TT_HW_PLANNER_SHARD_RUN", raising=False)

    def fake_pytest(**kwargs):
        captured.update(kwargs)
        return rc

    monkeypatch.setattr(m._cli, "_run_focused_pytest", fake_pytest)
    monkeypatch.setattr(
        m._cli,
        "_parse_pytest_report",
        lambda: (_ for _ in ()).throw(AssertionError("JUnit must not be parsed after a wall kill")),
    )
    monkeypatch.setattr(m._cli, "_scope_report_to_demo", lambda report, _demo: report)
    return captured


def test_run_pcc_passes_the_scaled_wall(bmcp, monkeypatch):
    """The helper is not enough — `_run_pcc` must actually hand the scaled value to
    the pytest runner, or sharded runs still die at the flat 1800s."""
    m, tmp = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 2)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    captured = _stub_focused_pytest(m, monkeypatch, tmp, rc=0, shard=True)
    # A passing run DOES parse JUnit; replace the kill-guard with an empty report.
    monkeypatch.setattr(
        m._cli,
        "_parse_pytest_report",
        lambda: {
            "passed_components": ["comp_sharded"],
            "failed_components": [],
            "skipped_components": [],
            "summary": "",
            "details": "",
            "per_skipped": {},
        },
    )
    res = m._run_pcc("comp")
    assert captured["timeout_s"] == 1800 + m._PCC_TIMEOUT_STEP_S
    assert res["passed"] is True


def test_wall_kill_classifies_as_hang_not_other(bmcp, monkeypatch):
    """rc=124 used to become OTHER (no parseable report) and re-queue forever.
    Folding the hang phrase into the report lets the existing classifier return
    HANG, and we must not read a stale previous JUnit XML."""
    m, tmp = bmcp
    monkeypatch.setattr(m, "_TIMEOUT", 1800)
    monkeypatch.setattr(m, "_SHARD_TP", 2)
    monkeypatch.setattr(m, "_SHARD_DP", 1)
    captured = _stub_focused_pytest(m, monkeypatch, tmp, rc=124, shard=True)
    res = m._run_pcc("comp")
    assert captured["timeout_s"] == 1800 + m._PCC_TIMEOUT_STEP_S
    assert res["passed"] is False
    assert res["failed"] is True
    assert cli._classify_failure(res["summary"], res["details"]) == "HANG"
    assert cli._classify_failure(res["summary"], res["details"]) != "OTHER"
