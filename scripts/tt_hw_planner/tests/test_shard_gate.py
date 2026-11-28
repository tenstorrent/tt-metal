# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Piece B of shard-aware bring-up: the gate's shard phase must be strictly opt-in (TT_HW_PLANNER_SHARD)
and only fire AFTER single-device graduation, for shard-eligible components lacking a shard snapshot.
Flag off => the gate behaves exactly as before (this is the non-breaking guarantee)."""
import importlib
import os

import pytest


@pytest.fixture()
def bmcp(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    (tmp_path / "_stubs").mkdir(parents=True, exist_ok=True)
    return m, tmp_path


def _native(tmp, c):
    (tmp / "_stubs" / f"{c}.py.last_good_native").write_text("x")


def _sharded(tmp, c):
    (tmp / "_stubs" / f"{c}.py.last_good_sharded").write_text("x")


def test_flag_default_off(bmcp, monkeypatch):
    m, _ = bmcp
    monkeypatch.delenv("TT_HW_PLANNER_SHARD", raising=False)
    assert m._shard_enabled() is False


def test_flag_on(bmcp, monkeypatch):
    m, _ = bmcp
    monkeypatch.setenv("TT_HW_PLANNER_SHARD", "1")
    assert m._shard_enabled() is True


def test_pending_shard_only_for_graduated_eligible(bmcp):
    m, tmp = bmcp
    _native(tmp, "self_attn")
    _native(tmp, "nemotron_h_mamba2_mixer")
    assert m._pending_shard_component(["self_attn", "nemotron_h_mamba2_mixer"]) == "self_attn"


def test_pending_skips_replicate_only(bmcp):
    m, tmp = bmcp
    _native(tmp, "input_layernorm")
    _native(tmp, "embed_tokens")
    assert m._pending_shard_component(["input_layernorm", "embed_tokens"]) is None


def test_pending_includes_derive_layers(bmcp):
    m, tmp = bmcp
    _native(tmp, "nemotron_h_mamba2_mixer")
    assert m._pending_shard_component(["nemotron_h_mamba2_mixer"]) == "nemotron_h_mamba2_mixer"


def test_pending_skips_ungraduated(bmcp):
    m, tmp = bmcp
    assert m._pending_shard_component(["self_attn"]) is None


def test_pending_none_once_sharded(bmcp):
    m, tmp = bmcp
    _native(tmp, "mlp")
    _sharded(tmp, "mlp")
    assert m._is_shard_graduated("mlp") is True
    assert m._pending_shard_component(["mlp"]) is None


def test_termination_check_no_shard_rung_when_flag_off(bmcp, monkeypatch):
    m, tmp = bmcp
    monkeypatch.delenv("TT_HW_PLANNER_SHARD", raising=False)
    _native(tmp, "self_attn")
    monkeypatch.setattr(m, "_components", lambda: ["self_attn"])
    r = m.termination_check()
    assert r["can_stop"] is True
    assert r["next_target"] is None
    assert r["shard_graduated"] == []


def test_termination_check_emits_shard_rung_when_flag_on(bmcp, monkeypatch):
    m, tmp = bmcp
    monkeypatch.setenv("TT_HW_PLANNER_SHARD", "1")
    _native(tmp, "self_attn")
    monkeypatch.setattr(m, "_components", lambda: ["self_attn"])
    r = m.termination_check()
    assert r["can_stop"] is False
    assert r["next_target"]["rung"] == "shard"
    assert r["next_target"]["unit"] == "self_attn"


def test_single_phase_fresh_shard_eligible_goes_straight_to_shard(bmcp, monkeypatch):
    m, tmp = bmcp
    monkeypatch.setenv("TT_HW_PLANNER_SHARD", "1")
    monkeypatch.setattr(m, "_components", lambda: ["self_attn"])
    r = m.termination_check()
    assert r["can_stop"] is False
    assert r["next_target"]["rung"] == "shard"
    assert r["next_target"]["unit"] == "self_attn"


def test_single_phase_flag_off_fresh_component_is_single_device(bmcp, monkeypatch):
    m, tmp = bmcp
    monkeypatch.delenv("TT_HW_PLANNER_SHARD", raising=False)
    monkeypatch.setattr(m, "_components", lambda: ["self_attn"])
    r = m.termination_check()
    assert r["next_target"]["rung"] in ("emit", "repair")


def test_single_phase_replicate_only_stays_single_device_when_tp_gt_1(bmcp, monkeypatch):
    m, tmp = bmcp
    monkeypatch.setenv("TT_HW_PLANNER_SHARD", "1")
    monkeypatch.setattr(m, "_components", lambda: ["input_layernorm"])
    r = m.termination_check()
    assert r["next_target"]["rung"] in ("emit", "repair")


def test_get_shard_plan_eligible_returns_guidance(bmcp):
    m, _ = bmcp
    p = m.get_shard_plan("self_attn")
    assert p["eligible"] is True
    assert "principles" in p and "reference_hints" in p
    assert "tt_transformers" in p["reference_hints"]
    assert "weights" not in p and "role" not in p


def test_get_shard_plan_derive_layer_also_eligible(bmcp):
    m, _ = bmcp
    p = m.get_shard_plan("nemotron_h_m_o_e")
    assert p["eligible"] is True
    assert "expert" in p["principles"].lower()


def test_get_shard_plan_replicate_only(bmcp):
    m, _ = bmcp
    p = m.get_shard_plan("input_layernorm")
    assert p["eligible"] is False


def test_fabric_failure_triggers_device_reset():
    from scripts.tt_hw_planner.cli import _output_indicates_device_reset_needed as needs_reset

    assert needs_reset("… Fabric Router Sync: Timeout after 10000 ms on Device 2 …") is True
    assert needs_reset("fabric_firmware_initializer.cpp:263: tt::exception") is True
    assert needs_reset("fabric_unavailable") is True
    assert needs_reset("PCC 0.83 below target") is False


def test_shard_mode_resets_fabric_once(bmcp, monkeypatch):
    m, tmp = bmcp
    (tmp / "_stubs" / "self_attn.py").write_text("x")
    resets = {"n": 0}
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: resets.__setitem__("n", resets["n"] + 1) or True)
    monkeypatch.setattr(
        m,
        "_run_pcc",
        lambda c: {
            "ran": True,
            "passed": False,
            "failed": True,
            "skipped": False,
            "summary": "",
            "details": "",
            "skip_reason": "",
        },
    )
    m.run_component("self_attn", mode="shard")
    m.run_component("self_attn", mode="shard")
    assert resets["n"] == 1
    assert m._load_state().get("shard_reset_done") is True


def test_single_mode_never_resets_fabric(bmcp, monkeypatch):
    m, tmp = bmcp
    (tmp / "_stubs" / "self_attn.py").write_text("x")
    resets = {"n": 0}
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: resets.__setitem__("n", resets["n"] + 1) or True)
    monkeypatch.setattr(
        m,
        "_run_pcc",
        lambda c: {
            "ran": True,
            "passed": False,
            "failed": True,
            "skipped": False,
            "summary": "",
            "details": "",
            "skip_reason": "",
        },
    )
    m.run_component("self_attn", mode="single")
    assert resets["n"] == 0


def test_shard_record_writes_sharded_snapshot_not_native(bmcp):
    m, tmp = bmcp
    stub = tmp / "_stubs" / "self_attn.py"
    stub.write_text("native body")
    r = m.record_result("self_attn", ok=True, pcc=0.999, mode="shard")
    assert r["shard_graduated"] is True and r["mode"] == "shard"
    assert (tmp / "_stubs" / "self_attn.py.last_good_sharded").is_file()
    assert not (tmp / "_stubs" / "self_attn.py.last_good_native").is_file()


def test_shard_record_fail_bumps_attempts_only(bmcp):
    m, tmp = bmcp
    (tmp / "_stubs" / "mlp.py").write_text("x")
    m.record_result("mlp", ok=False, pcc=0.3, mode="shard")
    st = m._load_state()
    assert st["shard_attempts"]["mlp"] == 1
    assert not (tmp / "_stubs" / "mlp.py.last_good_sharded").is_file()


def test_shard_cap_stops_pending(bmcp):
    m, tmp = bmcp
    _native(tmp, "self_attn")
    for _ in range(m._HARD_CAP):
        m.record_result("self_attn", ok=False, mode="shard")
    assert m._pending_shard_component(["self_attn"]) is None
