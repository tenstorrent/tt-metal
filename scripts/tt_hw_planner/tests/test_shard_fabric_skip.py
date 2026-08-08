import importlib
import os

import pytest


@pytest.fixture()
def gate(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_HW_PLANNER_SHARD", "1")
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    return m


def test_is_fabric_failure_matches_signatures(gate):
    assert gate._is_fabric_failure("... Fabric Router Sync: Timeout ...")
    assert gate._is_fabric_failure("crash in fabric_firmware_initializer.cpp:42")
    assert gate._is_fabric_failure("recorded fabric_unavailable for comp")
    assert not gate._is_fabric_failure("PCC 0.87 below threshold")
    assert not gate._is_fabric_failure("")


def test_pending_shard_skipped_when_fabric_unhealthy(gate, monkeypatch):
    monkeypatch.setattr(gate, "_is_graduated", lambda c: True)
    monkeypatch.setattr(gate, "_is_shard_graduated", lambda c: False)
    monkeypatch.setattr(gate._shard, "is_shard_eligible", lambda c: True)
    comps = ["decoder_layer"]
    assert gate._pending_shard_component(comps) == "decoder_layer"
    st = gate._load_state()
    st["fabric_unhealthy"] = True
    gate._save_state(st)
    assert gate._pending_shard_component(comps) is None
