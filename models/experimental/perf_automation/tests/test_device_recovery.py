# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device recovery must act on EVIDENCE, verify the result, and give up loudly.

llama3_1_8b_p150, 2026-07-27: a candidate lever wedged the card mid-measure_candidate. Eleven hours
later the board was still dead and the run had spent the last four polling it, ~6 minutes per no-op
gate call. Three independent defects, each of which alone would have caused it:

  WHICH  the reset target came from `--devices single` -> chip 0 -> board 0,1. But that flag is
         INTENT (use one chip), not PLACEMENT, and _visible_devices leaves every chip visible, so the
         runtime had placed the mesh on chip 3. Every reset hit the healthy board. The error text said
         "Read 0xffffffff over PCIe ID 3" the whole time.

  WHETHER a two-strike counter gated the reset, held in the MCP process -- which the client kills
         whenever a call runs long, including the 7-minute tt-smi hang the reset itself caused. The
         count never reached two.

  WORKED? the outcome was written to a stderr nobody captures and otherwise discarded, so a failed
         reset was indistinguishable from a successful one and the loop never escalated.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _mcp(tmp_path, monkeypatch, healthy_after=0, **env):
    """Load perf_mcp with tt-smi stubbed. `healthy_after` = number of resets before the device
    reports healthy; 0 = healthy immediately, None = never recovers."""
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("pm_recovery_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pm_recovery_ut"] = mod
    spec.loader.exec_module(mod)
    state = {"resets": [], "healthy": healthy_after == 0}
    mod._board_reset = lambda where, note: (
        state["resets"].append(note),
        state.__setitem__("healthy", healthy_after is not None and len(state["resets"]) >= healthy_after),
    )
    mod._device_is_healthy = lambda: state["healthy"]
    return mod, state


# --- WHICH: the target comes from the error, with the config guess kept as fallback ------------


def test_chip_id_is_read_out_of_the_error(tmp_path, monkeypatch):
    m, _ = _mcp(tmp_path, monkeypatch)
    assert m._dead_chip_from_error("RuntimeError: Read 0xffffffff over PCIe ID 3: the board should be reset.") == 3
    assert m._dead_chip_from_error("Read 0xffffffff over PCIe device 0") == 0


def test_no_chip_id_in_the_error_is_not_a_guess(tmp_path, monkeypatch):
    m, _ = _mcp(tmp_path, monkeypatch)
    for txt in ("some other failure", "", None, "PCIe ID abc"):
        assert m._dead_chip_from_error(txt) is None


def test_the_named_chip_is_reset_first(tmp_path, monkeypatch):
    """THE REGRESSION: chip 3 died, chip 0/1 was reset."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1, PERF_MCP_DEVICES="single")
    assert m._recover_device("ut", "Read 0xffffffff over PCIe ID 3") is True
    assert "target=3" in st["resets"][0], st["resets"]
    assert "from error" in st["resets"][0]


def test_config_guess_is_kept_as_the_fallback(tmp_path, monkeypatch):
    """The flag-derived target is NOT removed -- it is demoted below the evidence."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=2, PERF_MCP_DEVICES="0,1")
    assert m._recover_device("ut", "no chip id in this error") is True
    assert "target=0,1" in " ".join(st["resets"]), st["resets"]


def test_all_boards_is_the_last_resort(tmp_path, monkeypatch):
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1)
    m._recover_device("ut", "unparseable")
    assert "target=all" in " ".join(st["resets"])


def test_target_order_is_evidence_then_config_then_all(tmp_path, monkeypatch):
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None, PERF_MCP_DEVICES="0,1")
    m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    order = [r.split("target=")[1].split(",")[0].rstrip(")") for r in st["resets"]]
    assert order[0] == "3", order
    assert "all" in " ".join(st["resets"])


# --- WHETHER: a definitive signature recovers immediately, no counter --------------------------


def test_dead_board_signature_recognised(tmp_path, monkeypatch):
    m, _ = _mcp(tmp_path, monkeypatch)
    for txt in ("Read 0xffffffff over PCIe ID 3", "the board should be reset", "PCIe link down", "device hang"):
        assert m._is_dead_board(txt), txt
    for txt in ("PCC 0.71 below floor", "L1 overflow", ""):
        assert not m._is_dead_board(txt), txt


def test_first_dead_board_crash_recovers_without_waiting_for_a_second(tmp_path, monkeypatch):
    """The counter could never reach 2: the MCP process is killed whenever a call runs long."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1)
    m._CONSEC_CRASH["n"] = 0
    m._note_device_crash("ut", "Read 0xffffffff over PCIe ID 3")
    assert len(st["resets"]) >= 1, "a definitive signature must not wait for a second strike"


def test_an_ambiguous_crash_still_uses_the_counter(tmp_path, monkeypatch):
    """A single odd failure is not evidence of a wedge; do not reset the board over it."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1)
    m._CONSEC_CRASH["n"] = 0
    m._note_device_crash("ut", "some transient assertion")
    assert st["resets"] == [], "reset fired on one ambiguous crash"
    m._note_device_crash("ut", "some transient assertion")
    assert len(st["resets"]) >= 1, "second ambiguous crash should recover"


# --- WORKED? verification and escalation ------------------------------------------------------


def test_a_failed_reset_is_reported_as_failure(tmp_path, monkeypatch):
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None)
    assert m._recover_device("ut", "Read 0xffffffff over PCIe ID 3") is False


def test_the_counter_is_only_cleared_on_a_verified_healthy_device(tmp_path, monkeypatch):
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None)
    m._CONSEC_CRASH["n"] = 5
    m._note_device_crash("ut", "Read 0xffffffff over PCIe ID 3")
    assert m._CONSEC_CRASH["n"] == 5, "counter cleared despite the reset failing"


def test_repeated_failures_escalate_to_exhausted(tmp_path, monkeypatch):
    """An unrecoverable board must end the run, not produce hours of no-op polling."""
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None)
    m._RESET_FAILS["n"] = 0
    assert not m._recovery_exhausted()
    for _ in range(m._RESET_FAIL_LIMIT):
        m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    assert m._recovery_exhausted()


def test_a_successful_recovery_clears_the_failure_count(tmp_path, monkeypatch):
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=1)
    m._RESET_FAILS["n"] = 2
    assert m._recover_device("ut", "Read 0xffffffff over PCIe ID 3") is True
    assert m._RESET_FAILS["n"] == 0 and not m._recovery_exhausted()
