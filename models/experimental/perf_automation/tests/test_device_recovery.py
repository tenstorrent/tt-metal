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


def _mcp(tmp_path, monkeypatch, healthy_after=0, state_file=None, **env):
    """Load perf_mcp with tt-smi stubbed. `healthy_after` = number of resets before the device
    reports healthy; 0 = healthy immediately, None = never recovers.

    `state["resets"]` records the TARGET each reset was actually issued against, not the log note it
    was labelled with. The first version of this fixture recorded the note, and so happily passed
    while the real _board_reset ignored the target entirely and reset the configured board every time
    -- the exact defect these tests exist to catch, printed rather than performed. Assert on the
    argument that reaches the reset command.

    `state_file` pins the persistent-counter path so a test can reload the module and still see the
    counters a previous 'process' wrote.

    THE FIXTURE OWNS THE ENVIRONMENT IT DEPENDS ON. Every test here that wants a device config passes
    one; test_all_boards_is_the_last_resort wants NO config, and expressed that by saying nothing --
    which is not the same thing. It inherited PERF_MCP_DEVICES from whatever shell ran pytest, so it
    passed in a clean terminal and failed under the optimize run, whose environment sets it: the
    config target won and "all" was never reached. Found by the preflight, which runs the suite in
    the run's OWN environment -- the more faithful place, and the reason it saw what a clean shell
    could not. Cleared here so "no config" is stated rather than assumed, and any test wanting one
    sets it through **env below.
    """
    for _var in ("PERF_MCP_DEVICES", "TT_VISIBLE_DEVICES"):
        if _var not in env:
            monkeypatch.delenv(_var, raising=False)
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("pm_recovery_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pm_recovery_ut"] = mod
    spec.loader.exec_module(mod)
    devstate = Path(state_file) if state_file else (tmp_path / "devstate.json")
    state = {"resets": [], "healthy": healthy_after == 0}
    dr = mod._dr()
    monkeypatch.setattr(dr, "state_path", lambda: devstate)
    monkeypatch.setattr(dr, "device_is_healthy", lambda *a, **k: state["healthy"])
    monkeypatch.setattr(dr, "board_map", lambda: {"0": [0, 1], "1": [0, 1], "2": [2, 3], "3": [2, 3]})

    def _fake_reset(where, note, target=""):
        state["resets"].append(target)
        state["healthy"] = healthy_after is not None and len(state["resets"]) >= healthy_after
        return True

    mod._board_reset = _fake_reset
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
    assert st["resets"][0] == "2,3", st["resets"]


def test_config_guess_is_kept_as_the_fallback(tmp_path, monkeypatch):
    """The flag-derived target is NOT removed -- it is demoted below the evidence."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=2, PERF_MCP_DEVICES="0,1")
    assert m._recover_device("ut", "no chip id in this error") is True
    assert "0,1" in st["resets"], st["resets"]


def test_all_boards_is_the_last_resort(tmp_path, monkeypatch):
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1)
    m._recover_device("ut", "unparseable")
    assert "all" in st["resets"], st["resets"]


def test_target_order_is_evidence_then_config_then_all(tmp_path, monkeypatch):
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None, PERF_MCP_DEVICES="0,1")
    m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    assert st["resets"] == ["2,3", "0,1", "all"], st["resets"]


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


def _mcp_real_board_reset(tmp_path, monkeypatch, **env):
    """Load perf_mcp with the REAL _board_reset, stubbing only run.py's _reset_devices, so the test
    observes the spec that actually reaches the reset."""
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(tmp_path / "m.json"))
    (tmp_path / "m.json").write_text('{"config": {}, "perf_test_resolved": {"path": "t.py"}}')
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("pm_realreset_ut", _ROOT / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pm_realreset_ut"] = mod
    spec.loader.exec_module(mod)
    dr = mod._dr()
    monkeypatch.setattr(dr, "state_path", lambda: tmp_path / "devstate.json")
    monkeypatch.setattr(dr, "device_is_healthy", lambda *a, **k: True)
    monkeypatch.setattr(dr, "board_map", lambda: {"0": [0, 1], "1": [0, 1], "2": [2, 3], "3": [2, 3]})
    seen = []

    class _FakeRun:
        @staticmethod
        def _reset_devices(devices):
            seen.append(devices)
            return "reset ok"

    mod._run_module = lambda: _FakeRun
    mod._device_is_healthy = lambda: True
    return mod, seen


def test_board_reset_issues_the_requested_target(tmp_path, monkeypatch):
    """THE DEFECT THE FIRST FIX SHIPPED WITH: the target was computed, put in the log note, and
    dropped -- the message said target=3 while board 0,1 was reset."""
    m, seen = _mcp_real_board_reset(tmp_path, monkeypatch, PERF_MCP_DEVICES="0,1")
    m._board_reset("ut", "note", target="3")
    assert seen == ["3"], seen


def test_board_reset_falls_back_to_config_then_all(tmp_path, monkeypatch):
    m, seen = _mcp_real_board_reset(tmp_path, monkeypatch, PERF_MCP_DEVICES="0,1")
    m._board_reset("ut", "note")
    assert seen == ["0,1"], seen
    monkeypatch.delenv("PERF_MCP_DEVICES")
    m._board_reset("ut", "note")
    assert seen[-1] == "all", seen


def test_recover_device_end_to_end_resets_the_named_chip(tmp_path, monkeypatch):
    """End-to-end through the real _board_reset: chip 3 in the error -> `-r 3`, not the config board."""
    m, seen = _mcp_real_board_reset(tmp_path, monkeypatch, PERF_MCP_DEVICES="single")
    assert m._recover_device("ut", "Read 0xffffffff over PCIe ID 3: board should be reset") is True
    assert seen == ["2,3"], seen


def test_board_reset_reports_failure(tmp_path, monkeypatch):
    """`_sp` is the real subprocess module, shared by the whole pytest process: patch it through
    monkeypatch so it is restored. A bare assignment here left every later test's subprocess.run
    returning rc=1, which broke unrelated git-backed suites long after this test had passed."""
    m, _ = _mcp_real_board_reset(tmp_path, monkeypatch)
    m._run_module = lambda: None
    monkeypatch.setattr(m._sp, "run", lambda *a, **k: type("R", (), {"returncode": 1, "stdout": "", "stderr": ""})())
    assert m._board_reset("ut", "note", target="3") is False


def test_crash_counter_survives_a_process_restart(tmp_path, monkeypatch):
    """WHETHER, root cause: the counter lived in the MCP process, which the client kills whenever a
    call runs long -- and a wedged device is exactly what makes a call run long. Reaching two strikes
    was impossible."""
    shared = tmp_path / "shared_devstate.json"
    m1, _ = _mcp(tmp_path, monkeypatch, healthy_after=None, state_file=shared)
    m1._note_device_crash("ut", "ambiguous failure")
    assert m1._CONSEC_CRASH["n"] == 1
    m2, st2 = _mcp(tmp_path, monkeypatch, healthy_after=None, state_file=shared)
    assert m2._CONSEC_CRASH["n"] == 1, "counter reset to zero on restart -- can never reach the limit"
    m2._note_device_crash("ut", "ambiguous failure")
    assert st2["resets"], "second strike after a restart did not trigger recovery"


def test_reset_fail_budget_survives_a_process_restart(tmp_path, monkeypatch):
    """Escalation is only real if its budget is durable: a per-process limit that resets on every
    restart never trips, and the loop polls a dead board forever."""
    shared = tmp_path / "shared_devstate.json"
    for _ in range(3):
        m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None, state_file=shared)
        m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None, state_file=shared)
    assert m._recovery_exhausted(), "budget forgotten across restarts"


def test_counter_state_is_keyed_per_run(tmp_path, monkeypatch):
    """Two concurrent optimize runs must not share a crash history."""
    m, _ = _mcp(tmp_path, monkeypatch)
    dr = m._dr()
    monkeypatch.undo()
    monkeypatch.delenv("TT_RECOVERY_STATE", raising=False)
    monkeypatch.setenv("PERF_MCP_TASK", "task_a")
    p_a = dr.state_path()
    monkeypatch.setenv("PERF_MCP_TASK", "task_b")
    p_b = dr.state_path()
    assert p_a != p_b, (p_a, p_b)


def test_corrupt_state_file_does_not_break_recovery(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1, state_file=bad)
    assert m._CONSEC_CRASH["n"] == 0
    m._note_device_crash("ut", "Read 0xffffffff over PCIe ID 2")
    assert st["resets"] == ["2,3"], st["resets"]


def test_reclaim_mesh_verifies_and_reports(tmp_path, monkeypatch):
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1)
    assert m._reclaim_mesh("profile_model") is True
    assert st["resets"], "reclaim did not reset"


def test_reclaim_mesh_does_not_launder_a_failed_reset(tmp_path, monkeypatch):
    """It used to clear the crash counter unconditionally, so a reclaim that failed was recorded as a
    success and erased the history that would have escalated."""
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None)
    m._CONSEC_CRASH["n"] = 4
    assert m._reclaim_mesh("profile_model") is False
    assert m._CONSEC_CRASH["n"] == 4, "counter cleared by a reclaim that never brought the device back"


def test_reclaim_mesh_spends_the_escalation_budget(tmp_path, monkeypatch):
    """A hot loop reclaiming a dead mesh must reach the same give-up point as any other reset."""
    m, _ = _mcp(tmp_path, monkeypatch, healthy_after=None)
    for _ in range(3):
        m._reclaim_mesh("measure_candidate")
    assert m._recovery_exhausted()


def test_device_recover_goes_through_the_verified_path(tmp_path, monkeypatch):
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None)
    assert m._device_recover("ut") is False
    assert st["resets"], "no reset attempted"


def test_named_chip_expands_to_its_whole_board(tmp_path, monkeypatch):
    """Reading the right chip out of the error is only half the fix: a bare `-r 3` half-resets a
    p300c and leaves the other ASIC's clock arbiter inconsistent, which wedges device-open. The
    reset must cover the board, and expansion is the DEFAULT so no caller can forget it."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1, PERF_MCP_DEVICES="single")
    m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    assert st["resets"][0] == "2,3", st["resets"]
    assert st["resets"][0] != "3", "half-board reset issued"


def test_without_topology_it_widens_to_all_never_a_bare_chip(tmp_path, monkeypatch):
    """THE DANGEROUS CASE, and the common one.

    board_map() live-reads the topology from the card -- so during a real recovery, when the card is
    the thing that died, the topology is exactly what is NOT available. An earlier version of this
    fix fell back to the bare chip named in the error, which issues `-r 3`: a half-reset of a p300c
    that leaves the other ASIC's clock arbiter inconsistent and wedges the next device-open. When a
    target cannot be widened with confidence it must widen to every board, never narrow to one chip.
    """
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=1)
    monkeypatch.setattr(m._dr(), "board_map", lambda: None)
    m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    assert st["resets"] == ["all"], st["resets"]
    assert "3" not in st["resets"], "bare single-chip reset issued"


def test_no_target_is_ever_a_single_chip_of_a_multi_chip_board(tmp_path, monkeypatch):
    """The invariant, stated directly: every target reaches tt-smi as a whole board or as `all`."""
    board = {"0": [0, 1], "1": [0, 1], "2": [2, 3], "3": [2, 3]}
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None, PERF_MCP_DEVICES="2")
    monkeypatch.setattr(m._dr(), "board_map", lambda: board)
    m._recover_device("ut", "Read 0xffffffff over PCIe ID 3")
    whole_boards = {",".join(str(x) for x in v) for v in board.values()} | {"all"}
    for tgt in st["resets"]:
        assert tgt in whole_boards, "target %r is not a whole board" % tgt


def test_config_target_is_widened_too(tmp_path, monkeypatch):
    """The --devices guess is a chip spec as well: `--devices 2` must reset board 2,3, not chip 2."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None, PERF_MCP_DEVICES="2")
    m._recover_device("ut", "no chip named here")
    assert st["resets"] == ["2,3", "all"], st["resets"]


def test_single_means_board_zero_not_chip_zero(tmp_path, monkeypatch):
    """`--devices single` is INTENT (use one chip), and its board is still a whole board."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None, PERF_MCP_DEVICES="single")
    m._recover_device("ut", "no chip named here")
    assert st["resets"] == ["0,1", "all"], st["resets"]


def test_unparseable_config_target_widens_to_all(tmp_path, monkeypatch):
    """An unrecognised spec must not be passed through to tt-smi verbatim."""
    m, st = _mcp(tmp_path, monkeypatch, healthy_after=None, PERF_MCP_DEVICES="weird-spec")
    m._recover_device("ut", "no chip named here")
    assert st["resets"] == ["all"], st["resets"]
