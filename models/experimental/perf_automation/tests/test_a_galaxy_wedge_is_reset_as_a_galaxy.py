# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A wedged Galaxy is recognised, and reset the way a Galaxy resets.

2026-09-25, WH Galaxy (32 chips): an e2e run hung mid-collective and every later run failed at
device-open with "NOC0 is hung on PCIe device ID 9." Three things kept the board wedged:
  * that text matched no dead-board signature, so the temperature veto (the ARC was still
    publishing) cancelled every reset;
  * the chip id was not read out of it ("PCIe device ID 9" has both words);
  * a stage that never ran note_board had _GALAXY_HOST=None, so the reset was a plain `-r`,
    which does not reset a Galaxy.
"""
import subprocess

import pytest

from agent import device_recovery as dr
from agent import probes as P

NOC_HANG = "RuntimeError: NOC0 is hung on PCIe device ID 9.\nLocation: umd/device/tt_device/tt_device.cpp:90"
GLX = [["-glx_reset_auto"], ["-glx_reset"]]


def test_the_noc_hang_is_a_dead_board_signature():
    assert dr.is_dead_board(NOC_HANG)


def test_the_noc_hang_names_its_chip():
    assert dr.dead_chip_from_error(NOC_HANG) == 9


def test_the_older_chip_id_forms_still_parse():
    assert dr.dead_chip_from_error("Read 0xffffffff over PCIe ID 3: the board should be reset.") == 3
    assert dr.dead_chip_from_error("pcie device 5 stopped answering") == 5
    assert dr.dead_chip_from_error("PCIe: 7") == 7
    assert dr.dead_chip_from_error("an assertion about tensor shapes") is None


@pytest.fixture
def host(monkeypatch):
    """Pin the host kind, so no result depends on the hardware the suite happens to run on."""
    monkeypatch.delenv("TT_HW_PLANNER_RESET_ARGS", raising=False)
    monkeypatch.delenv("TT_HW_PLANNER_GALAXY", raising=False)

    def _set(galaxy, chips, card=""):
        monkeypatch.setattr(P, "_GALAXY_HOST", galaxy, raising=False)
        monkeypatch.setattr(P, "_enumerated_device_count", lambda: chips)
        monkeypatch.setattr(P, "_sysfs_card_types", lambda: card)

    return _set


def test_a_galaxy_tries_its_galaxy_resets_before_a_chip_targeted_r(host):
    host(True, 32)
    assert P.reset_commands("8,9") == GLX + [["-r", "8,9"]]
    assert P.reset_commands("all") == GLX + [["-r"]]


def test_a_plain_board_resets_exactly_the_target(host):
    host(False, 4)
    assert P.reset_commands("2,3") == [["-r", "2,3"]]
    assert P.reset_commands("") == [["-r", "0,1,2,3"]]


def test_an_explicit_override_is_never_rewritten(host, monkeypatch):
    host(False, 4)
    monkeypatch.setenv("TT_HW_PLANNER_RESET_ARGS", "-glx_reset_tray 2")
    assert P.reset_commands("2,3") == [["-glx_reset_tray", "2"]]


def _lying_probe(smi):
    raise AssertionError("the tt-smi probe ran at reset time; on a wedged Galaxy it answers 'not a Galaxy'")


def test_a_stage_that_never_noted_the_board_detects_a_wedged_galaxy_from_the_driver(host, monkeypatch):
    """What this host's driver publishes while wedged; the tt-smi probe must not be asked."""
    host(None, 32, card="galaxy-wormhole")
    monkeypatch.setattr(P, "_galaxy_capability_probe", _lying_probe)
    assert P.reset_commands("9")[:2] == GLX


def test_the_chip_count_alone_still_marks_a_galaxy(host, monkeypatch):
    host(None, 32)
    monkeypatch.setattr(P, "_galaxy_capability_probe", _lying_probe)
    assert P.reset_commands("9")[:2] == GLX


def test_a_plain_board_is_not_mistaken_for_a_galaxy_at_reset_time(host, monkeypatch):
    host(None, 4, card="n300")
    monkeypatch.setattr(P, "_galaxy_capability_probe", _lying_probe)
    assert P.reset_commands("2,3") == [["-r", "2,3"]]


def test_the_driver_card_type_is_read_from_sysfs(monkeypatch, tmp_path):
    for i, kind in enumerate(["galaxy-wormhole", "galaxy-wormhole"]):
        (tmp_path / ("tenstorrent!%d" % i)).mkdir()
        (tmp_path / ("tenstorrent!%d" % i) / "tt_card_type").write_text(kind + "\n")
    monkeypatch.setattr(P, "_SYSFS_TT_CLASS", str(tmp_path))
    assert P._sysfs_card_types() == "galaxy-wormhole"
    monkeypatch.setattr(P, "_SYSFS_TT_CLASS", str(tmp_path / "absent"))
    assert P._sysfs_card_types() == ""


def test_the_detection_is_made_once(host, monkeypatch):
    host(False, 4)

    def _reprobe(*a, **k):
        raise AssertionError("the board was re-probed after the decision was made")

    monkeypatch.setattr(P, "note_board", _reprobe)
    P.ensure_board_noted()


def test_a_detection_that_cannot_run_does_not_block_the_reset(host, monkeypatch):
    host(None, 4)

    def _boom(*a, **k):
        raise RuntimeError("tt-smi missing")

    monkeypatch.setattr(P, "note_board", _boom)
    assert P.reset_commands("2,3") == [["-r", "2,3"]]


def test_a_noc_hang_on_a_live_galaxy_is_reset_with_the_galaxy_reset(host, monkeypatch, tmp_path):
    """The whole regression: every chip still reports a temperature and the failure is a NOC hang."""
    host(True, 32)
    monkeypatch.setattr(dr, "state_path", lambda: tmp_path / "recovery.json")
    monkeypatch.setattr(dr, "reap_device_holders", lambda: [])
    monkeypatch.setattr(dr, "_board_needs_reset", lambda: False)  # the ARC is alive on every chip
    monkeypatch.setattr(dr, "board_map", lambda: {"9": [9]})
    monkeypatch.setattr(dr, "device_is_healthy", lambda *a, **k: True)
    seen = []

    class _Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(P.subprocess, "run", lambda cmd, **k: seen.append(list(cmd[1:])) or _Done())
    assert P._device_reset(error_text=NOC_HANG) is True
    assert seen and seen[0] == GLX[0], seen


# --- a host tool the reset needs is installed when the reset reports it missing (like tt-lang) -----
#
# The tool's name is never typed in the code: tt-smi names it ("sudo: <tool>: command not found").
# TOOL is test data standing in for whatever tt-smi reports.

TOOL = "bmc-tool-x"
REAL_FAILURE = "Resetting WH Galaxy trays with reset command...\nsudo: ipmitool: command not found\n"


def test_the_missing_tool_is_read_from_the_failure():
    assert P.missing_host_tool(REAL_FAILURE) == "ipmitool"  # what the WH Galaxy printed on 2026-09-25
    assert P.missing_host_tool("sudo: %s: command not found" % TOOL) == TOOL
    assert P.missing_host_tool("Error: POST_RESET failed for device 1.") is None
    assert P.missing_host_tool("") is None and P.missing_host_tool(None) is None


def _fake_host(monkeypatch, installed_after=True, have_sudo=True, managers=True):
    from agent import pkgtools

    state = {"present": False, "cmds": []}

    def _which(name):
        if name == TOOL:
            return "/usr/bin/" + name if state["present"] else None
        if name == pkgtools._ELEVATE[0]:
            return "/usr/bin/" + name if have_sudo else None
        if any(name == pm for pm, _ in pkgtools._PACKAGE_MANAGERS):
            return "/usr/bin/" + name if managers else None
        return "/usr/bin/" + name

    def _run(cmd, **k):
        state["cmds"].append(list(cmd))
        state["present"] = installed_after
        return subprocess.CompletedProcess(cmd, 0 if installed_after else 100, "", "")

    monkeypatch.delenv(pkgtools.NO_SYSTEM_INSTALL_ENV, raising=False)
    monkeypatch.delenv(pkgtools.SYSTEM_INSTALL_CMD_ENV, raising=False)
    monkeypatch.setattr(pkgtools, "_SYSTEM_TOOL_TRIED", {})
    monkeypatch.setattr(pkgtools.shutil, "which", _which)
    monkeypatch.setattr(pkgtools.subprocess, "run", _run)
    monkeypatch.setattr(pkgtools.os, "geteuid", lambda: 1000)
    return pkgtools, state


def test_a_missing_tool_is_installed_non_interactively(monkeypatch):
    pkgtools, state = _fake_host(monkeypatch)
    assert pkgtools.ensure_system_tool(TOOL) is True
    pm, args = pkgtools._PACKAGE_MANAGERS[0]
    assert state["cmds"] == [
        ["/usr/bin/" + pkgtools._ELEVATE[0], *pkgtools._ELEVATE[1:], "/usr/bin/" + pm, *args, TOOL]
    ]


def test_the_install_command_is_configurable(monkeypatch):
    pkgtools, state = _fake_host(monkeypatch, managers=False)
    monkeypatch.setenv(pkgtools.SYSTEM_INSTALL_CMD_ENV, "site-installer --yes")
    assert pkgtools.ensure_system_tool(TOOL) is True
    assert state["cmds"][0][-3:] == ["site-installer", "--yes", TOOL]


def test_no_package_manager_means_no_install_attempt(monkeypatch):
    pkgtools, state = _fake_host(monkeypatch, managers=False)
    assert pkgtools.ensure_system_tool(TOOL) is False
    assert state["cmds"] == []


def test_an_install_that_fails_is_reported_and_not_retried(monkeypatch):
    pkgtools, state = _fake_host(monkeypatch, installed_after=False)
    assert pkgtools.ensure_system_tool(TOOL) is False
    assert pkgtools.ensure_system_tool(TOOL) is False
    assert len(state["cmds"]) == 1, "a host where the install cannot work was asked again"


def test_no_sudo_means_no_install_attempt(monkeypatch):
    pkgtools, state = _fake_host(monkeypatch, have_sudo=False)
    assert pkgtools.ensure_system_tool(TOOL) is False
    assert state["cmds"] == []


def test_the_opt_out_is_honoured(monkeypatch):
    pkgtools, state = _fake_host(monkeypatch)
    monkeypatch.setenv(pkgtools.NO_SYSTEM_INSTALL_ENV, "1")
    assert pkgtools.ensure_system_tool(TOOL) is False
    assert state["cmds"] == []


def _reset_runs(monkeypatch, first_output, installed=True):
    from agent import pkgtools

    calls = []

    def _run(cmd, **k):
        calls.append(list(cmd))
        failed = len(calls) == 1
        return subprocess.CompletedProcess(cmd, 1 if failed else 0, first_output if failed else "done", "")

    monkeypatch.setattr(P.subprocess, "run", _run)
    monkeypatch.setattr(P.shutil, "which", lambda name: None)
    monkeypatch.setattr(pkgtools, "ensure_system_tool", lambda name: installed)
    return calls


def test_a_reset_that_reports_a_missing_tool_installs_it_and_retries(monkeypatch):
    calls = _reset_runs(monkeypatch, "sudo: %s: command not found" % TOOL)
    assert P.run_reset_command("/bin/tt-smi", ["-glx_reset"], 5).returncode == 0
    assert calls == [["/bin/tt-smi", "-glx_reset"]] * 2


def test_a_reset_that_fails_for_another_reason_is_not_retried(monkeypatch):
    calls = _reset_runs(monkeypatch, "Error: POST_RESET failed for device 1.")
    assert P.run_reset_command("/bin/tt-smi", ["-glx_reset"], 5).returncode == 1
    assert len(calls) == 1


def test_a_tool_that_cannot_be_installed_returns_the_original_failure(monkeypatch):
    calls = _reset_runs(monkeypatch, "sudo: %s: command not found" % TOOL, installed=False)
    proc = P.run_reset_command("/bin/tt-smi", ["-glx_reset"], 5)
    assert proc.returncode == 1 and TOOL in proc.stdout
    assert len(calls) == 1
