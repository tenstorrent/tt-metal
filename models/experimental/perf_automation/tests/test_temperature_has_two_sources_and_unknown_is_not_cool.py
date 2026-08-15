# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The kernel already publishes the die temperature, and a failed read is not a cool board.

TWO DEFECTS, ONE RUN. On 2026-08-15 a measurement started on a 93C board. It ran clamped at 800 MHz
instead of 1350, took roughly 1.7x longer, blew its 1806 s budget, and the timeout's reset bricked
the board. The gate that should have stopped it did run -- and waved it through.

    if cur is None or limit is None or cur <= limit:
        return True, cur          # `cur is None` == "could not read" == treated as cool

FIRST: THE READ. _read_asic_temp shelled out to `tt-smi -s`, which OPENS THE DEVICE to answer a
question the tenstorrent driver exposes as a plain file. Measured on this host:

    sysfs  0.0003 s   62.9C        (per-chip: 59.6 59.9 61.6 61.9)
    tt-smi 0.2695 s   63.3C

and under a live matmul, sysfs still answered in 0.3 ms (81.9C against tt-smi's 82.1C). Same number,
~900x cheaper, no subprocess, no device open, and per-chip values instead of one aggregate.

Both sources can be unavailable and they fail INDEPENDENTLY -- sysfs needs a driver that registers
hwmon (this host runs tenstorrent v2.8.0), tt-smi needs to not be contending -- so both are asked and
the HOTTER wins. A source saying hot is evidence; a source saying nothing is not evidence of cool.

Matched on the DRIVER, not the hwmon `name`: the name is the arch ("blackhole"), so keying on it
would find zero chips on Wormhole and silently report no temperature at all.

SECOND: THE FALLBACK. Unknown must not mean cool -- and the moment a read is most likely to fail is
right after a heavy run, exactly when the board is hottest. A host that has NEVER produced a reading
has no telemetry to wait for and still proceeds, unchanged. Having read 93C a minute ago and being
unable to read now is a different thing, and that last value is the best evidence there is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PA = Path(__file__).resolve().parent.parent
_PROBES = (_PA / "agent" / "probes.py").read_text()


# ---------------------------------------------------------------------- two sources


def test_sysfs_is_tried_and_matched_on_the_driver_not_the_arch_name():
    i = _PROBES.index("def _sysfs_asic_temps(")
    body = _PROBES[i : _PROBES.index("\ndef ", i + 1)]
    # Code only: the docstring names the arch to explain why it is NOT the key.
    code = body.split('"""', 2)[-1]
    assert "temp1_input" in code
    assert '"tenstorrent"' in code, "matched on something other than the driver"
    assert "blackhole" not in code, "keying on the arch name would find nothing on Wormhole"


def test_the_hotter_of_the_two_sources_wins(monkeypatch):
    """A source that says hot is evidence; one that says nothing is not evidence of cool."""
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [60.0, 71.0])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: 65.0)
    assert probes._read_asic_temp() == 71.0


def test_tt_smi_answers_when_sysfs_finds_no_chips(monkeypatch):
    """An older driver registers no hwmon; the second source is the whole point."""
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: 77.0)
    assert probes._read_asic_temp() == 77.0


def test_none_only_when_neither_source_answers(monkeypatch):
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: None)
    assert probes._read_asic_temp() is None


def test_the_tt_smi_read_is_bounded():
    """Its failure mode is to HANG, not to answer, so it cannot be consulted unbounded."""
    i = _PROBES.index("def _tt_smi_asic_temp(")
    body = _PROBES[i : _PROBES.index("\ndef ", i + 1)]
    assert "timeout=_TT_SMI_TEMP_TIMEOUT_S" in body


# ---------------------------------------------------------------------- unknown is not cool


def _mcp(monkeypatch, tmp_path):
    from cc_optimize import perf_mcp

    box = tmp_path / "box"
    box.mkdir(exist_ok=True)
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_BOARD_STATE_DIR", str(box))
    monkeypatch.setattr(perf_mcp, "_LAST_KNOWN_TEMP_C", None)
    monkeypatch.setattr(perf_mcp.time, "sleep", lambda _s: None)
    return perf_mcp


def test_a_failed_read_falls_back_to_the_last_known_temperature(monkeypatch, tmp_path):
    """THE RUN-20 CASE: read 93C, then the sensor goes quiet. That is not a cool board."""
    mcp = _mcp(monkeypatch, tmp_path)
    monkeypatch.setattr(mcp, "_clamp_threshold_c", lambda: 68.1)
    monkeypatch.setattr(mcp, "_LAST_KNOWN_TEMP_C", 93.0)
    seen = iter([None])
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: next(seen, 60.0))
    monkeypatch.setattr(mcp, "_THERMAL_POLL_S", 0.0)
    ok, temp = mcp._wait_for_thermal_headroom()
    assert ok and temp is not None and temp <= 68.1, (ok, temp)


def test_a_host_that_never_read_anything_still_proceeds(monkeypatch, tmp_path):
    """Unchanged, and deliberate: a missing sensor is not a hot board."""
    mcp = _mcp(monkeypatch, tmp_path)
    monkeypatch.setattr(mcp, "_clamp_threshold_c", lambda: 68.1)
    monkeypatch.setattr(mcp, "_read_die_temp_c", lambda: None)
    ok, temp = mcp._wait_for_thermal_headroom()
    assert ok is True and temp is None


def test_a_successful_read_is_remembered(monkeypatch, tmp_path):
    """The fallback is only as good as the memo behind it."""
    from agent import probes

    mcp = _mcp(monkeypatch, tmp_path)
    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [88.0])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: None)
    assert mcp._read_die_temp_c() == 88.0
    assert mcp._LAST_KNOWN_TEMP_C == 88.0


def test_the_gate_no_longer_lumps_unknown_in_with_cool():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert "if cur is None or limit is None or cur <= limit:" not in src, "unknown is treated as cool again"
