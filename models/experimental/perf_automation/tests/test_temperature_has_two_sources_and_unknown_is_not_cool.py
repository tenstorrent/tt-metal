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


# ------------------------------------------------- a successful read is not a reading (2026-08-16)
#
# The temperature comes from each chip's ARC management core. When the ARC is not running the driver
# still publishes temp1_input, filled with all ones:
#
#     76617      -> 76.6C        a temperature
#     65535999   -> 65535.999C   "no data", in a temperature's units
#
# Same file, same format, same successful read: no error to catch, no validity flag to consult. So
# the VALUE is the only thing that separates them, and nothing was checking it. On 2026-08-16 two
# chips lost their ARC (a leaked process tree was holding the board) and _read_asic_temp -- which
# takes the HOTTEST chip -- returned 65535.999 for a board whose live chips sat at 80C. Every thermal
# gate then waited its full 900s against a number that can never fall, and measured hot anyway.
#
# The second defect was in the combiner. Its docstring promised "TWO SOURCES, AND THE HOTTER WINS";
# the code asked tt-smi only when sysfs found NO chips, so `max` ran across sysfs's four chips and
# never between the sources. A fallback is the wrong shape here for a structural reason: it trusts
# the first source to know when it has failed, and a sentinel is precisely a failure that reports
# success. The one moment the second opinion is needed is the one moment it was never asked for.


def test_a_sentinel_chip_does_not_speak_for_the_board(monkeypatch):
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [65535.999, 65535.999, 76.6, 80.5])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: None)
    assert probes._read_asic_temp() == 80.5


def test_a_zero_reading_is_refused_too(monkeypatch):
    """The other sentinel shape. An all-zeros register reads as a plausible 0C and would drag a
    max-of-chips DOWN -- a board reported cool when it is not, which is the dangerous direction."""
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [0.0])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: 72.0)
    assert probes._read_asic_temp() == 72.0


def test_no_usable_reading_anywhere_is_unknown_not_cool(monkeypatch):
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [65535.999])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: 65535.999)
    assert probes._read_asic_temp() is None


def test_a_genuinely_hot_board_is_still_reported_hot(monkeypatch):
    """The bound must not swallow the readings the gate exists for. 95C is real and must survive."""
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [95.0])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: None)
    assert probes._read_asic_temp() == 95.0


def test_the_hotter_SOURCE_wins_not_merely_the_hotter_chip(monkeypatch):
    """THE BUG THE OLD TEST COULD NOT SEE. It asserted max(sysfs 71, tt-smi 65) and passed against
    code that ignored tt-smi entirely, because its sysfs value was the higher one. Reverse them."""
    from agent import probes

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [60.0])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", lambda: 90.0)
    assert probes._read_asic_temp() == 90.0


def test_tt_smi_is_asked_even_when_sysfs_answered(monkeypatch):
    """Not a fallback. A sentinel is a failure that reports success, so 'sysfs answered' is no reason
    to stop asking -- that is exactly the case where the second opinion decides."""
    from agent import probes

    asked = {"n": 0}

    def _smi():
        asked["n"] += 1
        return 70.0

    monkeypatch.setattr(probes, "_sysfs_asic_temps", lambda: [65.0])
    monkeypatch.setattr(probes, "_tt_smi_asic_temp", _smi)
    probes._read_asic_temp()
    assert asked["n"] == 1, "tt-smi is still only a fallback"


def test_the_sentinel_is_filtered_where_it_is_parsed(tmp_path, monkeypatch):
    """At the source, so every caller of _sysfs_asic_temps is covered, not just _read_asic_temp."""
    from agent import probes

    root = tmp_path / "hwmon"
    drv = tmp_path / "tenstorrent"  # the link must RESOLVE to a dir of this name
    drv.mkdir()
    for name, temp in (("hwmon0", "65535999"), ("hwmon1", "76617")):
        h = root / name
        (h / "device").mkdir(parents=True)
        (h / "device" / "driver").symlink_to(drv)
        (h / "temp1_input").write_text(temp)
    monkeypatch.setattr(probes, "_SYSFS_HWMON", str(root))
    assert probes._sysfs_asic_temps() == [76.617]


def test_a_chip_with_no_telemetry_is_recorded_not_just_dropped(tmp_path, monkeypatch):
    """A dead ARC is a BROKEN CHIP, not a temperature question. On 2026-08-16 it meant a leaked
    process tree was holding the board -- a fault worth acting on, and one a silent filter hides."""
    from agent import probes

    root = tmp_path / "hwmon"
    h = root / "hwmon3"
    (h / "device").mkdir(parents=True)
    drv = tmp_path / "tenstorrent"
    drv.mkdir(exist_ok=True)
    (h / "device" / "driver").symlink_to(drv)
    (h / "temp1_input").write_text("65535999")
    monkeypatch.setattr(probes, "_SYSFS_HWMON", str(root))
    probes._NO_TELEMETRY_CHIPS.clear()
    probes._sysfs_asic_temps()
    assert probes.chips_without_telemetry() == ["hwmon3"]

    (h / "temp1_input").write_text("71000")
    probes._sysfs_asic_temps()
    assert probes.chips_without_telemetry() == [], "a chip that recovered is still called broken"
