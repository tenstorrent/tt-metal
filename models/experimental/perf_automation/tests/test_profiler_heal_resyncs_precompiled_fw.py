# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A relinked libtt_metal.so must never be left with yesterday's pre-compiled firmware.

profiler_heal rebuilds libtt_metal.so by naming ninja targets, so the `precompile-fw` ALL-target
that regenerates tt_metal/pre-compiled/<build_key>/ did not re-run. The runtime prefers a matching
pre-compiled dir over a JIT build, so on 2026-09-22 -- after an upstream sync changed the
Blackhole L1 routing-table layout -- every plain device open ran old-layout firmware under a
new-layout runtime: all Tensix cores timed out in FW init ("failed to initialize FW! Try
resetting the board"), NOC0 hung, the ARC dropped off the bus. Resets and a host reboot could
not help, and the failure was misdiagnosed as a board wedge for a day.

The heal now regenerates the pre-compiled firmware with the same build tree after relinking, and
if it cannot, removes the stale directory so the runtime JIT-builds coherent firmware instead.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _heal():
    from agent import profiler_heal

    return profiler_heal


class _Done:
    def __init__(self, rc=0):
        self.returncode = rc
        self.stdout = ""
        self.stderr = ""


def _fake_run(calls, rc_for):
    def run(argv, **kw):
        calls.append(list(argv))
        return _Done(rc_for(argv))

    return run


def _root_with_stale_precompiled(tmp_path):
    root = tmp_path
    pre = root / "tt_metal" / "pre-compiled" / "123456"
    pre.mkdir(parents=True)
    (pre / "brisc.elf").write_bytes(b"old-layout-firmware")
    build = root / "build_Release"
    build.mkdir()
    (build / "build.ninja").write_text("")
    (build / "tt_metal").mkdir()
    (build / "tt_metal" / "libtt_metal.so").write_bytes(b"new-lib")
    (build / "ttnn").mkdir()
    (build / "ttnn" / "_ttnn.so").write_bytes(b"new-ttnn")
    return root, build


def test_resync_regenerates_with_the_same_build_tree(tmp_path, monkeypatch):
    h = _heal()
    root, build = _root_with_stale_precompiled(tmp_path)
    calls = []
    monkeypatch.setattr(h.subprocess, "run", _fake_run(calls, lambda a: 0))
    monkeypatch.setattr(h, "_rebuild_timeout", lambda: 5)
    assert h._resync_precompiled_fw(root, build) == "rebuilt"
    assert ["ninja", "-C", str(build), "precompile-fw"] in calls
    # Regenerated in place -- the directory the runtime looks in still exists.
    assert (root / "tt_metal" / "pre-compiled").is_dir()


def test_resync_removes_the_stale_dir_when_regeneration_fails(tmp_path, monkeypatch):
    """The fallback that guarantees coherence: no pre-compiled dir => the runtime JIT-builds
    firmware from the same sources the library was built from."""
    h = _heal()
    root, build = _root_with_stale_precompiled(tmp_path)
    monkeypatch.setattr(h.subprocess, "run", _fake_run([], lambda a: 1))
    monkeypatch.setattr(h, "_rebuild_timeout", lambda: 5)
    assert h._resync_precompiled_fw(root, build) == "wiped"
    assert not (root / "tt_metal" / "pre-compiled").exists()


def test_resync_survives_a_ninja_that_cannot_even_start(tmp_path, monkeypatch):
    h = _heal()
    root, build = _root_with_stale_precompiled(tmp_path)

    def boom(argv, **kw):
        raise FileNotFoundError("ninja")

    monkeypatch.setattr(h.subprocess, "run", boom)
    monkeypatch.setattr(h, "_rebuild_timeout", lambda: 5)
    assert h._resync_precompiled_fw(root, build) == "wiped"
    assert not (root / "tt_metal" / "pre-compiled").exists()


def test_rebuild_resyncs_after_relinking(tmp_path, monkeypatch):
    """The exact shape of the incident: _rebuild relinks the libraries and returns True. It must
    now also drive precompile-fw, AFTER the libraries, in the same build tree."""
    h = _heal()
    root, build = _root_with_stale_precompiled(tmp_path)
    calls = []
    monkeypatch.setattr(h.subprocess, "run", _fake_run(calls, lambda a: 0))
    monkeypatch.setattr(h, "_rebuild_timeout", lambda: 5)
    assert h._rebuild(root, build) is True
    ninja_calls = [c for c in calls if c[:1] == ["ninja"]]
    assert ninja_calls[0][:3] == ["ninja", "-C", str(build)]
    assert "tt_metal/libtt_metal.so" in ninja_calls[0]
    assert ninja_calls[-1] == ["ninja", "-C", str(build), "precompile-fw"]


def test_fw_init_failure_is_a_device_disruption():
    """The first FW-init failure on 2026-09-22 got no reset because the disruption list did not
    know the message. A reset is the precondition for ANY retry after this signature (NOC0 stays
    hung otherwise), so all three spellings the runtime emits must match."""
    from agent import perf_test_gen as g

    for msg in (
        "TT_THROW: Device 0 init: failed to initialize FW! Try resetting the board.",
        "Device 0: Timeout (10000 ms) waiting for physical cores to finish: 13-6, 11-7.",
        "RuntimeError: NOC0 is hung on PCIe device ID 0.",
    ):
        assert g._DEVICE_DISRUPTION_RE.search(msg), msg
    assert not g._DEVICE_DISRUPTION_RE.search("PCC 0.9993 >= 0.95 PASS")


def _coherent_tree(tmp_path):
    """A checkout whose pre-compiled firmware is NEWER than its library (the normal state)."""
    import os
    import time

    root = tmp_path
    lib = root / "build_Release" / "tt_metal" / "libtt_metal.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"lib")
    elf = root / "tt_metal" / "pre-compiled" / "42" / "brisc" / "brisc.elf"
    elf.parent.mkdir(parents=True)
    elf.write_bytes(b"fw")
    now = time.time()
    os.utime(lib, (now - 100, now - 100))
    os.utime(elf, (now - 50, now - 50))
    return root, lib, elf


def test_stale_precompiled_firmware_is_named_not_blamed_on_the_board(tmp_path):
    """Library relinked AFTER the firmware was pre-compiled => stale, and the verdict must be the
    actionable sentence (rebuild precompile-fw), not 'reboot the host'."""
    import os
    import time

    from agent import device_recovery as dr

    root, lib, elf = _coherent_tree(tmp_path)
    assert dr.stale_precompiled_firmware(root) is None
    now = time.time()
    os.utime(lib, (now, now))  # profiler_heal relinks the library; firmware untouched
    why = dr.stale_precompiled_firmware(root)
    assert why and "precompile-fw" in why and "not a board fault" in why
    assert str(lib) in why


def test_stale_check_is_silent_without_a_precompiled_tree_or_library(tmp_path):
    """No pre-compiled dir (wheel install, or already wiped) and no build*/ library are both
    'nothing to compare' -- never a false stale verdict on a healthy box."""
    from agent import device_recovery as dr

    assert dr.stale_precompiled_firmware(tmp_path) is None
    (tmp_path / "tt_metal" / "pre-compiled").mkdir(parents=True)
    assert dr.stale_precompiled_firmware(tmp_path) is None
    assert dr.stale_precompiled_firmware("/nonexistent/path") is None


def test_optimize_halt_asks_the_software_question_before_the_hardware_verdict():
    """Ordering asserted on the source, like test_a_held_board_is_not_a_dead_board: the stale
    pre-compiled-firmware check must be consulted before 'reboot the host' can be the verdict."""
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    i_stale = src.index("stale_precompiled_firmware")
    i_halt = src.index('"needs_host_reboot"')
    assert i_stale < i_halt, "the reboot verdict is reached before the stale-firmware check"
    assert '"stale_precompiled_fw"' in src


def test_disruption_message_names_stale_firmware_only_for_fw_init_failures(monkeypatch):
    from agent import device_recovery as dr
    from agent import perf_test_gen as g

    monkeypatch.setattr(dr, "stale_precompiled_firmware", lambda root: "STALE: rebuild precompile-fw")
    assert g._stale_fw_hint("Device 0 init: failed to initialize FW! Try resetting the board.").startswith("STALE")
    assert g._stale_fw_hint("RuntimeError: NOC0 is hung on PCIe device ID 0.").startswith("STALE")
    # A clock disruption is a real board event; the firmware hint would be noise there.
    assert g._stale_fw_hint("AICLK failed to settle") == ""
    monkeypatch.setattr(dr, "stale_precompiled_firmware", lambda root: None)
    assert g._stale_fw_hint("failed to initialize FW") == ""


def test_every_halt_kind_the_gate_can_emit_has_its_own_remedy():
    """run._HALT_REMEDY falls back to the tt-lang remedy for an unknown kind -- the exact
    mis-remedy its own comment warns about. So every `"halt": "<kind>"` literal in perf_mcp
    must have an entry, including the new stale_precompiled_fw."""
    import re

    from cc_optimize import run

    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    kinds = set()
    for a, b in re.findall(r'"halt":\s*"([a-z_]+)"(?:\s+if [^\n]*? else "([a-z_]+)")?', src):
        kinds.update(k for k in (a, b) if k)
    assert "stale_precompiled_fw" in kinds
    for kind in kinds:
        assert kind in run._HALT_REMEDY, f"halt kind {kind!r} has no operator remedy"
    assert "precompile-fw" in run._HALT_REMEDY["stale_precompiled_fw"]
