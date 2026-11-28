# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tool's own tests can never find, or SIGKILL, a real process using the board.

recover() reaps every holder of the device nodes before it resets. Several test files run it for real
and only some stubbed the reaper, so running this suite on a box where a model was on the device
killed that run: 2026-11-09, a live GLM-5.3-Flash coverage probe died mid-run with rc -9 and left the
Galaxy hung (NOC0). The device directory is now read at call time from TT_RECOVERY_DEVICE_DIR, and the
suite-wide fixture points it at an empty directory.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_every_test_sees_an_empty_device_directory():
    import agent.device_recovery as dr

    d = Path(dr.device_node_dir())
    assert d != Path(dr.DEVICE_NODE_DIR), "a test is looking at the real device nodes"
    assert not d.exists() or not any(d.iterdir())
    assert dr.device_holders() == set()


def test_the_override_is_read_at_call_time_and_survives_a_reload(monkeypatch, tmp_path):
    import agent.device_recovery as dr

    monkeypatch.setenv("TT_RECOVERY_DEVICE_DIR", str(tmp_path / "a"))
    assert dr.device_node_dir() == str(tmp_path / "a")
    importlib.reload(dr)
    assert dr.device_node_dir() == str(tmp_path / "a")
    monkeypatch.delenv("TT_RECOVERY_DEVICE_DIR")
    assert dr.device_node_dir() == dr.DEVICE_NODE_DIR, "unset, it is the real directory as before"


def test_recover_reaps_nothing_real_from_a_test():
    import agent.device_recovery as dr

    assert dr.reap_device_holders() == []
