# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""After a kill the reset covers every board, because a killed process held every device.

`--devices 0` says which chip the run should USE. It does not say which chips the process MAPPED: a
ttnn process opens every enumerated device, and a single probe was observed holding
/dev/tenstorrent/0, 1, 2 and 3 on a --devices 0 run. SIGKILL can therefore leave any of them
half-initialised.

MEASURED 2026-08-14. Run 10's full-pipeline measurement was killed at its budget:

    KILLED after 1686s (hard limit) ... reclaimed device (killed holders none) + tt-smi -r 0,1

The reset went to the --devices-derived board, 0,1. The health check passed for that board, so the
escalation ladder (error-named chip -> configured board -> all) never reached `all`. Device 2 was
never touched and stayed wedged:

    tenstorrent tenstorrent!2: Failed to set initial power state: -22   (repeating)

The next run died at Step 1 with `tt-smi -s` timing out after 120 s, and by then `tt-smi -r` hung on
it too -- only a host reboot cleared the board.

After a kill the scope is UNKNOWN, and expand_spec's own rule applies: an unverifiable target must
widen, never narrow. A reset covering too much is recoverable; one covering too little leaves a chip
nobody touches.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("_run_reclaim", _PA / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture(run, monkeypatch):
    """Record the config_target the reclaim hands to the recovery primitive."""
    seen = {}

    class _DR:
        @staticmethod
        def reap_device_holders():
            return ""

        @staticmethod
        def recover(where, reset, error_text="", config_target="", log=None, expand=None):
            seen["target"] = config_target
            return True

    monkeypatch.setattr(run, "_dr", lambda: _DR, raising=False)
    monkeypatch.setattr(run, "_reset_devices", lambda t: "tt-smi -r %s" % t, raising=False)
    return seen


def test_a_kill_resets_every_board(monkeypatch):
    """THE BUG: scoped to the configured board, which is not what the dead process held."""
    run = _run()
    seen = _capture(run, monkeypatch)
    run._reclaim_device("0", error_text="timed out", after_kill=True)
    assert seen["target"] == "all", seen


def test_a_non_kill_still_uses_the_configured_board(monkeypatch):
    """A crash that NAMES a chip has evidence, and the ladder starts from it -- widening every reset
    unconditionally would reset healthy boards on every routine recovery."""
    run = _run()
    seen = _capture(run, monkeypatch)
    run._reclaim_device("0", error_text="Read 0xffffffff over PCIe ID 3")
    assert seen["target"] == "0", seen


def test_the_flag_is_not_shadowed_by_the_holder_list():
    """`killed` was already a local in that function -- reap_device_holders()'s result. A parameter of
    the same name is overwritten before it is read, and silently, because the list is empty after a
    SIGKILL and an empty string is falsy. The first version of this fix did exactly that and reset
    the configured board anyway."""
    import inspect

    run = _run()
    src = inspect.getsource(run._reclaim_device)
    assert "after_kill" in src
    assert "killed: bool" not in src, "the flag shares a name with the holder-list local"


def test_the_timeout_path_reports_the_kill():
    """The reclaim cannot widen if the caller does not tell it a kill happened."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("_reclaim_device(devices, error_text=out")
    assert "after_kill=True" in src[i : i + 130], "the timeout path does not mark the reclaim as a kill"
