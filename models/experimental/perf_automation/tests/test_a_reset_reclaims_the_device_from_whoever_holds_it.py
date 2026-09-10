# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every reset reclaims the device first, at every recovery point -- not at three of eight.

WHAT THIS COST, measured on Voxtral, 2026-08-11.

A perf-test build blocked inside `ttnn.from_torch` at 18:12:51 and never returned. At 19:37 it was
still there: 85 minutes elapsed, 91 minutes of CPU across 65 threads, 8 open /dev/tenstorrent fds,
and not one line of log output after that first second.

Three perf-test regenerations ran in that window. Each detected a wedge, reset the chip, and
started again -- into a device the orphan still held. All three reported "device wedged on a
non-capturable step". SIGTERMing the orphan by hand let the same run walk straight through Step 7,
8 and 9 without another wedge.

The reset was never the problem. A reset clears the CHIP; the process mid-transfer keeps its
handles, now pointing at state that no longer exists, and it can neither progress nor be reset out
of the way. It has to be killed, and the only moment anyone knows to kill it is when a wedge is
being recovered.

WHY IT WAS MISSED. The reap existed -- run.py's `_reclaim_device`, whose own docstring calls it
"the UNIVERSAL device reclaim used at EVERY recovery point". It was wired into three of the eight
sites that reset. The other five (perf_test_gen x2, perf_mcp x2, probes) reset without it, and
those are precisely the paths a wedged perf-test build takes. A policy enforced at three of eight
call sites is not a policy; it is a coincidence.

So it moved into `recover()`, the one primitive every reset routes through, and the callers stopped
deciding. These tests pin that: the reap is in the primitive, it runs BEFORE the reset, no caller
carries a second copy, and it can never kill the process doing the recovering.
"""

import os
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent


def _dr():
    from models.experimental.perf_automation.agent import device_recovery

    return device_recovery


def test_the_reap_lives_in_the_one_primitive_every_reset_routes_through():
    assert callable(_dr().reap_device_holders), "the shared primitive has no reap"


def test_the_reap_runs_before_the_reset_not_after():
    """After the reset it is useless: the chip is already clean and the holder still owns handles to
    a device that no longer matches them."""
    src = (_PA / "agent" / "device_recovery.py").read_text()
    i = src.index("def recover(")
    # THE WHOLE FUNCTION, not a character window. A 4000-char slice made this fail whenever anything
    # was added above `targets_for(` -- the wedge gate did exactly that -- failing a test whose
    # subject had not moved. A window measured in characters is a guess about layout.
    _next = src.find("\ndef ", i + 1)
    body = src[i : _next if _next > 0 else len(src)]
    assert "reap_device_holders()" in body, "recover() does not reclaim"
    assert body.index("reap_device_holders()") < body.index("targets_for("), "the reap runs after the reset"


def test_the_caller_that_used_to_own_this_no_longer_carries_a_second_copy():
    """Two copies drift. run.py's version is what enforced the policy at three sites while five went
    without; it now delegates, so there is one implementation to be right or wrong."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _reclaim_device(")
    body = src[i : i + 2500]
    assert "reap_device_holders()" in body, "run.py does not use the shared reap"
    assert 'subprocess.run(["fuser"' not in body, "run.py still scans for holders itself"


def test_it_can_never_kill_the_process_doing_the_recovering():
    """Killing an ancestor kills the orchestrator or supervisor that would recover. A self-hold is
    handled by exiting to the supervisor, never by the holder shooting itself."""
    protected = _dr()._protected_pids()
    assert os.getpid() in protected, "the reaping process is not protected from itself"
    assert os.getppid() in protected, "the parent is not protected"
    assert 1 not in protected, "init is walked into"


def test_a_host_without_fuser_still_recovers():
    """This runs when the board is already in trouble. A reclaim that raises would turn a
    recoverable wedge into a dead run, so every step degrades to 'reaped fewer than there were'."""
    src = (_PA / "agent" / "device_recovery.py").read_text()
    i = src.index("def reap_device_holders(")
    body = src[i : i + 2000]
    assert body.count("except Exception") >= 2, "the scan or the kill can raise into the caller"
    # And it must actually run here, on a host that has no device at all.
    assert isinstance(_dr().reap_device_holders(), list)
