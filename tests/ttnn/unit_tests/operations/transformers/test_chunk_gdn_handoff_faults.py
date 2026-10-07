# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fault injection for the hand-off protocol's run-time checks (device/chunk_gdn_handoff_protocol.md, "Runtime
checks"): each case compiles one deliberate breach into the fused kernels
(ChunkGdnFusedProgramConfig(handoff_checks=True, handoff_fault=n)) and proves that the named check trips under the
watcher.

A tripped watcher assert stops the device and ends the process through the watcher's TT_THROW, so every fault runs in
a child pytest process and the board is reset between faults. Opt in with GDN_HANDOFF_FAULT_TESTS=1; the reset command
comes from GDN_HANDOFF_FAULT_RESET_CMD (default "tt-smi -r"). Nothing here runs in the regular suites."""

import os
import re
import shlex
import subprocess
import sys

import pytest

from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="chunk_gated_delta_rule is Blackhole-only")

_OPT_IN = os.environ.get("GDN_HANDOFF_FAULT_TESTS") == "1"
_CHILD_FAULT = os.environ.get("GDN_HANDOFF_FAULT_CHILD")  # set by the parent test: the fault id to compile in

# fault id (gdn_handoff::HandoffFault) -> name, RISCs that may assert, waypoints they may stop at, C9 timeout expected
FAULTS = {
    # C1 at the owner's credit poll or C2 at its teardown; or, when one receiver's doubled credit alone satisfies the
    # owner, it sends before the other receiver has reset its slot: C4 in that receiver's issue, or C9 after the reset
    # erased the early flag (the lost wakeup the exact-NV rule exists for).
    1: ("double_credit", {"BRISC", "NCRISC"}, {"TXCR", "DONE", "RXRS", "RXVL"}, False),
    2: ("wrong_canary", {"NCRISC"}, {"RXVL"}, False),  # C8 after the flag
    3: ("short_push", {"NCRISC"}, {"RXVL"}, False),  # C3 at the next push into the slot
    4: ("wrong_owner", {"BRISC", "NCRISC"}, {"TXCR", "RXVL"}, True),  # C9 on whichever side expires first
    5: ("no_credit", {"BRISC", "NCRISC"}, {"TXCR", "RXVL"}, True),  # C9
}
_RISC_ORDER = ["BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"]  # field order of the watcher's "Last waypoint" line
_TIMEOUT_WORD = re.compile(r"0x[ef][0-9a-f]{7}")  # ring-buffer stages 14 / 15: the bounded waits expired


@pytest.mark.skipif(_CHILD_FAULT is None, reason="child of test_handoff_fault_trips_check")
def test_handoff_fault_child(device):
    from tests.ttnn.unit_tests.operations.transformers.test_chunk_gdn_fused import _fused_vs_phased

    # bh12-nv2np7-nc8 at depth 2: NP >= 2 for the wrong-owner fault, NC > 3 for the never-credited chunk.
    _fused_vs_phased(
        device, 4, 12, 8, 2, 7, 20261005, handoff_depth=2, handoff_checks=True, handoff_fault=int(_CHILD_FAULT)
    )
    pytest.fail(f"handoff_fault={_CHILD_FAULT} ran to completion: the check it targets did not trip")


@pytest.mark.skipif(
    not _OPT_IN, reason="opt in with GDN_HANDOFF_FAULT_TESTS=1 (trips watcher asserts, resets the board)"
)
@pytest.mark.parametrize("fault", list(FAULTS), ids=lambda f: FAULTS[f][0])
def test_handoff_fault_trips_check(fault):
    name, riscs, waypoints, expect_timeout = FAULTS[fault]
    env = dict(
        os.environ,
        GDN_HANDOFF_FAULT_CHILD=str(fault),
        TT_METAL_WATCHER="1",
        TT_METAL_WATCHER_NOINLINE="1",
        TT_METAL_WATCHER_DISABLE_ETH="1",
    )
    env.pop("GDN_HANDOFF_FAULT_TESTS", None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__, "-k", "test_handoff_fault_child"]
    child = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    out = child.stdout + child.stderr
    if child.returncode != 0:
        # The watcher stopped the device: reset the board before anything else opens it.
        reset = shlex.split(os.environ.get("GDN_HANDOFF_FAULT_RESET_CMD", "tt-smi -r"))
        subprocess.run(reset, check=True, timeout=600, capture_output=True)
    # The watcher's report lines, for the failure messages (the raw tail is the abort's backtrace).
    tail = "\n".join(l for l in out.splitlines() if re.search(r"tripped an assert|Last waypoint|RISC\]0x", l))
    tail = tail or out[-2000:]
    assert child.returncode != 0, f"{name}: the child passed, no check tripped\n{out[-2000:]}"
    m = re.search(r"(BRISC|NCRISC|TRISC\d) tripped an assert", out)
    assert m, f"{name}: no tripped assert in the child's output\n{tail}"
    risc = m.group(1)
    assert risc in riscs, f"{name}: {risc} asserted, expected one of {sorted(riscs)}\n{tail}"
    wp = re.search(r"Last waypoint: ([^\n]*)", out)
    assert wp, f"{name}: no waypoint dump\n{tail}"
    stage = [f.strip() for f in wp.group(1).split(",")][_RISC_ORDER.index(risc)]
    assert stage in waypoints, f"{name}: {risc} stopped at {stage}, expected one of {sorted(waypoints)}\n{tail}"
    if expect_timeout:
        assert _TIMEOUT_WORD.search(out), f"{name}: no C9 timeout word in the ring-buffer dump\n{tail}"
