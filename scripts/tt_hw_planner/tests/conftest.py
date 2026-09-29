# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep this suite's temp files out of the shared system temp directory.

WHY THIS EXISTS. Several tests drive the real gate, and the gate creates its log directory with
`tempfile.mkdtemp(prefix="e2e_gate_")` -- so running the suite scatters `e2e_gate_*` directories
through the system temp dir, holding FIXTURE text: "RuntimeError: boom", "NOC0 is hung on PCIe
device ID 9", "[e2e] denoise step 37/50 done".

That is the same prefix a LIVE bring-up uses, and a live run's agent inspects those directories to
work out why its own gate failed. On 2026-09-28 it read this suite's fixtures twice and concluded
its gate had been SIGKILLed mid-run -- once at "denoise step 37/50", once at 43/50 -- and reported
hours of device time lost that had not been lost. Test data that is indistinguishable from
production evidence does not just litter; it gets believed.

`tempfile.tempdir` is redirected rather than the callers changed, so this holds for every test in
the suite including ones written later, and for any tool code they call. `gettempdir()` reads the
same global, so a test that globs the temp dir for its own leaks still sees exactly what it made.

AND THE SUITE MUST NOT BE ABLE TO REACH THE BOARD AT ALL.

Several of these tests drive the real gate, and the gate's failure paths end in a device reset that
begins by SIGKILLing every process holding /dev/tenstorrent. A test that reaches it unpatched does
not fail -- it succeeds, having killed whatever real work was on the board. That happened twice:
on 2026-09-27 and again on 2026-09-29 at 20:36:05, when this suite killed a live trace capture
eight and a half minutes into its run, and the gate reported the capture as a model defect. Both
times the audit that missed it looked for DIRECT calls to the reaper; both times the route in was a
production function that reaps several frames down (_retry_after_wedge -> _device_reset -> recover
-> reap_device_holders), which no scan of test bodies can see.

So the primitives that TOUCH the device are replaced for the whole suite, and a test that reaches
one raises instead of proceeding. Two deliberate choices: the block is on the primitives
(`device_holders`, the reset) and NOT on the orchestration above them, so a test that patches
`device_holders` itself and exercises `reap_device_holders` for real still works exactly as
written; and it RAISES rather than returning something harmless, because a reaper that silently
finds nothing would hide the very wiring this exists to surface.
"""

from __future__ import annotations

import tempfile

import pytest

_DEVICE_PRIMITIVES = (
    # (module path, attribute) -- the lowest points at which the tool touches real hardware.
    ("models.experimental.perf_automation.agent.device_recovery", "device_holders"),
    ("models.experimental.perf_automation.agent.device_recovery", "recover"),
    ("models.experimental.perf_automation.agent.probes", "_device_reset"),
    ("scripts.tt_hw_planner.trace_gate", "reclaim_mesh"),
    # Not a kill, but the same class of accident: this is what LAUNCHES a run on the board. A test
    # that drives the capture without patching above it would start a real device pytest beside a
    # live one, and contention is how the kills started.
    ("models.experimental.perf_automation.agent.perf_test_gen", "_run_perf_node"),
)


@pytest.fixture(autouse=True)
def _no_real_device_from_this_suite(monkeypatch, request):
    """Reaching a device primitive is a test defect, not a device operation."""
    import importlib

    def _blocked(name):
        def _raise(*_a, **_k):
            raise AssertionError(
                "%s() reached the real device from the test suite. Patch it (or the production "
                "function that calls it) in this test: unpatched, it SIGKILLs every process "
                "holding the board, which has twice killed a live run." % name
            )

        return _raise

    for mod_path, attr in _DEVICE_PRIMITIVES:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:  # noqa: BLE001 -- a module this checkout lacks cannot be reached either
            continue
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, _blocked("%s.%s" % (mod_path.rsplit(".", 1)[-1], attr)))
    yield


@pytest.fixture(autouse=True)
def _tempdir_is_not_shared(tmp_path_factory, monkeypatch):
    """Point tempfile at a per-test directory pytest will clean up."""
    private = tmp_path_factory.mktemp("systmp")
    monkeypatch.setattr(tempfile, "tempdir", str(private))
    monkeypatch.setenv("TMPDIR", str(private))
    yield private
