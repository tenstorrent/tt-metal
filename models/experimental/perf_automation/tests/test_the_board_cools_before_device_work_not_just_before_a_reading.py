# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The thermal gate covers every device phase, not only the moment a number is taken.

WHAT THIS COST, measured on a liquid-cooled p300c running Voxtral-Mini-3B.

The gate was written, tested, and correct -- and had exactly ONE production caller,
`_measure_full_pipeline_guarded`. So it protected readings and nothing else.

A Voxtral perf-test build is 30-60 minutes of unbroken device work: HF weights plus 17 graduated
stub uploads, repeated once per generator attempt (nine times on the first run). None of it was
gated. The chips went 57C -> 103C, the AICLK fell 1350 -> 800, and the first measurement then
started on a board already pinned to the clamp -- the exact condition the gate exists to prevent,
reached by the path the gate did not watch.

gemma never exposed this: its device time is mostly gated measurements, which cool between
readings. The build phase is where the heat actually comes from, and it is the phase nothing
watched.

The fix is placement, not policy: call the SAME gate where a device process is LAUNCHED. These
tests assert both launch points do, and that the gate is still defined in exactly one place.
"""

from pathlib import Path

_PA = Path(__file__).resolve().parent.parent


def test_the_launcher_every_device_subprocess_goes_through_gates_first():
    """_run_device_proc is the one funnel for device-touching subprocesses -- discovery, gates,
    coverage, full-pipeline runs. Gating there covers all of them with one call site."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _run_device_proc(")
    body = src[i : i + 1600]
    assert "_wait_for_thermal_headroom_before_device_work" in body, "device subprocesses launch ungated"
    # ...and BEFORE the process is spawned, or the gate is decorative.
    assert body.index("_wait_for_thermal_headroom_before_device_work") < body.index(
        "subprocess.Popen"
    ), "the gate runs after the process has already started"


def test_the_profiled_run_gates_before_building_the_model():
    """A profiled run builds the whole model before it samples anything; starting that on a hot
    board is how a profile ends up sampled entirely at the clamped 800 MHz."""
    src = (_PA / "agent" / "probes.py").read_text()
    i = src.index("def run_profiled(")
    body = src[i : i + 4000]
    assert "_wait_for_thermal_headroom_before_device_work" in body, "profiled runs launch ungated"
    assert body.index("_wait_for_thermal_headroom_before_device_work") < body.index(
        "build_tracy_command"
    ), "the gate runs after the tracy command is built"


def test_the_policy_still_lives_in_exactly_one_place():
    """A second copy of the wait loop would drift from the learned per-board threshold, and the
    threshold is the whole point -- it is derived from THIS board's clamp history, not a constant."""
    run_src = (_PA / "cc_optimize" / "run.py").read_text()
    probes_src = (_PA / "agent" / "probes.py").read_text()
    # The helper delegates; it does not reimplement the wait.
    i = run_src.index("def _wait_for_thermal_headroom_before_device_work(")
    helper = run_src[i : i + 2200]
    # Delegation is the property; the IMPORT MECHANISM is not. run.py is loaded BY PATH with no
    # package, so a bare `from .perf_mcp import ...` raises "attempted relative import with no known
    # parent package" -- silently, inside the gate's own except. That is why the board reached
    # 99-103C on 2026-08-29 with no gate running. _perf_mcp() resolves the sibling either way.
    assert "_perf_mcp()._wait_for_thermal_headroom()" in helper, "the helper does not reuse the gate"
    for src, name in ((run_src, "run.py"), (probes_src, "probes.py")):
        assert "_THERMAL_POLL_S" not in src, f"{name} grew its own copy of the wait loop"
        assert "_clamp_threshold_c" not in src, f"{name} grew its own threshold"


def test_a_gate_that_cannot_run_never_blocks_the_work():
    """A board whose temperature cannot be read must still be usable. Refusing to launch would turn
    a hot board into a failed run -- worse than a reading the clamp check already knows to reject."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _wait_for_thermal_headroom_before_device_work(")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "except Exception" in body, "the gate can raise into the launcher"
    tail = body.split("except Exception")[1][:600]
    # SWALLOW, BUT SAY SO. This used to assert the literal `pass` -- that is asserting on SILENCE,
    # and silence is exactly what let an inert gate go unnoticed on 2026-08-29 while the board held
    # 99-103C for an hour and two chips stopped answering. The property is that the failure does not
    # propagate into the launcher, AND that an operator can see protection is off.
    assert "raise" not in tail, "the gate propagates its own failure into the launcher"
    assert "WARNING" in tail, "an inert gate must announce itself, not fail silently"
