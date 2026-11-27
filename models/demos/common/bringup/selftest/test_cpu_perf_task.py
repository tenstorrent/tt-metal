# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F302: a perf-step task with `device: false` is gated, not measured as an on / off device A/B."""

from models.demos.common.bringup.orchestrator import Orchestrator


def test_a_cpu_only_perf_task_is_not_a_perf_pick():
    pick = Orchestrator.perf_pick
    assert not pick(None, {"step": "perf", "role": "perf", "device": False})
    assert pick(None, {"step": "perf", "role": "perf", "device": True})
    assert pick(None, {"step": "perf", "role": "perf"})  # unmarked perf picks stay picks
    assert not pick(None, {"step": "perf", "device": True})  # no role: the profile task, as before


def test_ab_mode_none_is_gated_not_measured():
    """F304: a device perf task with `ab_mode: none` (its gate measures what it needs) is not a perf pick."""
    pick = Orchestrator.perf_pick
    assert not pick(None, {"step": "perf", "role": "perf", "device": True, "ab_mode": "none"})
    assert pick(None, {"step": "perf", "role": "perf", "device": True, "ab_mode": "quick"})
    assert pick(None, {"step": "perf", "role": "perf", "device": True, "ab_mode": "full"})


def test_ab_test_names_stay_short_for_long_env_differences():
    """F306: two units of one test file that differ in many variables (WA.4: an 8192 packed run next to a 5120 one) get
    a short, distinct results-dir name (the first difference + a hash), never one past the file-name limit."""
    pk = " ".join(f"DS41_PACKED_X{i}=on" for i in range(20))
    a = f"BRINGUP_CHUNK=8192 {pk} SKIP_POS=1 scripts/run_safe_pytest.sh --run-all --no-precompile t/test_skip_positions.py"
    b = "BRINGUP_CHUNK=5120 SKIP_POS=1 scripts/run_safe_pytest.sh --run-all --no-precompile t/test_skip_positions.py"
    names = [n for n, _ in Orchestrator.ab_test_names([a, b])]
    assert len(set(names)) == 2 and all(len(n) < 120 for n in names), names
    assert names[0].startswith("test_skip_positions__BRINGUP_CHUNK-8192") and "BRINGUP_CHUNK-5120" in names[1]


def test_an_ab_side_overrides_an_inline_switch_of_the_gate():
    """F308: a gate command that sets the pick's switch inline runs each A/B side at the side's value."""
    from models.demos.common.bringup.orchestrator import side_cmd

    cmd = "A=1 DS41_X=on scripts/run_safe_pytest.sh t.py && DS41_X=on B=2 scripts/run_safe_pytest.sh u.py"
    off = side_cmd(cmd, {"DS41_X": "off"})
    assert off.count("DS41_X=off") == 2 and "DS41_X=on" not in off and "A=1" in off and "B=2" in off
    assert side_cmd(cmd, {"DS41_X": "on"}) == cmd
    assert side_cmd("XDS41_X=on t", {"DS41_X": "off"}) == "XDS41_X=on t"  # another variable's suffix stays
