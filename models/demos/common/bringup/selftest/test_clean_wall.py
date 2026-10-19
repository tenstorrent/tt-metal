# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F114: a profiled test_profile.py run is preceded by a clean wall run with the device profiler off."""

from models.demos.common.bringup.core.gate import gate_command, with_clean_wall

PROF = (
    "TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 "
    "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 scripts/run_safe_pytest.sh --run-all --no-precompile "
    "models/demos/common/bringup/tests/test_profile.py"
)


def test_profiled_run_gets_a_clean_wall_run_first():
    ladder = "BRINGUP_RUNG=last scripts/run_safe_pytest.sh --run-all models/demos/common/bringup/tests/test_ladder.py"
    segs = [s.strip() for s in with_clean_wall(f"{ladder} && {PROF}").split("&&")]
    assert segs[0] == ladder
    assert segs[1].startswith("BRINGUP_PROFILE_WALL=1 ") and "TT_METAL_" not in segs[1]
    assert segs[1].endswith("tests/test_profile.py")
    assert segs[2] == f"BRINGUP_WALL_CLEAN=1 {PROF}"


def test_other_commands_unchanged_and_idempotent():
    ladder = "scripts/run_safe_pytest.sh models/demos/common/bringup/tests/test_ladder.py"
    assert with_clean_wall(ladder) == ladder
    once = with_clean_wall(PROF)
    assert with_clean_wall(once) == once
    ops = "BRINGUP_FULL_PREFILL=1 BRINGUP_PROFILE_OPS=1 " + PROF
    assert "BRINGUP_PROFILE_OPS" not in with_clean_wall(ops).split("&&")[0]


def test_gate_command_applies_it():
    task = {"step": "perf", "gate": {"cmd": PROF}}
    assert "BRINGUP_PROFILE_WALL=1" in gate_command(task)
