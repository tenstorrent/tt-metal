#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parser checks for tests/run_ttsim_regression.sh reading ttsim-skip-list.yaml."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tt_metal/tt-llk/tests/run_ttsim_regression.sh"
LLK_PREFIX = "tt_metal/tt-llk/tests/python_tests/"
VECTOR_ADD = "quasar/test_vector_add_l1_quasar.py::test_vector_add_l1_quasar"
RAND = "quasar/test_rand_quasar.py::test_rand_quasar"
RAND_SEED = "quasar/test_rand_quasar.py::test_rand_seed_quasar"
# Exact parametrizations. A shorter prefix would also drop passing cases of the same test.
SFPENCC = (
    "quasar/test_eltwise_binary_sfpu_quasar.py::test_eltwise_binary_sfpu_int_quasar"
    "[formats:Int32->Int32-dest_acc:Yes-mathop:SfpuGcd-tile_indices:(0, 1, 0)]",
    "quasar/test_eltwise_binary_sfpu_quasar.py::test_eltwise_binary_sfpu_int_quasar"
    "[formats:Int32->Int32-dest_acc:Yes-mathop:SfpuGcd-tile_indices:(2, 3, 0)]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_gcd__op1_math2_SfpuGcd__operand1_format_Int32]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_gcd_pairs_extremes]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_gcd_pairs_fibonacci_nan]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_gcd_pairs_shared_factor]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_gcd_pairs_worst_case]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_lcm_pairs_equal_signs__op1_math2_SfpuGcd]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_lcm_pairs_extremes__op1_math2_SfpuGcd]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_lcm_pairs_shared_factor__op1_math2_SfpuGcd]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_lcm_pairs_worst_case__op1_math2_SfpuGcd]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_int32_lcm_pairs_zeros_units__op1_math2_SfpuGcd]",
)
CFG_WR32 = (
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_rand_dest16__op1_math2_Rand__op1_math3_Neg__op1_math4_Add1__op1_math5_Add1__op1_math6_Abs__op1_math7_Neg__op1_math8_Add1__op1_math9_GreaterThanEqualZero__operand1_format_Float16_b]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_rand_dest16__op1_math2_Rand__op1_math3_Neg__op1_math4_Add1__op1_math5_Add1__op1_math6_Abs__op1_math7_Neg__op1_math8_Add1__op1_math9_GreaterThanEqualZero__operand1_format_Float16]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_rand_dest32__op1_math2_Rand__op1_math3_Neg__op1_math4_Add1__op1_math5_Add1__op1_math6_Abs__op1_math7_Neg__op1_math8_Add1__op1_math9_GreaterThanEqualZero__operand1_format_Float16_b]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_rand_dest32__op1_math2_Rand__op1_math3_Neg__op1_math4_Add1__op1_math5_Add1__op1_math6_Abs__op1_math7_Neg__op1_math8_Add1__op1_math9_GreaterThanEqualZero__operand1_format_Float16]",
    "quasar/test_fused_quasar.py::test_fuser[quasar/sfpu_rand_dest32__op1_math2_Rand__op1_math3_Neg__op1_math4_Add1__op1_math5_Add1__op1_math6_Abs__op1_math7_Neg__op1_math8_Add1__op1_math9_GreaterThanEqualZero__operand1_format_Float32]",
)


def _run(*args, ignore_file=None):
    cmd = ["bash", str(SCRIPT), *args, "--list-ignore"]
    if ignore_file is not None:
        cmd[2:2] = ["--ignore-file", str(ignore_file)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_default_list_selects_quasar_llk_entries():
    quasar = _run("-a", "quasar")
    blackhole = _run("-a", "blackhole")
    wormhole = _run("-a", "wormhole")

    assert quasar.returncode == 0, quasar.stderr
    assert quasar.stdout.splitlines() == [
        VECTOR_ADD,
        RAND,
        RAND_SEED,
        *CFG_WR32,
        *SFPENCC,
    ]
    assert blackhole.returncode == 0, blackhole.stderr
    assert blackhole.stdout == ""
    assert wormhole.returncode == 0, wormhole.stderr
    assert wormhole.stdout == ""


def test_no_ignore_prints_nothing():
    result = _run("-a", "quasar", "--no-ignore")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_llk_paths_are_stripped_and_other_suites_are_left_alone(tmp_path):
    ignore_file = tmp_path / "ttsim-skip-list.yaml"
    ignore_file.write_text(
        "\n".join(
            [
                "wormhole_b0:",
                "  - tests/ttnn/unit_tests/test_conv.py  # ttnn, not collected here",
                f"  - {LLK_PREFIX}test_wh.py::test_wh  # llk",
                "blackhole:",
                f"  - {LLK_PREFIX}test_bh.py::test_bh",
                "quasar:",
                f"  - {LLK_PREFIX}quasar/test_other.py::test_other",
                "  - tests/ttnn/unit_tests/test_quasar_ttnn.py",
            ]
        )
        + "\n"
    )

    wormhole = _run("-a", "wormhole", ignore_file=ignore_file)
    blackhole = _run("-a", "blackhole", ignore_file=ignore_file)
    quasar = _run("-a", "quasar", ignore_file=ignore_file)

    assert wormhole.returncode == 0, wormhole.stderr
    assert wormhole.stdout.splitlines() == ["test_wh.py::test_wh"]
    assert blackhole.returncode == 0, blackhole.stderr
    assert blackhole.stdout.splitlines() == ["test_bh.py::test_bh"]
    assert quasar.returncode == 0, quasar.stderr
    assert quasar.stdout.splitlines() == ["quasar/test_other.py::test_other"]


def test_loader_rejects_an_entry_before_any_key(tmp_path):
    ignore_file = tmp_path / "ttsim-skip-list.yaml"
    ignore_file.write_text(f"- {LLK_PREFIX}quasar/foo.py::test_x\n")
    result = _run("-a", "quasar", ignore_file=ignore_file)
    assert result.returncode != 0
    assert "skip entry before any architecture key" in result.stderr


def test_missing_ignore_file(tmp_path):
    result = _run("-a", "quasar", ignore_file=tmp_path / "missing.yaml")
    assert result.returncode != 0
    assert "ttsim skip list not found" in result.stderr


def test_ignore_file_requires_a_path():
    result = subprocess.run(["bash", str(SCRIPT), "--ignore-file"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires a path" in result.stderr

    result = subprocess.run(
        ["bash", str(SCRIPT), "--ignore-file", "--no-ignore"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "requires a path" in result.stderr
