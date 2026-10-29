# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Golden functional runs of the cells the half sync end fold (matmul_half_end_fold) takes since it covers HiFi4 and HiFi3 with a 16-bit DEST for
block float operands. The bodies are test_matmul and test_matmul_mix."""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from test_matmul import generate_format_aware_matmul_combinations
from test_matmul import test_matmul as _matmul_body

try:
    from test_matmul_mix import test_matmul_mix as _mix_body
except ImportError:  # trees without the mixed format test
    _mix_body = None

F = DataFormat
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
WIDE = (F.Float16_b, F.Float16, F.Float32)
BLOCK = (F.Bfp8_b, F.Bfp4_b)
OUTPUTS = (F.Float16_b, F.Float32, F.Bfp8_b)  # Bfp4_b outputs fail the golden on main as well


def fold_changed(fidelity, in0, in1, out, dest_acc):
    """The cells where the end fold at half sync is taken now and was not before the HiFi4 and HiFi3 widening."""
    if fidelity not in (MathFidelity.HiFi3, MathFidelity.HiFi4):
        return False
    wide = (in0 in WIDE and in1 in WIDE) or (in0 == F.Bfp8_b and in1 == F.Bfp8_b and out == F.Float32)
    before = (fidelity == MathFidelity.HiFi4 or dest_acc == DestAccumulation.No) and wide
    now = fidelity == MathFidelity.HiFi4 or (
        dest_acc == DestAccumulation.No and not (in0 == F.Bfp8_b and in1 == F.Float16_b and out == F.Float32)
    )
    return now and not before


def sampled(combos):
    """Three block shapes (smallest, middle, largest) of every format and DEST cell."""
    cells = {}
    for combo in combos:
        cells.setdefault((str(combo[0]), combo[1]), []).append(combo)
    return [group[i] for group in cells.values() for i in sorted({0, len(group) // 2, len(group) - 1})]


def fold_params(formats):
    combos = sampled(generate_format_aware_matmul_combinations(formats, DEST_ACC_MODES))
    return [
        (fidelity, combo)
        for fidelity in (MathFidelity.HiFi3, MathFidelity.HiFi4)
        for combo in combos
        if fold_changed(
            fidelity,
            combo[0].input_format_B if combo[0].input_format_B is not None else combo[0].input_format,
            combo[0].input_format,
            combo[0].output_format,
            combo[1],
        )
    ]


FOLD_SAME = fold_params([InputOutputFormat(i, o) for i in BLOCK for o in OUTPUTS])
FOLD_MIX = fold_params(
    [
        InputOutputFormat(a, o, input_format_B=b)
        for a in BLOCK + WIDE[:1] + WIDE[2:]
        for b in BLOCK + WIDE[:1] + WIDE[2:]
        if a != b and (a in BLOCK or b in BLOCK)
        for o in OUTPUTS
    ]
)


@pytest.mark.parametrize("math_fidelity,format_dest_acc_and_dims", FOLD_SAME)
def test_matmul_fold_cells(math_fidelity, format_dest_acc_and_dims):
    _matmul_body(math_fidelity, format_dest_acc_and_dims)


@pytest.mark.skipif(_mix_body is None, reason="no mixed format matmul test in this tree")
@pytest.mark.parametrize("math_fidelity,format_dest_acc_and_dims", FOLD_MIX)
def test_matmul_mix_fold_cells(math_fidelity, format_dest_acc_and_dims):
    _mix_body(math_fidelity, format_dest_acc_and_dims, DestSync.Half)
