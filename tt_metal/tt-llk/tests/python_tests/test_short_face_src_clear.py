# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Short-face source clearing in the unpack A and unpack AB LLKs (Blackhole only).

A tile with fewer than 16 rows per face unpacks only those rows into SrcA/SrcB, while a datacopy
moves all 16 rows of a face and ELWMUL reads at least 8. The unpackers zero the rest of each SrcA
face, so a reduce over the copy or product sums only the tile. Stale SrcB rows meet zero SrcA rows,
and the FPU gives zero for that product even when the SrcB value is NaN or Inf.

sources/short_face_src_clear_test.cpp runs seven ops back to back (see its header): a full 32x32
poison op that fills both banks of SrcA and SrcB with finite values, NaN or Inf, then short ops
interleaved with a full-face control op, then a second poison op with no reduce tail followed by
a short op. Each op with a reduce tail is checked against a golden sum; the first poison op is
checked too when its values are finite, so a clear that hits a bank the math thread still owns
would show there. The tail zeroes a bank of each source, so only the second poison op leaves
stale data in SrcB for the next short op.
"""

from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.tile_constants import FACE_C_DIM, MAX_FACE_R_DIM, MAX_TILE_ELEMENTS
from helpers.utils import tolerances

# Every tile sits in its own 32x32 slot of the L1 buffer. Op indexes match the kernel.
NUM_OPS = 7
FULL_OP = 0
CONTROL_OP = 3
UNREDUCED_OP = 5
POISON_OPS = (FULL_OP, UNREDUCED_OP)
POISON = {
    "finite": lambda n: torch.empty(n).uniform_(2.0, 4.0),
    "nan": lambda n: torch.full((n,), float("nan")),
    "inf": lambda n: torch.full((n,), float("inf")),
}

FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]


@dataclass
class MUL_PATH(TemplateParameter):
    mul: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool MUL_PATH = {'true' if self.mul else 'false'};"


def _op_valid_elements(op, short_face_r_dim):
    """Elements an op's tile holds, at the start of its slot: poison ops a full tile, the control op 16x32, the rest short."""
    if op in POISON_OPS:
        return MAX_TILE_ELEMENTS
    face_r_dim = MAX_FACE_R_DIM if op == CONTROL_OP else short_face_r_dim
    return 2 * face_r_dim * FACE_C_DIM


def _operand(short_face_r_dim, tile_cnt, poison):
    """One slot per tile. Bytes past a tile's end hold 1000.0, which an unpack would never read."""
    slots = torch.full((NUM_OPS * tile_cnt, MAX_TILE_ELEMENTS), 1000.0)
    for op in range(NUM_OPS):
        n = _op_valid_elements(op, short_face_r_dim)
        for t in range(op * tile_cnt, (op + 1) * tile_cnt):
            slots[t, :n] = POISON[poison](n) if op in POISON_OPS else torch.rand(n)
    return slots.to(torch.bfloat16)


@parametrize(
    formats=FORMATS,
    mul_path=[False, True],
    short_face_r_dim=[1, 2, 4, 8, 16],
    tile_cnt=[1, 3],
    poison=list(POISON),
)
def test_short_face_src_clear(formats, mul_path, short_face_r_dim, tile_cnt, poison):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("Blackhole-only LLK behavior")

    dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else DestAccumulation.No
    )
    src_A = _operand(short_face_r_dim, tile_cnt, poison)
    src_B = _operand(short_face_r_dim, tile_cnt, poison)

    configuration = TestConfig(
        "sources/short_face_src_clear_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.HiFi4),
            MUL_PATH(mul_path),
        ],
        runtimes=[TILE_COUNT(tile_cnt), TEST_FACE_DIMS(face_r_dim=short_face_r_dim)],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=NUM_OPS * tile_cnt,
            tile_count_B=NUM_OPS * tile_cnt,
            tile_count_res=NUM_OPS,
            write_full_tiles=True,
        ),
        dest_acc=dest_acc,
    )
    res = configuration.run().result

    tol = tolerances[formats.output_format]
    errors = []
    for op in range(NUM_OPS):
        if op == UNREDUCED_OP or (op == FULL_OP and poison != "finite"):
            continue
        n = _op_valid_elements(op, short_face_r_dim)
        a = src_A[op * tile_cnt : (op + 1) * tile_cnt, :n].float()
        b = src_B[op * tile_cnt : (op + 1) * tile_cnt, :n].float()
        golden = float((a * b).sum() if mul_path else a.sum())
        got = float(res[op * MAX_TILE_ELEMENTS])
        if not abs(got - golden) <= tol.atol + tol.rtol * abs(golden):
            errors.append(f"op{op}: device={got} golden={golden}")
    assert not errors, "; ".join(errors)
