# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Short-face source clearing in the unpack A and unpack AB LLKs (Blackhole only).

A tile with fewer than 16 rows per face unpacks only those rows into SrcA/SrcB, while a datacopy
moves all 16 rows of a face and ELWMUL reads at least 8. The unpackers zero the rest of each face,
so a reduce over the copy or product sums only the tile.

sources/short_face_src_clear_test.cpp runs five sum-to-scalar ops back to back (see its header):
a full 32x32 poison op that fills both banks of SrcA and SrcB with finite values, NaN or Inf, then
short ops interleaved with a full-face control op. Each op's scalar is checked against a golden
sum; the poison op is checked too when its values are finite, so a clear that hits a bank the
math thread still owns would show there.
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
    TILE_COUNT,
    RuntimeParameter,
    TemplateParameter,
)
from helpers.utils import tolerances

ELEMENTS_PER_SLOT = 1024  # every tile sits in its own 32x32 slot of the L1 buffer
NUM_OPS = 5
FULL_OP = 0
CONTROL_OP = 3

FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]


@dataclass
class MUL_PATH(TemplateParameter):
    mul: bool

    def convert_to_cpp(self) -> str:
        return f"constexpr bool MUL_PATH = {'true' if self.mul else 'false'};"


@dataclass
class SHORT_FACE_R_DIM(RuntimeParameter):
    face_r_dim: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SHORT_FACE_R_DIM = {self.face_r_dim};"

    def convert_to_struct_fields(self) -> tuple[str, str]:
        return "std::uint32_t SHORT_FACE_R_DIM;", "I"


def _op_valid_elements(op, short_face_r_dim):
    """Elements an op's tile holds, at the start of its slot: op 0 a full tile, op 3 16x32, the rest short."""
    if op == FULL_OP:
        return ELEMENTS_PER_SLOT
    face_r_dim = 16 if op == CONTROL_OP else short_face_r_dim
    return 2 * face_r_dim * 16


def _poison(kind, n):
    if kind == "finite":
        return torch.empty(n).uniform_(2.0, 4.0)
    return torch.full((n,), float("nan") if kind == "nan" else float("inf"))


def _operand(short_face_r_dim, tile_cnt, poison):
    """One slot per tile. Bytes past a tile's end hold 1000.0, which an unpack would never read."""
    slots = torch.full((NUM_OPS * tile_cnt, ELEMENTS_PER_SLOT), 1000.0)
    for op in range(NUM_OPS):
        n = _op_valid_elements(op, short_face_r_dim)
        for t in range(op * tile_cnt, (op + 1) * tile_cnt):
            slots[t, :n] = _poison(poison, n) if op == FULL_OP else torch.rand(n)
    return slots.to(torch.bfloat16)


@parametrize(
    formats=FORMATS,
    mul_path=[False, True],
    short_face_r_dim=[1, 2, 4, 8, 16],
    tile_cnt=[1, 3],
    poison=["finite", "nan", "inf"],
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
        runtimes=[TILE_COUNT(tile_cnt), SHORT_FACE_R_DIM(short_face_r_dim)],
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
        if op == FULL_OP and poison != "finite":
            continue
        n = _op_valid_elements(op, short_face_r_dim)
        a = src_A[op * tile_cnt : (op + 1) * tile_cnt, :n].float()
        b = src_B[op * tile_cnt : (op + 1) * tile_cnt, :n].float()
        golden = float((a * b).sum() if mul_path else a.sum())
        got = float(res[op * ELEMENTS_PER_SLOT])
        if not abs(got - golden) <= tol.atol + tol.rtol * abs(golden):
            errors.append(f"op{op}: device={got} golden={golden}")
    assert not errors, "; ".join(errors)
