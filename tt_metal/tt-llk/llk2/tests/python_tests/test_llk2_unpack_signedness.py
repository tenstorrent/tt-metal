# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 back-to-back ops that differ in the signedness of their 8-bit integer inputs.

The kernel (llk2_unpack_signedness_compute_api_test.cpp) squares one matrix of bytes alternately as UInt8 and as Int8
(sign and magnitude in L1), ROUNDS times, with nothing between the matmuls. Every op boundary flips ALU_FORMAT_SPEC_REG0 SrcAUnsigned and
SrcBUnsigned, which the unpackers read for as long as an unpack lasts: llk_lib/unpack.h writes them behind a wait
for both unpackers. Without the wait the previous matmul's last tiles are converted with the wrong signedness
(measured: this test then fails with math_spin > 0).

math_spin > 0 keeps the math thread away from its Src banks before every op, so the previous op's last unpacks stand
still in the unpackers while the next configure is issued. Integer matmuls are exact: every round must match its
golden.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from llk2_common import TILE_DATUMS, TILE_DIM, tilize_operands, unwrap_case
from llk2_paths import cpp_source, include_dirs

SIDE = 2  # every tensor is [SIDE, SIDE] tiles
TILES = SIDE * SIDE
ROUNDS = 6
FIDELITY = MathFidelity.HiFi4  # arbitrary 8-bit integers need all four phases


@dataclass
class LLK2_UNPACK_SIGNEDNESS(TemplateParameter):
    math_spin: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t ROUNDS = {ROUNDS};\nconstexpr std::uint32_t MATH_SPIN = {self.math_spin};"


@parametrize(math_spin=[0, 3000])
def test_llk2_unpack_signedness(math_spin):
    math_spin = unwrap_case(math_spin)
    formats = InputOutputFormat(DataFormat.UInt8, DataFormat.Int32, DataFormat.UInt8)
    dims = [SIDE * TILE_DIM, SIDE * TILE_DIM]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=DataFormat.UInt8,
        input_dimensions_A=dims,
        stimuli_format_B=DataFormat.UInt8,
        input_dimensions_B=dims,
    )
    # The harness's UInt8 stimuli stay below 128; a signed read must change the operand, so take the whole range
    # for a quarter of it.
    src_A = torch.as_tensor(src_A)
    generator = torch.Generator().manual_seed(math_spin)
    high = torch.randint(128, 256, src_A.shape, generator=generator).to(src_A.dtype)
    src_A = torch.where(
        torch.rand(src_A.shape, generator=generator) < 0.25, high, src_A
    )
    assert (src_A > 127).any()
    # The same bytes as Int8 in L1: bit 7 the sign, bits 6..0 the magnitude.
    as_int = src_A.to(torch.int32)
    signed = torch.where(as_int > 127, -(as_int & 0x7F), as_int).to(torch.int8)

    def square_golden(values):
        # Integer matmuls are exact: the product in int64, as tiles.
        matrix = values.to(torch.int64).view(dims[0], dims[1])
        return tilize_block(
            matrix @ matrix, dimensions=dims, stimuli_format=formats.output_format
        ).flatten()

    torch_format = format_dict[formats.output_format]
    goldens = [
        square_golden(src_A).to(torch_format),
        square_golden(signed).to(torch_format),
    ]
    tilized_A, _ = tilize_operands(src_A, dims, src_A, dims, DataFormat.UInt8)
    assert not torch.equal(goldens[0], goldens[1])

    configuration = TestConfig(
        cpp_source("llk2_unpack_signedness_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(FIDELITY),
            DEST_SYNC(),
            LLK2_UNPACK_SIGNEDNESS(math_spin),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            DataFormat.UInt8,
            tilized_A,
            DataFormat.UInt8,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=2 * ROUNDS * TILES,
        ),
        dest_acc=DestAccumulation.Yes,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )
    res = configuration.run().result
    op_len = TILES * TILE_DATUMS
    assert len(res) == 2 * ROUNDS * op_len, "Result tensor does not hold every round"
    results = torch.tensor(res, dtype=torch_format).view(ROUNDS, 2, op_len)
    wrong = [
        (r, name)
        for r in range(ROUNDS)
        for j, name in enumerate(("unsigned", "signed"))
        if not torch.equal(results[r][j], goldens[j])
    ]
    assert not wrong, f"wrong results (round, operand kind): {wrong}"
