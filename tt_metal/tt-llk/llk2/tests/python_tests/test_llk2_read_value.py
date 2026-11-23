# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 read_value: a scalar from a dataflow buffer on all three threads (compute_api/sync.h).

The kernel runs rounds. Each round reads from a tile of 32-bit words in a control buffer its number of tiles n (word
0, and again as a 16-bit word further into the tile) and whether it subtracts (word 1), then adds or subtracts n tiles
of a and b and packs n tiles. Unpack, math and pack each use n for their own loops and page counts and take the
branch, so the result is right only if all three got the same values in every round. The counts and the choices are
stimuli, not compile-time parameters: one build runs every sequence of them.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.utils import passed_test
from llk2_common import TILE_DATUMS, TILE_DIM, tilize_operands
from llk2_paths import cpp_source, kernel_test_include_dirs

WIDE_INDEX = 6  # the 16-bit word the kernel reads the count from a second time: the low half of 32-bit word 3
CT_DIM = 4  # the most tiles a round takes: a 32-bit Dest section


@dataclass
class LLK2_READ_VALUE(TemplateParameter):
    rounds: int
    total_tiles: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t ROUNDS = {self.rounds};\n"
            f"constexpr std::uint32_t TOTAL_TILES = {self.total_tiles};\n"
            f"constexpr std::uint32_t CT_DIM = {CT_DIM};\n"
            f"constexpr std::uint32_t WIDE_INDEX = {WIDE_INDEX};"
        )


def _formats():
    fp16b, fp32 = input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32], same=True
    )
    return [(fp16b, DestAccumulation.No), (fp32, DestAccumulation.Yes)]


# (tiles, subtract) per round. 8 tiles in all, in 2 rounds to 8 (more reads than a mailbox holds).
_ROUNDS = [
    [(4, 0), (4, 1)],
    [(1, 1), (3, 0), (4, 1)],
    [(3, 0), (1, 0), (2, 1), (2, 0)],
    [(2, 1), (1, 1), (1, 0), (3, 1), (1, 0)],
    [(1, 0), (1, 1), (1, 1), (1, 0), (1, 1), (1, 0), (1, 0), (1, 1)],
]


@parametrize(case=_formats(), rounds=_ROUNDS)
def test_llk2_read_value(case, rounds):
    formats, dest_acc = case
    counts = [count for count, _ in rounds]
    total = sum(counts)
    dims = [TILE_DIM, total * TILE_DIM]
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims,
    )
    a, b = tilize_operands(src_A, dims, src_B, dims, formats.input_format)
    sums, differences = (
        get_golden_generator(EltwiseBinaryGolden)(
            op,
            a,
            b,
            formats.output_format,
            MathFidelity.LoFi,
            input_format=formats.input_format,
            input_format_B=formats.input_format,
            dest_acc=dest_acc,
        )
        for op in (MathOperation.Elwadd, MathOperation.Elwsub)
    )

    # One tile of words per round: the count in word 0 and in word WIDE_INDEX / 2, the choice in word 1, other values
    # around them. The golden: each round's tiles from the sums or from the differences.
    words = torch.full((len(rounds), TILE_DATUMS), 0x7FFF0000, dtype=torch.int64)
    golden = sums.clone()
    first = 0
    for round_index, (count, subtract) in enumerate(rounds):
        words[round_index, 0] = count
        words[round_index, 1] = subtract
        words[round_index, WIDE_INDEX // 2] = count
        if subtract:
            datums = slice(first * TILE_DATUMS, (first + count) * TILE_DATUMS)
            golden[datums] = differences[datums]
        first += count
    words = words.flatten()

    configuration = TestConfig(
        cpp_source("llk2_read_value_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.LoFi),
            DEST_SYNC(),
            LLK2_READ_VALUE(len(counts), total),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            a,
            formats.input_format,
            b,
            formats.input_format,
            formats.output_format,
            tile_count_A=total,
            tile_count_B=total,
            tile_count_res=total,
            buffer_C=words,
            stimuli_C_format=DataFormat.UInt32,
            tile_count_C=len(counts),
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=kernel_test_include_dirs(),
    )
    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    assert passed_test(
        golden,
        torch.tensor(res, dtype=format_dict[formats.output_format]),
        formats.output_format,
    ), "Assert against golden failed"
