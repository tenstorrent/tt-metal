# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 copy through the Compute API, against the goldens of test_eltwise_unary_datacopy.py and
test_transpose_dest.py.

Shapes and the transpose modifier are compile time in LLK 2.0 and reach the kernel as constexprs (LLK2_COPY_SHAPE);
the formats with compile_time_formats=True. A 32-bit input with a 32-bit Dest unpacks straight to Dest unless it is
transposed (the harness's unpack_to_dest rule).
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    DataCopyGolden,
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, MathFidelity, Transpose, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.tilize_untilize import untilize_block
from helpers.utils import passed_test
from llk2_common import DEST_HALF_TILES, FACES_PER_TILE, TILE_DIM, unwrap_case
from llk2_paths import cpp_source, include_dirs

SECTIONS = 2  # copy into both Dest sections, one acquire/commit/pack/release round each
STRAIGHT_COPIES = 4  # test_llk2_copy_to_dest_repeat: copies straight to Dest in a row


@dataclass
class LLK2_COPY_SHAPE(TemplateParameter):
    tiles_y: int
    tiles_x: int
    transpose: Transpose

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES_Y = {self.tiles_y};\n"
            f"constexpr std::uint32_t TILES_X = {self.tiles_x};\n"
            f"constexpr bool TRANSPOSE = {str(self.transpose == Transpose.Yes).lower()};\n"
            f"constexpr std::uint32_t SECTIONS = {SECTIONS};"
        )


@dataclass
class LLK2_COPY_REPEAT_SHAPE(TemplateParameter):
    tiles_y: int
    tiles_x: int
    first_is_matmul: bool = False

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES_Y = {self.tiles_y};\n"
            f"constexpr std::uint32_t TILES_X = {self.tiles_x};\n"
            f"constexpr std::uint32_t STRAIGHT_COPIES = {STRAIGHT_COPIES};\n"
            f"constexpr bool FIRST_IS_MATMUL = {str(self.first_is_matmul).lower()};"
        )


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
            DataFormat.Bfp4_b,
            DataFormat.Float32,
        ],
        same=True,
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            # A Dest half holds 8 tiles of 16 bits or 4 of 32 bits.
            for shape in [(1, 1), (2, 2), (1, 4), (2, 4)]:
                if shape[0] * shape[1] <= DEST_HALF_TILES[dest_acc]:
                    combos.append((fmt, dest_acc, shape))
    return combos


@parametrize(
    transpose=[Transpose.No, Transpose.Yes],
    format_dest_acc_and_shape=_combinations(),
)
def test_llk2_copy(transpose, format_dest_acc_and_shape):
    formats, dest_acc, (tiles_y, tiles_x) = format_dest_acc_and_shape
    input_dimensions = [tiles_y * TILE_DIM, tiles_x * TILE_DIM]
    unpack_to_dest = (
        formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes
        and transpose == Transpose.No
    )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # The stimuli are laid out tile by tile already: copy and transpose work per tile.
    golden = get_golden_generator(DataCopyGolden)(
        src_A,
        formats.output_format,
        num_faces=FACES_PER_TILE,
        input_dimensions=input_dimensions,
    )
    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)
        golden = t_matrix.transpose_faces_multi_tile(
            golden,
            formats.output_format,
            num_tiles=tile_cnt_A,
            input_dimensions=input_dimensions,
        )
        golden = t_matrix.transpose_within_faces_multi_tile(
            golden,
            formats.output_format,
            num_tiles=tile_cnt_A,
            input_dimensions=input_dimensions,
        )

    # Every round packs the same result to its own part of the output.
    golden = golden.repeat(SECTIONS)

    configuration = TestConfig(
        cpp_source("llk2_copy_compute_api_test.cpp"),
        formats,
        templates=[
            DEST_SYNC(),
            LLK2_COPY_SHAPE(tiles_y, tiles_x, transpose),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A * SECTIONS,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res, dtype=torch_format)
    if unpack_to_dest:
        # Straight to Dest is lossless: every bit of the FP32 input must come back.
        assert torch.equal(
            res_tensor, golden.to(torch_format)
        ), "Unpack to Dest lost bits"
    assert passed_test(
        golden, res_tensor, formats.output_format
    ), "Assert against golden failed"


# Copies straight to Dest in a row, behind a transposed copy (through SrcA) that leaves the math thread's address
# modifiers stepping the Dest row counter: llk2_copy_to_dest_repeat_compute_api_test.cpp has the walk. A straight copy
# that relied on the address modifier it found would, from the fourth one on, mark the other Dest section's rows as
# written, and the section the ring hands out last would hold the previous copy instead of zero. Four tiles: a whole
# 32-bit section, so every straight copy issues 16 face ZEROACCs.
# first: the op in front of the straight copies. A transposed copy leaves slots 0 and 1 stepping; a matmul (of the
# input with itself, so a square tile grid) leaves every slot the FPU ops share stepping the Dest row counter.
@parametrize(
    case=[((1, 4), "transposed_copy"), ((2, 2), "transposed_copy"), ((2, 2), "matmul")]
)
def test_llk2_copy_to_dest_repeat(case):
    (tiles_y, tiles_x), first = unwrap_case(case)
    formats = input_output_formats([DataFormat.Float32], same=True)[0]
    input_dimensions = [tiles_y * TILE_DIM, tiles_x * TILE_DIM]

    # The matmul's products stay in range with inputs in [0, 1); the copies take the harness's default stimuli.
    specs = {}
    if first == "matmul":
        specs = {
            "spec_A": StimuliSpec.uniform(low=0.0, high=1.0),
            "spec_B": StimuliSpec.uniform(low=0.0, high=1.0),
        }
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        **specs,
    )

    copied = get_golden_generator(DataCopyGolden)(
        src_A,
        formats.output_format,
        num_faces=FACES_PER_TILE,
        input_dimensions=input_dimensions,
    )
    t_matrix = get_golden_generator(TransposeGolden)
    transposed = t_matrix.transpose_faces_multi_tile(
        copied,
        formats.output_format,
        num_tiles=tile_cnt_A,
        input_dimensions=input_dimensions,
    )
    transposed = t_matrix.transpose_within_faces_multi_tile(
        transposed,
        formats.output_format,
        num_tiles=tile_cnt_A,
        input_dimensions=input_dimensions,
    )
    torch_format = format_dict[formats.output_format]
    if first == "matmul":
        # The kernel reads the stimuli as tiles: the matrix they hold is their untilized form.
        matrix = untilize_block(src_A, formats.input_format, input_dimensions).flatten()
        first_golden = get_golden_generator(MatmulGolden)(
            matrix,
            matrix,
            formats.output_format,
            MathFidelity.LoFi,  # the kernel's default fidelity
            input_A_dimensions=input_dimensions,
            input_B_dimensions=input_dimensions,
            tilize=True,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
    else:
        first_golden = transposed
    # The first op, the straight copy (every round packs to the same tiles), the unwritten section.
    golden = torch.cat(
        [
            torch.as_tensor(first_golden).to(torch_format).flatten(),
            torch.as_tensor(copied).to(torch_format),
            torch.zeros(len(copied), dtype=torch_format),
        ]
    )

    configuration = TestConfig(
        cpp_source("llk2_copy_to_dest_repeat_compute_api_test.cpp"),
        formats,
        templates=[
            DEST_SYNC(),
            LLK2_COPY_REPEAT_SHAPE(tiles_y, tiles_x, first == "matmul"),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A * 3,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=torch_format)
    n = len(copied)
    assert torch.equal(
        res_tensor[n : 2 * n], golden[n : 2 * n]
    ), "Unpack to Dest lost bits"
    assert (
        torch.count_nonzero(res_tensor[2 * n :]) == 0
    ), "the Dest section the ring handed out is not zero: a copy straight to Dest marked its rows as written"
    assert passed_test(
        golden, res_tensor, formats.output_format
    ), "Assert against golden failed"
