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
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, Transpose, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.utils import passed_test
from llk2_common import DEST_HALF_TILES, FACES_PER_TILE, TILE_DIM
from llk2_paths import cpp_source, include_dirs

SECTIONS = 2  # copy into both Dest sections, one acquire/commit/pack/release round each


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


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
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
        src_A, formats.output_format, num_faces=FACES_PER_TILE, input_dimensions=input_dimensions
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
