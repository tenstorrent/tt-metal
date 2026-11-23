# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 back-to-back ops whose unpack configuration differs.

The kernel (llk2_reconfig_compute_api_test.cpp) copies input A and input B alternately, ROUNDS times, with nothing
between the ops. The inputs have different formats, so every op boundary on the unpack thread changes real unpacker
state (L1 and register formats, strides; with a 32-bit Dest also SrcA versus straight to Dest) while the previous
op's tiles may still be on their way. Every round must reproduce its input.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.utils import passed_test
from llk2_common import DEST_HALF_TILES, FACES_PER_TILE, TILE_DIM
from llk2_paths import cpp_source, include_dirs

ROUNDS = 4  # A, B, A, B: three op boundaries, each a reconfiguration to the other format


@dataclass
class LLK2_RECONFIG(TemplateParameter):
    tiles_y: int
    tiles_x: int
    a_to_dest: bool
    b_to_dest: bool

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t TILES_Y = {self.tiles_y};\n"
            f"constexpr std::uint32_t TILES_X = {self.tiles_x};\n"
            f"constexpr std::uint32_t ROUNDS = {ROUNDS};\n"
            f"constexpr bool A_TO_DEST = {str(self.a_to_dest).lower()};\n"
            f"constexpr bool B_TO_DEST = {str(self.b_to_dest).lower()};"
        )


# (input A, input B, output, Dest mode). A 32-bit input with a 32-bit Dest goes straight to Dest; everything else
# through SrcA.
CASES = [
    (DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Bfp8_b, DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float16_b, DestAccumulation.Yes),
    (DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float32, DestAccumulation.Yes),
    (DataFormat.Float16_b, DataFormat.Float32, DataFormat.Float32, DestAccumulation.Yes),
    (DataFormat.Float32, DataFormat.Bfp8_b, DataFormat.Float32, DestAccumulation.Yes),
]


# Every shape fits a Dest half in both Dest modes (4 tiles of 32 bits).
SHAPES = [(1, 1), (1, 2), (2, 2), (1, 4)]
assert all(y * x <= min(DEST_HALF_TILES.values()) for y, x in SHAPES)


@parametrize(case=CASES, shape=SHAPES)
def test_llk2_reconfig(case, shape):
    format_a, format_b, format_out, dest_acc = case
    tiles_y, tiles_x = shape
    input_dimensions = [tiles_y * TILE_DIM, tiles_x * TILE_DIM]
    formats = InputOutputFormat(format_a, format_out, format_b)
    to_dest = [fmt.is_32_bit() and dest_acc == DestAccumulation.Yes for fmt in (format_a, format_b)]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=format_a,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=format_b,
        input_dimensions_B=input_dimensions,
    )

    copy_golden = get_golden_generator(DataCopyGolden)
    golden_a = copy_golden(
        src_A, format_out, num_faces=FACES_PER_TILE, input_dimensions=input_dimensions, input_format=format_a
    )
    golden_b = copy_golden(
        src_B, format_out, num_faces=FACES_PER_TILE, input_dimensions=input_dimensions, input_format=format_b
    )
    golden = torch.cat([golden_a if r % 2 == 0 else golden_b for r in range(ROUNDS)])

    configuration = TestConfig(
        cpp_source("llk2_reconfig_compute_api_test.cpp"),
        formats,
        templates=[
            DEST_SYNC(),
            LLK2_RECONFIG(tiles_y, tiles_x, to_dest[0], to_dest[1]),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            format_a,
            src_B,
            format_b,
            format_out,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A * ROUNDS,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=to_dest[0],
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(golden), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=format_dict[format_out])
    assert passed_test(golden, res_tensor, format_out), "Assert against golden failed"
