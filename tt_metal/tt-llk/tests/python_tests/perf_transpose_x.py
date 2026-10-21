# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""llk_analysis transpose family (Blackhole) shadow perf driver for sources/transpose_perf_x.cpp, the rows with the four run types (one run-type set per module): transpose_dest alone
(mode 0), transpose_tile of 32-bit tensors (mode 1, unpack to dest plus transpose_dest<false, true>) and transpose_tile
of 16-bit tensors (mode 2, the unpacker transpose), with the DEST section size as a parameter (the ttnn transpose kernel
uses one tile per section). Not part of the tree."""

from collections import OrderedDict
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_TRANSPOSE_FACES,
    TILE_COUNT,
    TemplateParameter,
)


@dataclass
class TRANSPOSE_MODE(TemplateParameter):
    transpose_mode: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t TRANSPOSE_MODE = {self.transpose_mode};"


@dataclass
class SECTION_TILES(TemplateParameter):
    section_tiles: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t SECTION_TILES = {self.section_tiles};"


F = DataFormat
NO, YES = DestAccumulation.No, DestAccumulation.Yes
L2L = PerfRunType.L1_TO_L1
ISO = [PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE]


def v(mode, fin, fout, acc, faces, section, run_types, utd=False):
    return dict(
        mode=mode,
        formats=InputOutputFormat(fin, fout),
        acc=acc,
        faces=faces,
        section=section,
        run_types=run_types,
        unpack_to_dest=utd,
    )


VARIANTS = OrderedDict(
    [
        # transpose_dest alone
        ("dest_bf16_faces_s8", v(0, F.Float16_b, F.Float16_b, NO, Transpose.Yes, 8, ISO)),
        ("dest_fp32_faces_s4", v(0, F.Float32, F.Float32, YES, Transpose.Yes, 4, ISO)),
        ("dest_fp32_within_s4", v(0, F.Float32, F.Float32, YES, Transpose.No, 4, ISO)),
        ("dest_int32_faces_s4", v(0, F.Int32, F.Int32, YES, Transpose.Yes, 4, ISO)),
        ("dest_int32_within_s4", v(0, F.Int32, F.Int32, YES, Transpose.No, 4, ISO)),
        ("dest_bf16acc_faces_s4", v(0, F.Float16_b, F.Float16_b, YES, Transpose.Yes, 4, ISO)),
        # transpose_tile, 16-bit tensors (the unpacker transpose)
        ("tile16_bf16_s8", v(2, F.Float16_b, F.Float16_b, NO, Transpose.Yes, 8, ISO)),
        ("tile16_bfp8_s8", v(2, F.Bfp8_b, F.Bfp8_b, NO, Transpose.Yes, 8, ISO)),
        ("tile16_fp16_s8", v(2, F.Float16, F.Float16, NO, Transpose.Yes, 8, ISO)),
        ("tile16_bf16acc_s4", v(2, F.Float16_b, F.Float16_b, YES, Transpose.Yes, 4, ISO)),
    ]
)


@pytest.mark.perf
@parametrize(variant=list(VARIANTS.keys()))
def test_perf_transpose_x(perf_report, variant):
    if isinstance(variant, tuple):
        variant = variant[0]
    p = VARIANTS[variant]
    formats = p["formats"]
    tile_count = 16

    configuration = PerfConfig(
        "sources/transpose_perf_x.cpp",
        formats,
        run_types=p["run_types"],
        templates=[
            TRANSPOSE_MODE(p["mode"]),
            SECTION_TILES(p["section"]),
            MATH_TRANSPOSE_FACES(p["faces"]),
        ],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(16)],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        unpack_to_dest=p["unpack_to_dest"],
        dest_acc=p["acc"],
    )
    configuration.run(perf_report)
