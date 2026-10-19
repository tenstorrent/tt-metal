# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch sweep (not for merge): Blackhole fast tilize (experimental unpack, math, pack) over formats, DEST width and
row width; the tree's own sources/fast_tilize_bh_test.cpp. FTW emits #59143's W-step selection (the ttnn.tilize rule,
tilize_helpers.inl); trees without it ignore the two constants."""
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, NUM_FACES, TILE_COUNT, TemplateParameter, generate_input_dim

F = DataFormat
NO, YES = DestAccumulation.No, DestAccumulation.Yes


@dataclass
class FTW(TemplateParameter):
    ft_w_step: bool = False
    ft_two_tile_row: bool = False

    def convert_to_cpp(self) -> str:
        return (f"constexpr bool FAST_TILIZE_W_STEP = {str(self.ft_w_step).lower()};\n"
                f"constexpr bool FAST_TILIZE_TWO_TILE_ROW = {str(self.ft_two_tile_row).lower()};")


def ftw(ct, fmt):
    two = ct == 2
    return FTW(fmt.input_format.size == 2 and (ct >= 3 or (two and fmt.output_format.size != 2)), two)


FMTS = [InputOutputFormat(a, b) for a, b in (
    (F.Float16_b, F.Float16_b), (F.Float16_b, F.Bfp8_b), (F.Float16_b, F.Bfp4_b), (F.Float32, F.Float16_b),
    (F.Float32, F.Bfp8_b), (F.Float32, F.Bfp4_b), (F.Float32, F.Float32))]
CASES = [(fmt, acc, ct) for fmt in FMTS for acc in (NO, YES) for ct in (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32)]


@pytest.mark.perf
@pytest.mark.parametrize("fmt,acc,ct", CASES,
                         ids=[f"{f.input_format.name}-{f.output_format.name}-{a.name}-c{c}" for f, a, c in CASES])
def test_perf_fast_tilize_allcfg(perf_report, fmt, acc, ct):
    dims = (32, ct * 32)
    configuration = PerfConfig(
        "sources/fast_tilize_bh_test.cpp",
        fmt,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[ftw(ct, fmt)],
        runtimes=[generate_input_dim(dims, dims), TILE_COUNT(ct), LOOP_FACTOR(32), NUM_FACES(4)],
        variant_stimuli=StimuliConfig(None, fmt.input_format, None, fmt.input_format, fmt.output_format,
                                      tile_count_A=ct, tile_count_B=ct, tile_count_res=ct),
        dest_acc=acc,
        compile_time_formats=True,
    )
    configuration.run(perf_report)
