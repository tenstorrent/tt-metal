# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shadow (not checked in), fast untilize half: the routing rule for #58736. The fast untilize and the plain pack untilize (in the ttnn
helper's blocks: the largest divisor of the row up to 4 with a 32-bit DEST, up to 8 with a 16-bit DEST) at the rows the
helper sends to the fast untilize today, wider than four tiles: bf16, bfp8_b and bfp4_b inputs to bf16, both DEST
modes, DEST sync half, one tile row, LOOP_FACTOR 32."""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import DEST_SYNC, LOOP_FACTOR, TILE_COUNT, generate_input_dim

FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Float16_b),
]
WIDTHS = [5, 6, 7, 8, 9, 10, 12, 14, 16, 24, 25, 32]


def _stimuli(formats, tiles):
    return StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tiles,
        tile_count_B=tiles,
        tile_count_res=tiles,
    )


@pytest.mark.perf
@parametrize(formats=FORMATS, dest_acc=[DestAccumulation.No, DestAccumulation.Yes], ct_dim=WIDTHS)
def test_perf_r3_fast_sweep(perf_report, formats, dest_acc, ct_dim):
    dimensions = (32, ct_dim * 32)
    PerfConfig(
        "sources/fast_untilize_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[generate_input_dim(dimensions, dimensions), DEST_SYNC(DestSync.Half)],
        runtimes=[TILE_COUNT(ct_dim), LOOP_FACTOR(32)],
        variant_stimuli=_stimuli(formats, ct_dim),
        compile_time_formats=True,
        dest_acc=dest_acc,
    ).run(perf_report)
