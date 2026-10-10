# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the experimental block row max (reduce_block_max_row).

tile_cnt counts input tiles, as in perf_reduce.py, so the report's cycles per tile are per input tile.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, REDUCE_BLOCK_CT_DIM, TILE_COUNT


def _dest_accs(formats):
    if formats.output_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]


@pytest.mark.perf
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    ],
    dest_acc=_dest_accs,
    block_ct_dim=[1, 2, 4, 8],
)
def test_perf_reduce_block_max(perf_report, formats, dest_acc, block_ct_dim):
    tile_count = 16
    configuration = PerfConfig(
        "sources/reduce_block_max_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[REDUCE_BLOCK_CT_DIM(block_ct_dim)],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(64)],
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
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    configuration.run(perf_report)
