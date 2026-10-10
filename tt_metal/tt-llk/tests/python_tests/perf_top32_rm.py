# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
The top32_rm chunk walk of the DeepSeek sampling kernel (sources/top32_rm_perf.cpp, the perf twin of top32_rm_test.cpp)
on one row and its index row. TILE_LOOP is cycles per 64-element chunk of the row.
"""

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, PerfRunType, format_dict
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT, TOP32_RM_PERF

pytestmark = [skip_for_quasar, skip_for_wormhole]

ELEMENTS_PER_TILE = 1024
ELEMENTS_PER_CHUNK = 64

# (format, 32-bit Dest, row elements); the sampling kernel builds a 32-bit Dest
ROWS = [
    (DataFormat.Float16_b, True, 256),
    (DataFormat.Float16_b, True, 1024),
    (DataFormat.Float16_b, True, 4096),
    (DataFormat.Float16_b, False, 256),
    (DataFormat.Float32, True, 256),
    (DataFormat.Float32, True, 1024),
]
ROW_IDS = [f"{fmt.name}-dest{'32' if d32 else '16'}-row{row}" for fmt, d32, row in ROWS]


@pytest.mark.perf
@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_perf_top32_rm(perf_report, row):
    data_format, dest32, row_elements = row
    formats = InputOutputFormat(data_format, data_format)
    is_32bit = data_format == DataFormat.Float32
    torch_format = format_dict[data_format]
    tiles = -(-row_elements // ELEMENTS_PER_TILE)
    generator = torch.Generator().manual_seed(1234)
    values = torch.randn(tiles * ELEMENTS_PER_TILE, generator=generator).to(
        torch_format
    )
    indices = torch.arange(tiles * ELEMENTS_PER_TILE, dtype=torch.float32).to(
        torch_format
    )
    PerfConfig(
        "sources/top32_rm_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            TOP32_RM_PERF(
                top32_perf_row_elements=row_elements,
                top32_perf_datum_bytes=4 if is_32bit else 2,
            )
        ],
        runtimes=[TILE_COUNT(row_elements // ELEMENTS_PER_CHUNK), LOOP_FACTOR(8)],
        variant_stimuli=StimuliConfig(
            values,
            formats.input_format,
            indices,
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles,
            tile_count_B=tiles,
            tile_count_res=2,
        ),
        unpack_to_dest=is_32bit,
        dest_acc=DestAccumulation.Yes if dest32 else DestAccumulation.No,
    ).run(perf_report)
