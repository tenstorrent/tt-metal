# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""X6 face-transpose PERF vehicle (lane FV, 2026-08-22).

sources/facetranspose_x6_perf.cpp -- the math_transpose_perf kernel with
the math transpose phase on the TYPED X6 surface.  Hand comparator =
test_perf_math_transpose[Int32/Float32, unpack_transpose_faces=Yes,
math_transpose_faces=No] (the 32-bit within-face combination).
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType, Transpose
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import TILE_COUNT, UNPACK_TRANS_FACES


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Int32, DataFormat.Float32], same=True),
)
def test_perf_facetranspose_x6(perf_report, formats):
    if isinstance(formats, tuple):
        formats = formats[0]
    tile_count = 16

    configuration = PerfConfig(
        "sources/facetranspose_x6_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(Transpose.Yes),
        ],
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
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
    )

    configuration.run(perf_report)
