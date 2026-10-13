# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType, Transpose
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    TemplateParameter,
)


@dataclass
class UNPACK_TRANSPOSE_BLOCK(TemplateParameter):
    def convert_to_cpp(self) -> str:
        return "#define UNPACK_TRANSPOSE_BLOCK"


@dataclass
class TRANSPOSE_BLOCK_BACK_TO_BACK(TemplateParameter):
    def convert_to_cpp(self) -> str:
        return "#define TRANSPOSE_BLOCK_BACK_TO_BACK true"


# perf_unpack_transpose.py's faces and within face rows with each DEST section unpacked by one block call
@skip_for_wormhole
@skip_for_quasar
@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b],
        same=True,
    ),
    unpack_transpose_faces=[Transpose.Yes],
    unpack_transpose_within_face=[Transpose.Yes],
)
def test_perf_unpack_transpose_block(
    perf_report, formats, unpack_transpose_faces, unpack_transpose_within_face
):
    tile_count = 16

    configuration = PerfConfig(
        "sources/unpack_transpose_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE],
        templates=[
            UNPACK_TRANSPOSE_BLOCK(),
            # uncompressed tiles lie back to back, as in a CB
            *(
                []
                if formats.input_format.is_block_float()
                else [TRANSPOSE_BLOCK_BACK_TO_BACK()]
            ),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            UNPACK_TRANS_FACES(unpack_transpose_faces),
            UNPACK_TRANS_WITHIN_FACE(unpack_transpose_within_face),
            LOOP_FACTOR(32),
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
    )

    configuration.run(perf_report)
