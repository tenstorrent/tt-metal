# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import PerfRunType
from helpers.param_config import generate_perf_input_dimensions, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import CLAMPED_SILU_PARAMS, TILE_COUNT
from test_sfpu_clamped_silu import FORMATS, GATE, _valid_dest_acc


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    dest_acc=lambda formats: _valid_dest_acc(formats),
    input_dimensions=lambda dest_acc: generate_perf_input_dimensions(dest_acc),
    op=[GATE],
)
def test_perf_sfpu_clamped_silu(perf_report, formats, dest_acc, input_dimensions, op):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=StimuliSpec.uniform(low=-10.0, high=10.0, seed=0),
    )

    configuration = PerfConfig(
        "sources/sfpu_clamped_silu_test.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            CLAMPED_SILU_PARAMS(clamped_silu_op=op, scalar0=7.0, scalar1=1.702),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    configuration.run(perf_report)
