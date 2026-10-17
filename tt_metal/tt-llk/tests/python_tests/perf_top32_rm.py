# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MATH_ISOLATE cycles per call of each DeepSeek top32_rm SFPU entry point (Blackhole only).

One "tile" here is one call of the selected kernel (see sources/top32_rm_perf.cpp for the
list); TILE_LOOP / (loop_factor * tile_cnt) is cycles per call. Correctness of the same
kernels is test_top32_rm.py.
"""

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT, TOP32_RM_PERF

pytestmark = [skip_for_wormhole, skip_for_quasar]


@pytest.mark.perf
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    dest_acc=[DestAccumulation.Yes],
    top32_perf_kernel=list(TOP32_RM_PERF.KERNELS),
    loop_factor=[16],
)
def test_perf_top32_rm(perf_report, formats, dest_acc, top32_perf_kernel, loop_factor):
    tile_count = 1

    configuration = PerfConfig(
        "sources/top32_rm_perf.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[TOP32_RM_PERF(top32_perf_kernel=top32_perf_kernel)],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(loop_factor)],
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
        disable_format_inference=True,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
