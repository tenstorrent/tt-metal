# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MATH_ISOLATE perf sweep for the softmax_k SFPU entry (Blackhole only).

`_softmax_k_<k, is_fp32_dest_acc_en>` is a single-shot kernel over one 4-row band of
face 0 (see test_sfpu_softmax_k.py), so TILE_LOOP here is the cost of one call, not a
32-iteration vector loop: one SFPU instruction removed from the body is ~1 cycle.

Variants: bf16 DEST with an even k (no odd-tail fix-up) and an odd k
(`_zero_paired_odd_tail_lane_` runs), and a 32-bit DEST at k=16 (the only k the
32-bit DEST supports). Every even k compiles to the same body, as does every odd k.
"""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, PerfRunType
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    SOFTMAX_K,
    TILE_COUNT,
    generate_input_dim,
)

# (input/output format, dest_acc) -> the k values measured with it.
VARIANTS = {
    (DataFormat.Float16_b, DestAccumulation.No): (7, 8),
    (DataFormat.Float32, DestAccumulation.Yes): (16,),
}


@skip_for_wormhole
@pytest.mark.perf
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=[7, 8, 16],
    loop_factor=[10, 50, 100, 200],
)
def test_perf_sfpu_softmax_k(perf_report, formats, dest_acc, k, loop_factor):
    if k not in VARIANTS.get((formats.input_format, dest_acc), ()):
        pytest.skip("not a measured softmax_k variant")

    input_dimensions = [32, 32]
    tile_count = 1

    configuration = PerfConfig(
        "sources/sfpu_softmax_k_perf.cpp",
        formats,
        run_types=[
            PerfRunType.MATH_ISOLATE,
        ],
        templates=[
            SOFTMAX_K(softmax_k=k),
            generate_input_dim(input_dimensions, input_dimensions),
        ],
        runtimes=[
            TILE_COUNT(tile_count),
            LOOP_FACTOR(loop_factor),
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
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        disable_format_inference=True,
        compile_time_formats=True,
    )

    configuration.run(perf_report)
