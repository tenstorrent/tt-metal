# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Final round shadow driver (not for merging): the production forms of the Int32 column SUM, L1_TO_L1.
ttnn form (sources/sfpu_reduce_ttnn_col_perf_x.cpp): ht tiles per column, chunk columns per DEST section, add_int
fold, one sfpu_reduce per column. tt-mlir D2M form (sources/sfpu_reduce_mlir_col_perf_x.cpp): accumulator copy, then
per input tile copy, sfpu_reduce, add_int. One loop is one DEST section.
"""

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
    ReducePool,
)
from helpers.param_config import input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)

TILE_DIM = 32
RING = 16
FORMATS = input_output_formats([DataFormat.Int32], same=True)[0]


def _run(perf_report, source, ht, chunk, n_in, loop_factor):
    dims = [TILE_DIM * ht, TILE_DIM * chunk]
    n_stim = min(n_in, RING)
    configuration = PerfConfig(
        source,
        FORMATS,
        run_types=[PerfRunType.L1_TO_L1],
        templates=[
            MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=ReducePool.Sum),
            APPROX_MODE(ApproximationMode.No),
            generate_input_dim(dims, dims),
        ],
        runtimes=[TILE_COUNT(n_in), LOOP_FACTOR(loop_factor)],
        variant_stimuli=StimuliConfig(
            None,
            FORMATS.input_format,
            None,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=n_stim,
            tile_count_B=n_stim,
            tile_count_res=n_stim,
        ),
        unpack_to_dest=True,
        dest_acc=DestAccumulation.Yes,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    configuration.run(perf_report)


@pytest.mark.perf
@pytest.mark.parametrize("loop_factor", [1, 16])
@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize("ht", [1, 2, 4, 16, 64])
def test_perf_ttnn_col_sum(perf_report, ht, chunk, loop_factor):
    _run(perf_report, "sources/sfpu_reduce_ttnn_col_perf_x.cpp", ht, chunk, ht * chunk, loop_factor)


@pytest.mark.perf
@pytest.mark.parametrize("loop_factor", [1, 16])
@pytest.mark.parametrize("ht", [1, 2, 4, 5])
def test_perf_mlir_col_sum(perf_report, ht, loop_factor):
    _run(perf_report, "sources/sfpu_reduce_mlir_col_perf_x.cpp", ht, 1, 1 + ht, loop_factor)


@pytest.mark.perf
@pytest.mark.parametrize("loop_factor", [1, 16])
@pytest.mark.parametrize("chunk", [1, 2, 3])
@pytest.mark.parametrize("ht", [1, 2, 4, 16, 64])
def test_perf_ttnn_col_sum_h(perf_report, ht, chunk, loop_factor):
    _run(perf_report, "sources/sfpu_reduce_ttnn_col_perf_hx.cpp", ht, chunk, ht * chunk, loop_factor)
