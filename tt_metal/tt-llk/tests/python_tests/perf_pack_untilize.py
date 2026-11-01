# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_untilize_common import (
    FAST_UNTILIZE_CT_DIMS,
    FAST_UNTILIZE_RT_DIMS,
    fast_untilize_dest_acc_modes,
    fast_untilize_formats,
)
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import dest_acc_for
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestSync,
    PerfRunType,
)
from helpers.param_config import (
    input_output_formats,
    max_dest_tiles,
    parametrize,
)
from test_pack_untilize import test_pack_untilize as run_pack_untilize

STANDARD_BENCHMARK = "standard"
FAST_UNTILIZE_BASELINE = "fast_untilize_baseline"

STANDARD_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Bfp8_b,
        DataFormat.Fp8_e4m3,
    ]
)
STANDARD_RUN_TYPES = [
    PerfRunType.L1_TO_L1,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]


def pack_untilize_formats(benchmark):
    return (
        STANDARD_FORMATS if benchmark == STANDARD_BENCHMARK else fast_untilize_formats()
    )


def pack_untilize_dest_acc_modes(benchmark, formats):
    if benchmark == STANDARD_BENCHMARK:
        return [dest_acc_for(formats.input_format)]
    return fast_untilize_dest_acc_modes(formats)


def pack_untilize_rt_dims(benchmark):
    if benchmark == STANDARD_BENCHMARK:
        return list(range(1, 9))
    return FAST_UNTILIZE_RT_DIMS


def pack_untilize_ct_dims(benchmark):
    if benchmark == STANDARD_BENCHMARK:
        return list(range(1, 9))
    return FAST_UNTILIZE_CT_DIMS


def pack_untilize_run_types(benchmark):
    # Keep one run-type schema for both benchmark families. Otherwise an
    # unfiltered local run cannot merge their worker CSVs.
    return [STANDARD_RUN_TYPES]


def pack_untilize_loop_factors(benchmark, formats, dest_acc, full_rt_dim, full_ct_dim):
    if benchmark == STANDARD_BENCHMARK:
        return [32]
    return [16, 32, 64]


def _largest_divisor_at_most(value, cap):
    for candidate in range(min(value, cap), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def pack_untilize_block_ct_dim(benchmark, formats, dest_acc, full_ct_dim):
    if benchmark == FAST_UNTILIZE_BASELINE:
        return _largest_divisor_at_most(full_ct_dim, max_dest_tiles(dest_acc))

    max_block_dim = 4 if formats.input_format.is_32_bit() else 8
    if (
        formats.input_format
        in (
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        )
        and formats.output_format == DataFormat.Float16
    ):
        max_block_dim = 4

    return _largest_divisor_at_most(full_ct_dim, max_block_dim)


@pytest.mark.perf
@parametrize(
    benchmark=[STANDARD_BENCHMARK, FAST_UNTILIZE_BASELINE],
    formats=pack_untilize_formats,
    dest_acc=pack_untilize_dest_acc_modes,
    full_rt_dim=pack_untilize_rt_dims,
    full_ct_dim=pack_untilize_ct_dims,
    run_types=pack_untilize_run_types,
    loop_factor=pack_untilize_loop_factors,
    compile_time_formats=lambda benchmark: [benchmark == FAST_UNTILIZE_BASELINE],
    is_perf=[True],
)
def test_perf_pack_untilize(
    perf_report,
    benchmark,
    formats,
    dest_acc,
    full_rt_dim,
    full_ct_dim,
    run_types,
    loop_factor,
    compile_time_formats,
    is_perf,
):
    if (
        benchmark == FAST_UNTILIZE_BASELINE
        and get_chip_architecture() != ChipArchitecture.BLACKHOLE
    ):
        pytest.skip("Fast-untilize baseline comparison is Blackhole-only")

    block_ct_dim = pack_untilize_block_ct_dim(benchmark, formats, dest_acc, full_ct_dim)
    dimensions = [full_rt_dim * 32, full_ct_dim * 32]

    run_pack_untilize(
        formats,
        dest_acc,
        dimensions,
        DestSync.Half,
        tile_dst_ct_offset=0,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        block_ct_dim=block_ct_dim,
        compile_time_formats=compile_time_formats,
        benchmark=benchmark,
    )
