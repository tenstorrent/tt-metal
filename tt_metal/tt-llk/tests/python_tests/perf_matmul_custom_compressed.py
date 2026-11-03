# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.compressed_utils import (
    DEEPSEEK_T420,
    assign_clustered,
    assign_interleaved,
    assign_random,
    generate_exact_assignment,
)
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_matmul_custom_compressed import (
    COMPRESSION_GRANULARITY,
    DEEPSEEK_SHAPES,
    LAYOUT_SHAPES,
    LAYOUTS,
    MULTI_CALL_CASES,
    MULTI_FORMATS,
    SHAPES,
    SINGLE_FORMATS,
    WIDE_CASES,
    run_tile_compressed,
)
from test_matmul_custom_compressed import (
    test_matmul_custom_compressed_multi_call as _run_matmul_custom_compressed_multi_call,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf, compression_case):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        compression_case=compression_case,
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_compressed_single(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"single:{','.join(formats)}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_compressed_random(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"random:{','.join(formats)}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_compressed_clustered(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"clustered:{','.join(formats)}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    interleave_n=[1, 2, 4, 8, 16, 32],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_compressed_interleaved(
    perf_report, shape, formats, interleave_n, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_interleaved(
        K, N, formats, COMPRESSION_GRANULARITY, interleave_n
    )
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"interleaved:{','.join(formats)}:n={interleave_n}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    shape=DEEPSEEK_SHAPES,
    seed=[0],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_compressed_deepseek(
    perf_report, shape, seed, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = generate_exact_assignment(K, N, DEEPSEEK_T420, seed=seed)
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"deepseek:seed={seed}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    shape=LAYOUT_SHAPES,
    formats=[("bfp4", "bfp2", "bfp0")],
    dest_layout=LAYOUTS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_custom_compressed_layouts(
    perf_report, shape, formats, dest_layout, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        dest_layout=dest_layout,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"layouts:{','.join(formats)}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize(
    "M,K,N,dest_layout,dest_sync",
    [pytest.param(*case[1:], id=case[0]) for case in WIDE_CASES],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_matmul_custom_compressed_wide(
    perf_report, M, K, N, dest_layout, dest_sync, run_types, loop_factor, is_perf
):
    assignment = assign_random(K, N, ("bfp4", "bfp2", "bfp0"), COMPRESSION_GRANULARITY)
    run_tile_compressed(
        M,
        K,
        N,
        assignment,
        dest_layout=dest_layout,
        dest_sync=dest_sync,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            "wide:random:bfp4,bfp2,bfp0",
        ),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize(
    "M,K,N,pattern,rearm",
    [pytest.param(*case[1:], id=case[0]) for case in MULTI_CALL_CASES],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_matmul_custom_compressed_multi_call(
    perf_report, M, K, N, pattern, rearm, run_types, loop_factor, is_perf
):
    _run_matmul_custom_compressed_multi_call(
        M,
        K,
        N,
        pattern,
        rearm,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"multi:{','.join(pattern)}:rearm={rearm}",
        ),
    )
