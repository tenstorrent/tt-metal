# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.compressed_utils import (
    DEEPSEEK_T420,
    assign_clustered,
    assign_interleaved,
    assign_random,
    assign_tile_matched,
    generate_refined_face_assignment,
)
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_matmul_face_compressed import (
    COMPRESSION_GRANULARITY,
    DEEPSEEK_SHAPES,
    FACE_SWITCH_MULTS,
    MULTI_FORMATS,
    SHAPES,
    SINGLE_FORMATS,
    run_face_compressed,
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
def test_perf_matmul_face_compressed_single(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_face_compressed(
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
def test_perf_matmul_face_compressed_random(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_face_compressed(
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
def test_perf_matmul_face_compressed_clustered(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_face_compressed(
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
def test_perf_matmul_face_compressed_interleaved(
    perf_report, shape, formats, interleave_n, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_interleaved(
        K, N, formats, COMPRESSION_GRANULARITY, interleave_n
    )
    run_face_compressed(
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
@pytest.mark.nightly
@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_face_compressed_single_matched(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_tile_matched(assign_clustered, K, N, formats)
    run_face_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"single-matched:{','.join(formats)}",
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
def test_perf_matmul_face_compressed_random_matched(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_tile_matched(assign_random, K, N, formats)
    run_face_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"random-matched:{','.join(formats)}",
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
def test_perf_matmul_face_compressed_clustered_matched(
    perf_report, shape, formats, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_tile_matched(assign_clustered, K, N, formats)
    run_face_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"clustered-matched:{','.join(formats)}",
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
def test_perf_matmul_face_compressed_interleaved_matched(
    perf_report, shape, formats, interleave_n, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = assign_tile_matched(assign_interleaved, K, N, formats, interleave_n)
    run_face_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"interleaved-matched:{','.join(formats)}:n={interleave_n}",
        ),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    shape=DEEPSEEK_SHAPES,
    switch_mult=FACE_SWITCH_MULTS,
    seed=[0],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_matmul_face_compressed_deepseek(
    perf_report, shape, switch_mult, seed, run_types, loop_factor, is_perf
):
    M, K, N = shape
    assignment = generate_refined_face_assignment(
        K, N, DEEPSEEK_T420, switch_mult=switch_mult, seed=seed
    )
    run_face_compressed(
        M,
        K,
        N,
        assignment,
        **_perf_kwargs(
            perf_report,
            run_types,
            loop_factor,
            is_perf,
            f"deepseek:switch={switch_mult}:seed={seed}",
        ),
    )
