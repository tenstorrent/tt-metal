# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from helpers.test_variant_parameters import CUSTOM_MM_CALLS
from test_custom_mm import (
    COMPACT_CT_DIMS_16BIT,
    COMPACT_CT_DIMS_32BIT,
    COMPACT_SYNC_FULL_CT_DIMS,
    COMPACT_WIDE_BFP_CASES,
    CT_DIMS_16BIT,
    CT_DIMS_32BIT,
    KT_DIMS,
    LONG_K_CASES,
    MULTI_CALL_CASES,
    NARROW_PACK_SRC_FORMATS,
    ODD_K_CASES,
    SPLIT_ACC_LAYOUTS,
    SUPPORTED_M,
    _run_compact_against_default,
    _run_custom_mm,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    M=list(SUPPORTED_M),
    kt=KT_DIMS,
    ct=CT_DIMS_16BIT,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    M=list(SUPPORTED_M),
    kt=KT_DIMS,
    ct=CT_DIMS_32BIT,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_fp32_dest(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.Yes,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=NARROW_PACK_SRC_FORMATS,
    M=[1, 8],
    kt=KT_DIMS,
    ct=[1, 4, 7],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_fp32_dest_narrow_pack_src(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.Yes,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    layout=SPLIT_ACC_LAYOUTS,
    M=[1, 8],
    kt=[1, 2, 9],
    ct=[1, 3, 7],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_split_acc(
    perf_report, formats, layout, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        layout=layout,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(layout=SPLIT_ACC_LAYOUTS, ct=[1, 2], **PERF_MEASUREMENT_AXES)
def test_perf_custom_mm_split_acc_multi_call(
    perf_report, layout, ct, run_types, loop_factor, is_perf
):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    _run_custom_mm(
        8,
        8,
        ct,
        formats,
        DestAccumulation.No,
        CUSTOM_MM_CALLS(num_calls=4),
        layout=layout,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", LONG_K_CASES)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_custom_mm_long_k(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", ODD_K_CASES)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_custom_mm_odd_k(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    M=[1, 8],
    kt=[1, 3],
    ct=COMPACT_CT_DIMS_16BIT,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_compact(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_compact_against_default(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    M=[1, 8],
    kt=[2],
    ct=COMPACT_CT_DIMS_32BIT,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_compact_fp32_dest(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_compact_against_default(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.Yes,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=NARROW_PACK_SRC_FORMATS,
    M=[1, 8],
    kt=[2],
    ct=COMPACT_CT_DIMS_32BIT,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_compact_fp32_dest_narrow_pack_src(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_compact_against_default(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.Yes,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    M=[1, 8],
    kt=[2],
    ct=COMPACT_SYNC_FULL_CT_DIMS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_compact_sync_full(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_compact_against_default(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        dest_sync=DestSync.Full,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize("M,kt,ct,formats", COMPACT_WIDE_BFP_CASES)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_custom_mm_compact_wide_bfp(
    perf_report, formats, M, kt, ct, run_types, loop_factor, is_perf
):
    _run_compact_against_default(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize(
    "M,kt,ct,num_calls,in1_format",
    [pytest.param(*case[1:], id=case[0]) for case in MULTI_CALL_CASES],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_custom_mm_multi_call(
    perf_report, M, kt, ct, num_calls, in1_format, run_types, loop_factor, is_perf
):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b, in1_format)
    _run_custom_mm(
        M,
        kt,
        ct,
        formats,
        DestAccumulation.No,
        CUSTOM_MM_CALLS(num_calls=num_calls),
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
