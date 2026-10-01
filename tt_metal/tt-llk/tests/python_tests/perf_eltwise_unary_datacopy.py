# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.llk_params import Tilize
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_unary_datacopy import (
    DATACOPY_SUB_BYTE_SWEEP,
    DATACOPY_SWEEP,
    _run_unary_datacopy_test,
    get_valid_dest_accumulation_modes,
)

# The same-format pairs plus the converting pairs the pack side prices differently.
DATACOPY_BLOCK_PERF_FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
        DataFormat.Bfp8_b,
        DataFormat.Fp8_e4m3,
    ],
    same=True,
) + [
    fmt
    for fmt in input_output_formats([DataFormat.Bfp8_b, DataFormat.Float16_b, DataFormat.Float32])
    if fmt.input_format != fmt.output_format
]


@pytest.mark.perf
@parametrize(
    **DATACOPY_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy(
    perf_report,
    formats,
    dest_acc,
    num_faces,
    tilize,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_unary_datacopy_test(
        formats,
        dest_acc,
        num_faces,
        tilize,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@parametrize(
    formats=DATACOPY_BLOCK_PERF_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    num_faces=[1, 2, 4],
    tilize=Tilize.No,
    input_dimensions=[[128, 256]],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_block(
    perf_report,
    formats,
    dest_acc,
    num_faces,
    tilize,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_unary_datacopy_test(
        formats,
        dest_acc,
        num_faces,
        tilize,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        unpack_block=True,
    )


@pytest.mark.perf
@parametrize(
    **DATACOPY_SUB_BYTE_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_sub_byte_bfp(
    perf_report,
    formats,
    dest_acc,
    num_faces,
    tilize,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_unary_datacopy_test(
        formats,
        dest_acc,
        num_faces,
        tilize,
        input_dimensions,
        quantize_golden_input=True,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
