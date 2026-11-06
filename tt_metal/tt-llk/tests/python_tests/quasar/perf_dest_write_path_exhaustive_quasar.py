# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.param_config import parametrize
from quasar.test_dest_write_path_quasar import (
    FORMAT_SWEEP,
    PERF_LOOP_FACTOR,
    PERF_RUN_TYPES,
    SFPU_TILES_PER_BANK_SWEEP,
    TILES_PER_BANK_SWEEP,
    WRITER_ONLY_FORMATS,
    DestWriteKernel,
    DestWritePath,
    SrcsIssueOpt,
    run_dest_write_path_eltwise,
    writer_only_srcs_opt,
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=list(DestWritePath),
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    data_format=[DataFormat.Float32, DataFormat.Int32],
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_write_path_copy_quasar(
    perf_report, path, tiles_per_bank, data_format, run_types
):
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        data_format,
        DestWriteKernel.Copy,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=list(DestWritePath),
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    formats=FORMAT_SWEEP,
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_write_path_format_sweep_quasar(
    perf_report, path, tiles_per_bank, formats, run_types
):
    l1_format, out_format = formats
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        l1_format,
        DestWriteKernel.Copy,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
        out_format=out_format,
        stamp_sync=False,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=[DestWritePath.B_DRAIN, DestWritePath.B_DONE],
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    formats=FORMAT_SWEEP,
    srcs_opt=[SrcsIssueOpt.BOTH],
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_write_path_format_srcs_opt_quasar(
    perf_report, path, tiles_per_bank, formats, srcs_opt, run_types
):
    l1_format, out_format = formats
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        l1_format,
        DestWriteKernel.Copy,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
        out_format=out_format,
        stamp_sync=False,
        srcs_opt=srcs_opt,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=[DestWritePath.B_DRAIN, DestWritePath.B_DONE],
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    srcs_opt=list(SrcsIssueOpt),
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_write_path_srcs_opt_quasar(
    perf_report, path, tiles_per_bank, srcs_opt, run_types
):
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        DataFormat.Float32,
        DestWriteKernel.Copy,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
        stamp_sync=False,
        srcs_opt=srcs_opt,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    path=list(DestWritePath),
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    formats=WRITER_ONLY_FORMATS,
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_write_path_writer_only_quasar(
    perf_report, path, tiles_per_bank, formats, run_types
):
    l1_format, out_format = formats
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        l1_format,
        DestWriteKernel.Copy,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
        out_format=out_format,
        stamp_sync=False,
        srcs_opt=writer_only_srcs_opt(path),
        writer_only=True,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    kernel=[DestWriteKernel.Square, DestWriteKernel.Exp],
    path=list(DestWritePath),
    tiles_per_bank=SFPU_TILES_PER_BANK_SWEEP,
    run_types=PERF_RUN_TYPES,
)
def test_perf_dest_write_path_sfpu_quasar(
    perf_report, kernel, path, tiles_per_bank, run_types
):
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        DataFormat.Float32,
        kernel,
        run_types=run_types,
        loop_factor=PERF_LOOP_FACTOR,
        is_perf=True,
        perf_report=perf_report,
    )
