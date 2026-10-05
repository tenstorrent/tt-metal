# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.perf import contract
from tests.perf.unit.expect import expect_error  # noqa: F401

DECL = {"perf.metric.IterationTime": "unit=s;better=lower;aggregate=min"}


def _write(tmp_path, rows, context=DECL, name="out.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"context": context, "benchmarks": rows}))
    return path


def _row(name, value, metric="IterationTime", **extra):
    return {"name": name, "run_name": name, "run_type": "iteration", metric: value, **extra}


def test_iteration_rows_use_declared_aggregation(tmp_path):
    path = _write(tmp_path, [_row("BM/a/kernel_size:256", 2.0), _row("BM/a/kernel_size:256", 1.5)])
    run = contract.read([path])
    assert contract.aggregate(run) == {"BM/a/kernel_size:256": {"IterationTime": 1.5}}
    assert contract.repetitions(run) == 2


def test_aggregate_only_output_uses_matching_aggregate_row(tmp_path):
    context = {"perf.metric.bytes_per_second": "unit=B/s;better=higher;aggregate=median"}
    rows = [
        {
            "name": "Read/size:1_median",
            "run_name": "Read/size:1",
            "run_type": "aggregate",
            "aggregate_name": "median",
            "repetitions": 5,
            "bytes_per_second": 10.0,
        },
        {
            "name": "Read/size:1_mean",
            "run_name": "Read/size:1",
            "run_type": "aggregate",
            "aggregate_name": "mean",
            "repetitions": 5,
            "bytes_per_second": 99.0,
        },
    ]
    run = contract.read([_write(tmp_path, rows, context)])
    assert contract.aggregate(run) == {"Read/size:1": {"bytes_per_second": 10.0}}
    assert contract.repetitions(run) == 5


def test_separate_process_outputs_are_merged_as_repetitions(tmp_path):
    paths = [_write(tmp_path, [_row("compile/num_kernels:300", v)], name=f"{i}.json") for i, v in enumerate([3, 1, 2])]
    run = contract.read(paths)
    assert contract.aggregate(run) == {"compile/num_kernels:300": {"IterationTime": 1.0}}
    assert contract.repetitions(run) == 3


def test_errors_and_context_are_collected(tmp_path):
    rows = [
        {"name": "BM/bad", "run_name": "BM/bad", "error_occurred": True, "error_message": "TT_FATAL"},
        _row("BM/good", 1.0, ctx_aiclk_mhz=1000),
        _row("BM/good2", 1.0, ctx_aiclk_mhz=800),
    ]
    run = contract.read([_write(tmp_path, rows, {**DECL, "perf.context.iommu": "on"})])
    assert run.errors == {"BM/bad": "TT_FATAL"}
    assert contract.run_context(run) == {"iommu": "on", "aiclk_mhz": "800..1000"}


@pytest.mark.parametrize(
    "context, rows, message",
    [
        ({}, [_row("BM/a", 1.0)], "declares no perf.metric"),
        (DECL, [_row("BM/a", 1.0, metric="Other")], "reports none of the declared metrics"),
        (DECL, [_row("BM/a", float("nan"))], "invalid IterationTime"),
        (DECL, [_row("BM/a", 0)], "invalid IterationTime"),
        ({"perf.metric.x": "unit=s;better=up;aggregate=min"}, [], "better='up'"),
    ],
)
def test_contract_violations_are_rejected(expect_error, tmp_path, context, rows, message):
    with expect_error(contract.ContractError, message):
        contract.read([_write(tmp_path, rows, context)])


def test_group_is_the_prefix_before_swept_arguments():
    assert contract.group_of("BM_pgm_dispatch/brisc_only_trace/kernel_size:256/manual_time") == (
        "BM_pgm_dispatch/brisc_only_trace"
    )
    assert contract.group_of("BM_pgm_dispatch/all_processors_no_trace/manual_time") == (
        "BM_pgm_dispatch/all_processors_no_trace"
    )
    assert contract.group_of("Read/page_size:32/size:64/type:0/device:0/real_time") == "Read"
