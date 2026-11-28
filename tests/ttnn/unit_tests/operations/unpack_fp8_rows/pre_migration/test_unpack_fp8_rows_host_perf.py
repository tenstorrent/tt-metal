# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of unpack_fp8_rows: native C++ op versus the frozen generic_op route."""

import json
import os

import pytest
import torch
import ttnn
import intree_host_comparison as hc
import unpack_fp8_rows_generic  # frozen generic route, provided by the driver
import ttnn.bringup.fp8_rows_ttnn as op_pkg

import unpack_fp8_rows_pre_migration_helpers as H

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

# (rows shape a chip, rows memory, out shape, kwargs): cases of the acceptance suite
CASES = [
    pytest.param((1, 1, 32, H.ROW), "dram", (1, 1, 32, H.VALUES), dict(row_lo=0, row_hi=32), id="one_band_min"),
    pytest.param(
        (3, 1, 512, H.ROW),
        "nd16",
        (1, 1, 640, H.VALUES),
        dict(row_lo=64, row_hi=448, batch=1, out_row0=128, max_grid="cap"),
        id="nd_offset_batch_outrow",
    ),
    pytest.param(
        (1, 1, 8192, H.ROW),
        "nd16",
        (1, 1, 8192, H.VALUES),
        dict(row_lo=0, row_hi=8192, max_grid="cap"),
        id="many_bands_8192",
    ),
    pytest.param(
        (1, 1, 2048, H.ROW),
        "dram",
        (1, 1, 2048, H.VALUES),
        dict(row_lo=512, row_hi=1696, max_grid="cap", whole_grid=True, common_values=True),
        id="whole_grid_common_values",
    ),
]


@pytest.mark.parametrize(("rows_shape", "mem", "out_shape", "kwargs"), CASES)
def test_host_perf(device, rows_shape, mem, out_shape, kwargs, request):
    torch.manual_seed(0)
    kwargs = dict(kwargs)
    if kwargs.get("max_grid") == "cap":
        kwargs["max_grid"] = H.grid_cap(device)
    rows_host = H.make_rows(1, rows_shape, 0)
    prior = H.fill(1, out_shape, 0)
    lo, hi = kwargs["row_lo"], kwargs["row_hi"]
    expected = H.expected_out(prior, rows_host, lo, hi, kwargs.get("batch", 0), kwargs.get("out_row0", lo))
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else H.nd_config(device, int(mem[2:]), len(rows_shape))

    def make_input():
        return H.rows_to_device(device, rows_host, mc), H.out_to_device(device, prior)

    def validate(out):
        got = H.to_host(out)
        H.assert_exact(got, expected, "host perf output")

    def route(name, op):
        return hc.RouteAdapter(name, lambda case, phase, block: hc.BoundCall(op, validate, make_input(), dict(kwargs)))

    native = lambda *a, **k: op_pkg.unpack_fp8_rows(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", unpack_fp8_rows_generic.unpack_fp8_rows),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
