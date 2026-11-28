# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of ckv_bfp4_rows: native C++ op versus the frozen generic_op route."""
import json
import os

import intree_host_comparison as hc
import ckv_bfp4_rows_generic  # frozen generic route, provided by the driver
import pytest
import torch
import ttnn
from ttnn.bringup import ckv_bfp4_ttnn as op_pkg

from tests.ttnn.unit_tests.operations.ckv_bfp4_rows.pre_migration.ckv_rows_reference import (
    FILL,
    RECORD_BYTES,
    WIDTH,
    encode_rows,
    make_rows,
    nd_sharded_dram,
    table_host,
)

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

# (src shape, src layout, table shape, (src_row0, n, row0), kwargs): cases of the acceptance suite
CASES = [
    # smallest: one band (the suite's one_band_idle_cores)
    pytest.param((1, 1, 64, WIDTH), "interleaved", (1, 1, 64, RECORD_BYTES), (0, 32, 0), {}, id="one_band"),
    # device time matters: 256 bands, several a core (the suite's many_bands_per_core)
    pytest.param(
        (4, 1, 4096, WIDTH), "interleaved", (1, 1, 9216, RECORD_BYTES), (4096, 8192, 0), {}, id="many_bands_8192"
    ),
    # ND-sharded source (the suite's nd_sharded_16)
    pytest.param((4, 1, 1024, WIDTH), "nd16", (1, 1, 2048, RECORD_BYTES), (1056, 1280, 256), {}, id="nd_sharded_16"),
    # the optional max_grid (the suite's max_grid_2x2_uneven)
    pytest.param(
        (1, 1, 512, WIDTH),
        "interleaved",
        (1, 1, 512, RECORD_BYTES),
        (64, 224, 32),
        {"max_grid": (2, 2)},
        id="max_grid_2x2",
    ),
]


@pytest.mark.parametrize(("shape", "layout", "table_shape", "call", "kwargs"), CASES)
def test_host_perf(device, shape, layout, table_shape, call, kwargs, request):
    torch.manual_seed(0)
    total = int(torch.Size(shape).numel()) // WIDTH
    x = make_rows(total, seed=7)
    src_row0, n, row0 = call
    expected = encode_rows(x[src_row0 : src_row0 + n])
    mem = ttnn.DRAM_MEMORY_CONFIG if layout == "interleaved" else nd_sharded_dram(device, int(layout[2:]))

    def make_input():
        return ttnn.from_torch(
            x.reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
        )

    def make_table():
        return ttnn.from_torch(
            torch.full(table_shape, FILL, dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def bind(op):
        table = make_table()  # a fresh table (the op's in-place output) for every bound call

        def validate(out):
            assert out is None or out.buffer_address() == table.buffer_address()
            for got in table_host(table):
                assert torch.equal(got[row0 : row0 + n], expected), "records differ from the host rule"
                assert bool((got[:row0] == FILL).all()) and bool((got[row0 + n :] == FILL).all())

        return hc.BoundCall(op, validate, (make_input(), table, src_row0, n, row0), dict(kwargs))

    def route(name, op):
        return hc.RouteAdapter(name, lambda case, phase, block: bind(op))

    native = lambda *a, **k: op_pkg.ckv_bfp4_rows(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", ckv_bfp4_rows_generic.ckv_bfp4_rows),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
