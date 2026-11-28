# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of pack_fp8_rows: native C++ op versus the frozen generic_op route."""
import importlib
import json
import os

import pytest
import torch
import ttnn
import intree_host_comparison as hc
import pack_fp8_rows_generic  # frozen generic route, provided by the driver

import pack_fp8_rows_pre_migration_helpers as H

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))
PUBLIC_MODULE = "ttnn.bringup.fp8_rows_ttnn"

CASES = [
    pytest.param([1, 1, 1, 128], False, {}, id="m1_single_row"),
    pytest.param([1, 1, 4100, 128], False, {}, id="m4100_blocks_exceed_cores"),
    pytest.param([1, 1, 320, 128], True, {"max_grid": (2, 3)}, id="m320_caller_out_grid2x3"),
]


@pytest.mark.parametrize(("shape", "caller_out", "kwargs"), CASES)
def test_host_perf(device, shape, caller_out, kwargs, request):
    vals, scales = H.make_case_bytes(shape, seed=sum(shape))
    expected = H.reference_rows(vals, scales.contiguous().view(torch.uint8)).reshape(1, H.rows_of(shape), H.ROW_BYTES)

    def make_inputs():
        return (H.to_device_e4m3(vals, device), H.to_device_scale(scales, device))

    def make_kwargs():
        k = dict(kwargs)
        if caller_out:
            k["output_tensor"] = H.preallocated_output(shape, device, fill=0xA5)
        return k

    def validate(out):
        H.assert_rows_equal(H.output_rows(out), expected)

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda case, phase, block: hc.BoundCall(op, validate, make_inputs(), make_kwargs())
        )

    native = lambda *a, **k: getattr(importlib.import_module(PUBLIC_MODULE), "pack_fp8_rows")(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", pack_fp8_rows_generic.pack_fp8_rows),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
