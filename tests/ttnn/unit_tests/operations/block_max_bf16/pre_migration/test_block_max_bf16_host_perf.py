# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of block_max_bf16: native C++ op versus the frozen generic_op route."""

import json
import os

import intree_host_comparison as hc
import block_max_bf16_generic  # frozen generic route, provided by the driver
import pytest
import torch
import ttnn
import ttnn.bringup.block_select_ttnn as op_pkg

from tests.ttnn.unit_tests.operations.block_max_bf16.pre_migration.block_max_ref import (
    SENTINEL_FILL,
    bits,
    fold_input,
    fold_ref,
    strip_device_view,
    strip_input,
    strip_ref,
    up_n,
    written,
)

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

# (mode, geometry, kwargs) from the acceptance suite / golden cut:
#   fold geometry (shape, factor); strip geometry (S, NB). kwargs: n (a value) or n_tensor (a device tensor's value).
CASES = [
    pytest.param("fold", ([1, 1, 1, 64], 1), {"n": 64}, id="fold_min_r1_w64"),
    pytest.param("fold", ([1, 1, 96, 8192], 4), {"n_tensor": 1000}, id="fold_r96_w8192_f4_ntensor"),
    pytest.param("strip", (8, 16384), {"n": 16384}, id="strip_s8_nb16384"),
]


@pytest.mark.parametrize(("mode", "geom", "kwargs"), CASES)
def test_host_perf(device, mode, geom, kwargs, request):
    torch.manual_seed(0)
    if mode == "fold":
        shape, F = geom
        x = fold_input(shape, seed=0)
        expected = fold_ref(x, F)
        x_dev, x_dtype, extra = x, ttnn.float32, {"factor": F}
    else:
        S, NB = geom
        x = strip_input([1, 1], S, NB, seed=0)
        expected = strip_ref(x, S, NB)
        x_dev, x_dtype, extra = strip_device_view(x, S, NB), ttnn.bfloat16, {"strip_major": True}
    n = kwargs.get("n", kwargs.get("n_tensor"))
    m = written(n, expected.shape[-1])
    prev = torch.full(list(expected.shape), SENTINEL_FILL, dtype=torch.bfloat16)

    def make_input():
        return ttnn.from_torch(
            x_dev, dtype=x_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def make_kwargs():
        kw = dict(extra)
        kw["output_tensor"] = ttnn.from_torch(
            prev,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if "n" in kwargs:
            kw["n"] = kwargs["n"]
        else:
            kw["n_tensor"] = up_n(kwargs["n_tensor"], device)
        return kw

    def validate(out):
        d = ttnn.to_torch(out).reshape(expected.shape)
        assert torch.equal(bits(d[..., :m]), bits(expected[..., :m]))
        assert torch.equal(bits(d[..., m:]), bits(prev[..., m:]))

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda case, phase, block: hc.BoundCall(op, validate, (make_input(),), make_kwargs())
        )

    native = lambda *a, **k: op_pkg.block_max_bf16(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", block_max_bf16_generic.block_max_bf16),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
