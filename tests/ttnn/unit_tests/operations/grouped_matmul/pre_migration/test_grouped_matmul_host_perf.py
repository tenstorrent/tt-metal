# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of grouped_matmul: native C++ op versus the frozen generic_op route."""

import json
import os

import grouped_matmul_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import torch
import ttnn
import ttnn.bringup.grouped_matmul_ttnn as op_pkg
from ttnn.bringup.grouped_matmul_ttnn import bank

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))
REL_TOL = 2e-2  # the acceptance suite's HiFi2 bound (vs fp64 of the device operands)

# (G, M, K, N, kwargs, extra): cases of the acceptance suite. extra: "out" passes a caller's output buffer, "bank" the
# bank-local weight layout (band, 10 cores a bank, one kernel group).
CASES = [
    pytest.param(1, 32, 32, 32, {}, None, id="min_one_tile"),
    pytest.param(
        3, 96, 512, 352, dict(dtype=ttnn.bfloat16, max_grid=(4, 3), block_w=8, depth=3), "out", id="uneven_4x3_out_buf"
    ),
    pytest.param(
        8, 160, 4096, 1024, dict(dtype=ttnn.bfloat8_b, max_grid=(11, 10), block_w=4, depth=4), None, id="wo_a_8x160"
    ),
    pytest.param(
        8,
        32,
        256,
        1024,
        dict(dtype=ttnn.bfloat8_b, block_w=4, depth=4, w_inflight=2, x_ahead=1, one_group=True),
        "bank",
        id="bank_band10_one_group",
    ),
]


@pytest.mark.parametrize(("G", "M", "K", "N", "kwargs", "extra"), CASES)
def test_host_perf(device, G, M, K, N, kwargs, extra, request):
    torch.manual_seed(0)
    x = torch.randn(1, G, M, K, dtype=torch.float32)
    w = torch.randn(1, G, K, N, dtype=torch.float32) / K**0.5
    lay = None
    if extra == "bank":
        lay = bank.plan(device, G, K, N, max_cores=110, cores_per_bank=10)
        assert lay is not None
    odt = kwargs.get("dtype", ttnn.bfloat8_b)

    def dev(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def chip0(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()

    expected = chip0(dev(x)) @ chip0(dev(w))

    def make_input():
        args = (dev(x), dev(w) if lay is None else bank.upload(w, lay, device))
        kw = dict(kwargs)
        if lay is not None:
            kw["w_layout"] = lay
        if extra == "out":
            kw["output_tensor"] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, G, M, N]), odt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
        return args, kw

    def validate(out):
        assert list(out.shape) == [1, G, M, N] and out.dtype == odt
        got = chip0(out)
        assert float((got - expected).norm() / expected.norm()) <= REL_TOL

    def route(name, op):
        def bind(case, phase, block):
            args, kw = make_input()
            return hc.BoundCall(op, validate, args, kw)

        return hc.RouteAdapter(name, bind)

    native = lambda *a, **k: op_pkg.grouped_matmul(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", grouped_matmul_generic.grouped_matmul),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
