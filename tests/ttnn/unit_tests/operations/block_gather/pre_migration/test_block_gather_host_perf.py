# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of block_gather: native C++ op versus the frozen generic_op route."""

import json
import os

import block_gather_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import torch
import ttnn
import ttnn.bringup.block_select_ttnn as op_pkg

from block_gather_pm_ref import bits_equal, make_ids, make_scores, mismatch, ref_block_gather, to_dev, to_host

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

# (lead dims, W, K, block, kwargs); taken from the acceptance suite / golden cut
CASES = [
    pytest.param((1, 1, 33), 512, 16, 2, {}, id="small_k16_b2"),
    pytest.param((1, 1, 96), 65536, 512, 8, {}, id="large_w65536_k512_b8"),
    pytest.param((1, 1, 50), 8192, 128, 32, {"output_tensor": True}, id="b32_output_tensor"),
    pytest.param((1, 1, 100), 4096, 64, 8, {"max_grid": (2, 3)}, id="max_grid_2x3"),
]


@pytest.mark.parametrize(("lead", "W", "K", "B", "kwargs"), CASES)
def test_host_perf(device, lead, W, K, B, kwargs, request):
    g = torch.Generator().manual_seed(0)
    s = make_scores((*lead, W), g)
    ids = make_ids(lead, K, W // B, g)
    expected = ref_block_gather(s, ids, B)

    def bind_args():
        kw = {"block": B}
        if "max_grid" in kwargs:
            kw["max_grid"] = kwargs["max_grid"]
        if kwargs.get("output_tensor"):
            kw["output_tensor"] = to_dev(torch.zeros(expected.shape, dtype=torch.bfloat16), device, ttnn.bfloat16)
        return (to_dev(s, device, ttnn.bfloat16), to_dev(ids, device, ttnn.uint32)), kw

    def validate(out):
        (got,) = to_host(out)
        assert bits_equal(got, expected), mismatch(got, expected)

    def route(name, op):
        def bind(case, phase, block):
            args, kw = bind_args()
            return hc.BoundCall(op, validate, args, kw)

        return hc.RouteAdapter(name, bind)

    native = lambda *a, **k: op_pkg.block_gather(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", block_gather_generic.block_gather),
        route("native", native),
        hc.BenchmarkRuntime.from_ttnn(device),
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
