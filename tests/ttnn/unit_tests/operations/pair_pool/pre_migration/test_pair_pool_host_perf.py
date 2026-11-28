# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of pair_pool: native C++ op versus the frozen generic_op route."""

import json
import os

import intree_host_comparison as hc
import pair_pool_generic  # frozen generic route, provided by the driver
import pytest
import torch
import ttnn
from ttnn.bringup import pair_pool_ttnn as op_pkg

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))

# From the acceptance suite: the smallest input (one unit), the model's 640 x 1024 on the whole grid (device time
# matters), and the optional arguments set (a caller's output buffer, a 2x2 grid, an uneven split).
CASES = [
    pytest.param((64, 64), False, {}, id="min_one_unit"),
    pytest.param((1, 1, 640, 1024), False, {}, id="model_640x1024"),
    pytest.param((1, 1, 192, 192), True, {"max_grid": (2, 2)}, id="out_buf_uneven_2x2"),
]


def _reference(x):
    """float64: out[t] = kv[2t] w0 + kv[2t+1] w1, (w0, w1) = softmax(gate[2t], gate[2t+1]) per column."""
    x = x.double()
    hd = x.shape[-1] // 2
    kv0, g0, kv1, g1 = x[..., 0::2, :hd], x[..., 0::2, hd:], x[..., 1::2, :hd], x[..., 1::2, hd:]
    w = torch.softmax(torch.stack([g0, g1]), dim=0)
    return kv0 * w[0] + kv1 * w[1], kv0.abs() * w[0] + kv1.abs() * w[1]


@pytest.mark.parametrize(("shape", "out_buf", "kwargs"), CASES)
def test_host_perf(device, shape, out_buf, kwargs, request):
    torch.manual_seed(0)
    *lead, rows, width = shape
    kv = torch.randn(*lead, rows, width // 2)
    gate = torch.randn(*lead, rows, width // 2) * 3.0
    x = torch.cat([kv, gate], dim=-1).float().contiguous()
    expected, magnitude = _reference(x)
    out_shape = list(shape)
    out_shape[-2] //= 2
    out_shape[-1] //= 2

    def make_input():
        return ttnn.from_torch(
            x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def make_kwargs():
        k = dict(kwargs)
        if out_buf:
            k["output_tensor"] = ttnn.allocate_tensor_on_device(
                ttnn.Shape(out_shape), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
            )
        return k

    def validate(out):
        got = ttnn.to_torch(out).double()
        assert list(got.shape) == out_shape and out.dtype == ttnn.float32
        assert ((got - expected).norm() / expected.norm()).item() <= 1e-6
        assert ((got - expected).abs() <= 1e-5 * magnitude + 1e-12).all()

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda case, phase, block: hc.BoundCall(op, validate, (make_input(),), make_kwargs())
        )

    native = lambda *a, **k: op_pkg.pair_pool(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", pair_pool_generic.pair_pool), route("native", native), hc.BenchmarkRuntime.from_ttnn(device)
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
