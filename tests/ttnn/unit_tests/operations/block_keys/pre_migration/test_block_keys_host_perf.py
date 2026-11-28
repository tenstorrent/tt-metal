# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host cold/hot timing of block_keys: native C++ op versus the frozen generic_op route."""

import json
import os

import block_keys_generic  # frozen generic route, provided by the driver
import intree_host_comparison as hc
import pytest
import torch
import ttnn
import ttnn.bringup.block_select_ttnn as op_pkg

BLOCKS = int(os.environ.get("BENCH_BLOCKS", "20"))
SENT = 0xFFFFFFFF

# (lead dims, K, KP, block, width, kwargs); "output_tensor": True binds a fresh caller's output buffer per call
CASES = [
    pytest.param((1, 1, 1), 16, 16, 8, 4096, {}, id="m1_single_row"),
    pytest.param((1, 1, 96), 512, 512, 8, 65536, {}, id="large_w65536_k512_b8"),
    pytest.param((1, 1, 320), 64, 96, 8, 4096, {"output_tensor": True, "max_grid": (2, 3)}, id="caller_out_grid2x3"),
]


def _u32(t):
    return t.to(torch.int64) & 0xFFFFFFFF


def ref_keys(pos, ids, B, width):
    pos, ids = _u32(pos), _u32(ids)
    K = ids.shape[-1]
    okp = pos < K * B
    blk = torch.gather(ids, -1, (pos.clamp(max=K * B - 1) // B))
    ok = okp & (blk < width // B)
    return torch.where(ok, blk * B + pos % B, torch.full_like(pos, SENT))


@pytest.mark.parametrize(("lead", "K", "KP", "B", "W", "kwargs"), CASES)
def test_host_perf(device, lead, K, KP, B, W, kwargs, request):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, W // B, (*lead, K), generator=g)
    ids.view(-1, K)[0, -1] = SENT
    pos = torch.randint(0, K * B, (*lead, KP), generator=g)
    pos.view(-1, KP)[-1, 0] = SENT
    expected = ref_keys(pos, ids, B, W)

    def up(t):
        return ttnn.from_torch(
            _u32(t).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def make_kwargs():
        kw = {"block": B, "width": W}
        if "max_grid" in kwargs:
            kw["max_grid"] = kwargs["max_grid"]
        if kwargs.get("output_tensor"):
            kw["output_tensor"] = up(torch.zeros(*lead, KP, dtype=torch.int64))
        return kw

    def validate(out):
        assert list(out.shape) == [*lead, KP] and out.dtype == ttnn.uint32
        assert torch.equal(_u32(ttnn.to_torch(out)), expected)

    def route(name, op):
        return hc.RouteAdapter(
            name, lambda case, phase, block: hc.BoundCall(op, validate, (up(pos), up(ids)), make_kwargs())
        )

    native = lambda *a, **k: op_pkg.block_keys(*a, **k)  # noqa: E731
    case_id = request.node.callspec.id
    result = hc.HostComparison(
        route("generic", block_keys_generic.block_keys), route("native", native), hc.BenchmarkRuntime.from_ttnn(device)
    ).run(case_id, None, blocks=BLOCKS)
    out = hc.evidence_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.as_dict(), indent=1, sort_keys=True))
