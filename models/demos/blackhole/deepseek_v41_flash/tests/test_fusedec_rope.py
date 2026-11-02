# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSV41_ROPE_FULL_FUSE (tt/cmp_fuse.py::rope_full): bit equality and traced time against CompactRope.full (concat of a constant block + the 64 rotated columns, reshape, to_layout(TILE)),
all 4 layouts of the step state, cos and sin, T = 4 .. 16 rows, 32 chips. Env FD_OUT (json)."""

import json
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import cmp_fuse as CF

OUT = os.environ.get("FD_OUT", "/tmp/fusedec_rope")


def _up(md, t, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def old_full(md, row64, cos, T, width, shape, _cache={}):
    key = (T, width, cos)
    if (
        key not in _cache
    ):  # (host writes are not allowed inside a trace capture: the constant block is built on first use, in the eager comparison)
        _cache[key] = _up(md, torch.full((T, 1, width - 64), 1.0 if cos else 0.0).to(torch.bfloat16))
    const = _cache[key]
    full = ttnn.concat([const, row64], dim=-1)
    return ttnn.to_layout(ttnn.reshape(full, shape), ttnn.TILE_LAYOUT)


def traced_us(md, fn, k=8, reps=20):
    def cap(kk):
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(kk):
            fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        return tid

    def run(tid):
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        return (time.perf_counter() - t) / reps * 1e6

    t1, tk = cap(1), cap(k)
    a, b = run(t1), run(tk)
    return (b - a) / (k - 1)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}],
    indirect=True,
)
@torch.no_grad()
def test_rope_full(mesh_device):
    md = mesh_device
    res = {}
    g = torch.Generator().manual_seed(3)
    for T in (4, 8, 10, 12, 16):
        r_t = (torch.randn(T, 1, 64, generator=g) * 2).to(torch.bfloat16)
        row64 = _up(md, r_t)
        for width, shapes in (
            (512, ([1, T, 1, 512], [1, 1, T, 512])),
            (128, ([T, 1, 1, 128], [1, 1, T, 128])),
        ):
            for shape in shapes:
                for cos in (True, False):
                    a = old_full(md, row64, cos, T, width, shape)
                    b = CF.rope_full(row64, shape, width, cos)
                    ttnn.synchronize_device(md)
                    eq = all(
                        torch.equal(ttnn.to_torch(x).float(), ttnn.to_torch(y).float())
                        for x, y in zip(ttnn.get_device_tensors(a)[::7], ttnn.get_device_tensors(b)[::7])
                    )
                    key = f"T{T}_{'x'.join(map(str, shape))}_{'cos' if cos else 'sin'}"
                    rec = dict(eq=eq)
                    if cos:
                        rec["us_old"] = traced_us(md, lambda: old_full(md, row64, cos, T, width, shape))
                        rec["us_new"] = traced_us(md, lambda: CF.rope_full(row64, shape, width, cos))
                    res[key] = rec
                    print("ROPEFULL " + json.dumps({key: rec}), flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(res, open(OUT + ".json", "w"), indent=1)
    assert all(v["eq"] for v in res.values()), [k for k, v in res.items() if not v["eq"]]
