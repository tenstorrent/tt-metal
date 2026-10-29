# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark of the per-column argmax of the drafter sampler: fp32 row-major logits [1,1,U,16160] per device, ``ttnn.argmax`` default vs ``enable_secondary_dm`` vs a split of every row
into 8 sub-rows (more cores) + a second-level argmax. Checks the indices against torch (first maximum). Prints ARGMAX_BENCH <variant> <us per call>."""

import time

import pytest
import torch

import ttnn

V8 = 16160


def _split(rm, U, parts=8):
    sub = ttnn.reshape(rm, [1, 1, U * parts, V8 // parts])
    li = ttnn.argmax(sub, dim=-1, keepdim=True)  # [1,1,U*parts,1] local index in the sub-row
    mx = ttnn.gather(sub, 3, li)  # [1,1,U*parts,1]
    mx2 = ttnn.reshape(mx, [1, 1, U, parts])
    best = ttnn.argmax(mx2, dim=-1, keepdim=True)  # [1,1,U,1] first maximum
    li2 = ttnn.reshape(ttnn.typecast(li, ttnn.float32), [1, 1, U, parts])
    il = ttnn.gather(li2, 3, best)
    tok = ttnn.add(ttnn.multiply(ttnn.typecast(best, ttnn.float32), float(V8 // parts)), il)
    return ttnn.typecast(tok, ttnn.uint32)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 20_000_000}, id="plain")],
    indirect=True,
)
@pytest.mark.parametrize("U", [2, 4])
@torch.no_grad()
def test_argmax_bench(mesh_device, U):
    md = mesh_device
    torch.manual_seed(U)
    lg = torch.randn(1, 1, U, V8)
    lg[0, 0, :, 777] = lg[0, 0, :, 9000] = 40.0  # a tie: the lower index wins
    want = lg[0, 0].argmax(-1)
    rm = ttnn.from_torch(
        lg,
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    variants = {
        "default": lambda: ttnn.argmax(rm, dim=-1, keepdim=True),
        "secondary_dm": lambda: ttnn.argmax(rm, dim=-1, keepdim=True, enable_secondary_dm=True),
        "split8": lambda: _split(rm, U, 8),
        "split4": lambda: _split(rm, U, 4),
    }
    for name, fn in variants.items():
        try:
            o = fn()
            ttnn.synchronize_device(md)
            got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(-1).long()
            ok = torch.equal(got[:U], want) if name.startswith("split") else torch.equal(got[:U], want)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            for _ in range(20):
                fn()
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.synchronize_device(md)
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            us = (time.perf_counter() - t0) / 100 * 1e6
            ttnn.release_trace(md, tid)
            print(f"ARGMAX_BENCH U={U} {name} {us:.1f} us correct={ok}", flush=True)
        except Exception as e:  # noqa
            print(f"ARGMAX_BENCH U={U} {name} FAILED {type(e).__name__} {str(e)[:120]}", flush=True)
