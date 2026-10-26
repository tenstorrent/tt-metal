# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eager vs traced composite mHC wrap at the prefill row count on the 8x4 mesh (tt-blaze G53F-0188).

Per wrap call around a trivial bf16 sublayer, on ``[1,1,T,4*4096]`` fp32 streams replicated over the mesh, chaining
each call's output into the next call (as the layers do) and cycling over ``MHCB_WRAPS`` wraps with different
parameters:

* ``eager``: ``TtMHCWrap.forward`` (the prefill default), back to back, one sync at the end (host issue and device
  work overlap, as in the prefill);
* ``eager_sync``: the same with a sync after every call (host + device serialized);
* ``whole_trace``: one wrap call captured as ONE trace and replayed (the composite's device time);
* ``traced``: ``TtMHCWrapTraced`` with ``GLM53_MHC_PREFILL_TRACE=1`` (two shared traces, the input and parameter
  copies, the eager sublayer, the output clone);
* ``fast``: ``TtMHCFast`` eager (the fused path, G53F-0187, for reference).

and checks ``traced`` == ``eager`` bitwise after the chain (chips 0 and the last).

    MHCB_T=2048 pytest models/demos/glm53_flash/tests/test_mhc_trace_bench.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mhc import (
    MHC_PREFILL_TRACE_ENV,
    TtMHCFast,
    TtMHCWrap,
    TtMHCWrapTraced,
    mhc_kernel_config,
)

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
T = int(os.environ.get("MHCB_T", "2048"))
N = int(os.environ.get("MHCB_N", "24"))
WRAPS = int(os.environ.get("MHCB_WRAPS", "3"))
TRACE_MB = int(os.environ.get("MHCB_TRACE_MB", "64"))
FAST = os.environ.get("MHCB_FAST", "1") == "1"


def _chain(hcs, x, sub, n, sync_each=False, dev=None):
    y = x
    for i in range(n):
        z = hcs[i % len(hcs)].forward(y, sub, sublayer_dtype=ttnn.bfloat16)
        if y is not x:
            ttnn.deallocate(y)
        y = z
        if sync_each:
            ttnn.synchronize_device(dev)
    return y


def _timed_chain(dev, hcs, x, sub, n, sync_each=False):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    y = _chain(hcs, x, sub, n, sync_each, dev)
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / n
    ttnn.deallocate(y)
    return dt


def _chips(t):
    d = ttnn.get_device_tensors(t)
    return ttnn.to_torch(d[0]), ttnn.to_torch(d[-1])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mhc_trace_bench(mesh_device):
    cfg = GLM53FlashConfig()
    kcfg = mhc_kernel_config(cfg)
    n, C = cfg.hc_mult, cfg.hidden_size
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    params = []
    for w in range(WRAPS):
        g = torch.Generator().manual_seed(100 + w)
        fn = (torch.randn(kcfg.mix_hc, n * C, generator=g) * 0.02).bfloat16().float()
        base = torch.randn(kcfg.mix_hc, generator=g) * 0.5
        scale = 0.5 + torch.rand(3, generator=g)
        params.append((fn, base, scale))
    traced = [TtMHCWrapTraced(mesh_device, kcfg, *p, dtype=ttnn.float32) for p in params]
    gain = ttnn.from_torch(
        torch.full((1, 1, 1, C), 0.75),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=rep,
    )

    def sub(t):  # one bf16 op standing in for attention / MoE
        return ttnn.multiply(t, gain)

    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(1, 1, T, n * C), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    rows = {}
    # eager composite (the subclass with the trace off IS TtMHCWrap.forward)
    os.environ[MHC_PREFILL_TRACE_ENV] = "0"
    ttnn.deallocate(_chain(traced, x, sub, 2))  # compile
    rows["eager"] = _timed_chain(mesh_device, traced, x, sub, N)
    rows["eager_sync"] = _timed_chain(mesh_device, traced, x, sub, min(N, 8), sync_each=True)
    ref = _chain(traced, x, sub, 2 * WRAPS)
    ref0, ref1 = _chips(ref)
    ttnn.deallocate(ref)
    # whole wrap as one trace: the device time of the composite
    y = TtMHCWrap.forward(traced[0], x, sub, sublayer_dtype=ttnn.bfloat16)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y = TtMHCWrap.forward(traced[0], x, sub, sublayer_dtype=ttnn.bfloat16)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(N):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    rows["whole_trace"] = (time.perf_counter() - t0) / N
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(y)
    # traced: first call captures
    os.environ[MHC_PREFILL_TRACE_ENV] = "1"
    t0 = time.perf_counter()
    ttnn.deallocate(_chain(traced, x, sub, 1))
    ttnn.synchronize_device(mesh_device)
    rows["traced_capture_call"] = time.perf_counter() - t0
    out = _chain(traced, x, sub, 2 * WRAPS)
    o0, o1 = _chips(out)
    ttnn.deallocate(out)
    rows["traced"] = _timed_chain(mesh_device, traced, x, sub, N)
    rows["traced_sync"] = _timed_chain(mesh_device, traced, x, sub, min(N, 8), sync_each=True)
    os.environ[MHC_PREFILL_TRACE_ENV] = "0"
    eq = (torch.equal(o0, ref0), torch.equal(o1, ref1))
    d = (o0.float() - ref0.float()).abs().max().item()
    logger.info(
        f"[mhc-trace] T={T} chain of {2 * WRAPS} wraps: traced == eager bitwise chip0/last {eq}, max|d| {d:.3e}"
    )
    if FAST:
        fast = [TtMHCFast(mesh_device, kcfg, *p, dtype=ttnn.float32) for p in params]
        ttnn.deallocate(_chain(fast, x, sub, 2))
        rows["fast"] = _timed_chain(mesh_device, fast, x, sub, N)
    for k, v in rows.items():
        logger.info(f"[mhc-trace] T={T} {k}: {v * 1e3:.3f} ms per wrap")
    assert all(eq), eq
