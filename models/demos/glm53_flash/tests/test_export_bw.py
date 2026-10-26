# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device -> host read bandwidth of the seed export's DSA latent cache (tt-blaze G53F-0182).

The seed server exports a DSA layer's latent rows as ``ttnn.slice`` + ``to_torch`` of ONE chip's replica: ~1 GiB at
1M rows, ~1.5 s per layer, 16.8 s of the 1M live TTFT. The cache is replicated on all 32 chips, so each chip can
supply a different 1/32 of the rows. This test times, on a ``[NB, 1, 32, 512]`` bf16 TILE tensor replicated over the
mesh:

* ``one``: today's path (slice the first ``NB`` blocks, read chip 0);
* ``part``: ``ttnn.mesh_partition`` over dim 0 (a per-device slice, no fabric) + one mesh read concatenated on dim 0;
* ``part_rm``: the same with the partition converted to ROW_MAJOR on device (no host untilize).

and checks ``part`` / ``part_rm`` equal ``one`` bitwise on a fully random tensor of ``EXP_NB_CHECK`` blocks.

    EXP_NB=32768 pytest models/demos/glm53_flash/tests/test_export_bw.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
NB = int(os.environ.get("EXP_NB", "32768"))  # 32-row blocks: 32768 = 1M rows = 1 GiB of bf16 per chip
NB_CHECK = int(os.environ.get("EXP_NB_CHECK", "1024"))
REPS = int(os.environ.get("EXP_REPS", "2"))


def _free_view(part, src):
    """Deallocate a slice unless it is src itself (a full-size ttnn.slice returns the same tensor)."""
    if part.buffer_address() != src.buffer_address():
        ttnn.deallocate(part)


def _one(x, nb):
    part = ttnn.slice(x, [0, 0, 0, 0], [nb, 1, 32, 512])
    out = ttnn.to_torch(ttnn.get_device_tensors(part)[0])
    _free_view(part, x)
    return out


def _part(mesh_device, x, nb, rm: bool):
    part = ttnn.slice(x, [0, 0, 0, 0], [nb, 1, 32, 512])
    p = ttnn.mesh_partition(part, 0)
    _free_view(part, x)
    if rm:
        q = ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(p)
        p = q
    out = ttnn.to_torch(p, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(p)
    return out


def _timed(fn, *a):
    ttnn.synchronize_device(a[0] if not isinstance(a[0], ttnn.Tensor) else a[0].device())
    t0 = time.perf_counter()
    out = fn(*a)
    return out, time.perf_counter() - t0


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_export_bw(mesh_device):
    nd = mesh_device.get_num_devices()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    # correctness on a fully random small tensor
    torch.manual_seed(0)
    small = torch.randn(NB_CHECK, 1, 32, 512).to(torch.bfloat16)
    xs = ttnn.from_torch(small, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    a = _one(xs, NB_CHECK)
    b = _part(mesh_device, xs, NB_CHECK, False)
    c = _part(mesh_device, xs, NB_CHECK, True)
    ok = [torch.equal(a, small), torch.equal(b.reshape(a.shape), a), torch.equal(c.reshape(a.shape), a)]
    logger.info(f"[export-bw] check NB={NB_CHECK}: one==src {ok[0]}  part==one {ok[1]}  part_rm==one {ok[2]}")
    ttnn.deallocate(xs)
    # bandwidth on the full-size tensor (content: a random 1/32 block repeated; only the timing matters here)
    base = torch.randn(NB // 32, 1, 32, 512).to(torch.bfloat16)
    xb = ttnn.from_torch(base, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    x = ttnn.repeat(xb, ttnn.Shape([32, 1, 1, 1]))
    ttnn.deallocate(xb)
    gib = NB * 32 * 512 * 2 / 2**30
    for r in range(REPS):
        for name, fn in (
            ("one", lambda: _one(x, NB)),
            ("part", lambda: _part(mesh_device, x, NB, False)),
            ("part_rm", lambda: _part(mesh_device, x, NB, True)),
        ):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            logger.info(
                f"[export-bw] rep {r} {name}: {dt:.3f} s for {gib:.3f} GiB = {gib / dt:.2f} GiB/s  shape {tuple(out.shape)}"
            )
            del out
    # the host part alone: one chip's read without untilize (cpu()) vs to_torch
    part = ttnn.slice(x, [0, 0, 0, 0], [NB - 1, 1, 32, 512])  # a real copy (a full-size slice is the tensor itself)
    d0 = ttnn.get_device_tensors(part)[0]
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    h = d0.cpu()
    t1 = time.perf_counter()
    _ = h.to_torch()
    t2 = time.perf_counter()
    logger.info(f"[export-bw] one chip: device read {t1 - t0:.3f} s, host to_torch (untilize) {t2 - t1:.3f} s, nd={nd}")
    assert all(ok), ok
