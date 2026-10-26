# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does one prefill mHC wrap's time go? (tt-blaze G53F-0202, on the 8x4 mesh)

``TtMHCFast.forward`` on prefill rows (the production path: the composite projection, the ``mhc_split_sinkhorn``
program, the pre-mix kernel for h, typecast + rms_norm, the v2 post-mix kernel) costs ~4 ms per wrap at 64 rows and
at 2048 rows alike (``test_mhc_fast_exact.py``, G53F-0197): the cost is not the streams' bytes. This times every step
of the wrap on its own -- ``MHCS_N`` back-to-back calls of that step between two synchronizes (the step's
steady-state cost, host or device, whichever binds) -- and the whole wrap the same way, at each ``MHCS_T`` row count.
Each projection op is also timed alone (the composite ``_project``: matmul, x*x, sum over W, the scale ops).

    pytest models/demos/glm53_flash/tests/test_mhc_prefill_split.py -s
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mhc import TtMHCFast, _rows_up, _view_rows, mhc_kernel_config

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
TS = [int(t) for t in os.environ.get("MHCS_T", "2048,64").split(",")]
N = int(os.environ.get("MHCS_N", "20"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mhc_prefill_split(mesh_device):
    cfg = GLM53FlashConfig()
    kcfg = mhc_kernel_config(cfg)
    n, C, eps = cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    g = torch.Generator().manual_seed(7)
    fn = (torch.randn(kcfg.mix_hc, n * C, generator=g) * 0.02).bfloat16().float()
    base = torch.randn(kcfg.mix_hc, generator=g) * 0.5
    scale = 0.5 + torch.rand(3, generator=g)
    fast = TtMHCFast(mesh_device, kcfg, fn, base, scale, dtype=ttnn.float32)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    w = up((1.0 + 0.1 * torch.randn(1, 1, 1, C, generator=g)), ttnn.bfloat16)
    gain = up(torch.full((1, 1, 1, C), 0.75), ttnn.bfloat16)
    ckc = fast.ckc

    enq = {}

    def timed(fn_, name=None):
        for _ in range(2):
            out = fn_()
            ttnn.synchronize_device(mesh_device)
            _free(out)
        t0 = time.perf_counter()
        outs = [fn_() for _ in range(N)]
        t1 = time.perf_counter()  # every call enqueued: the host's part
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / N * 1e3
        if name:
            enq[name] = round((t1 - t0) / N * 1e3, 3)
        for o in outs:
            _free(o)
        return round(dt, 3)

    def _free(o):
        for t in o if isinstance(o, (tuple, list)) else (o,):
            if isinstance(t, ttnn.Tensor) and t.is_allocated():
                ttnn.deallocate(t)

    for T in TS:
        x = up(torch.randn(1, 1, T, n * C, generator=g), ttnn.float32)
        x32 = _view_rows(x, _rows_up(T))
        Tp = x32.shape[-2]
        row = {"T": T}
        row["wrap"] = timed(
            lambda: fast.forward(x, lambda t: ttnn.multiply(t, gain), ttnn.bfloat16, norm=(w, eps)), "wrap"
        )
        # the composite projection, op by op (models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py::_project)
        row["proj.matmul"] = timed(lambda: ttnn.matmul(x32, fast.fn_T, compute_kernel_config=ckc))
        row["proj.square"] = timed(lambda: ttnn.multiply(x32, x32))
        sq = ttnn.multiply(x32, x32)
        row["proj.sum"] = timed(lambda: ttnn.sum(sq, dim=-1, keepdim=True))
        ttnn.deallocate(sq)
        row["project"] = timed(lambda: fast._project(x32, None), "project")
        mixes = fast._project(x32, None)
        row["sinkhorn"] = timed(lambda: fast._parametrize(mixes), "sinkhorn")
        pre, post, comb = fast._parametrize(mixes)

        def hmix():
            h = fast.post_mix.allocate_h(x32)
            fast.post_mix.h_mix(x32, _view_rows(pre, Tp), h)
            return h

        row["hmix"] = timed(hmix, "hmix")
        h = hmix()
        row["typecast"] = timed(lambda: ttnn.typecast(h, ttnn.bfloat16))
        hb = ttnn.typecast(h, ttnn.bfloat16)
        row["rms_norm"] = timed(lambda: ttnn.rms_norm(hb, epsilon=eps, weight=w, compute_kernel_config=ckc))
        y = ttnn.multiply(hb, gain)

        def post_mix():
            out = fast.post_mix.allocate_output(x32, False)
            fast.post_mix(
                x32, _view_rows(y, Tp), _view_rows(post, Tp), _view_rows(comb, Tp), out, False, post_first=True
            )
            return out

        row["post_mix"] = timed(post_mix, "post_mix")
        row["pre"] = timed(lambda: fast._pre(x, (w, eps))[:3])
        for t in (mixes, pre, post, comb, h, hb, y, x):
            if t.is_allocated():
                ttnn.deallocate(t)
        row["enqueue_ms"] = dict(enq)  # host time per call (enqueue only) of the named steps
        enq.clear()
        logger.info(f"[mhc-split] {json.dumps(row)}")
