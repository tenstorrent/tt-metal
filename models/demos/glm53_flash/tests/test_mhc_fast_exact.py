# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the fused prefill mHC path bit-identical to the composite? (tt-blaze G53F-0188, on the 8x4 mesh)

The decoder's two prefill wrap paths, on the same random streams and parameters:

* composite: ``TtMHCWrap.forward(x, normed, bf16)`` with the decoder's closure ``normed(h) = F(rms_norm(h))``
  (``decoder_layer._hc_call_impl``, the decoder's ``norm_compute``);
* fused: ``TtMHCFast.forward(x, F, bf16, norm=(w, eps))`` (the norm inside its pre-block, the v2 post-mix kernel),
  under ``GLM53_MHC_POST_ORDER`` = ``composite`` (the composite's term order) and ``kernel`` (the original order),
  and in the composite order with ``GLM53_MHC_PREFILL_HMIX=1`` (the pre-mix kernel for h, G53F-0197).

F is one bf16 op. Reports, per row count, whether the sublayer input (the pre-block + norm) and the wrap output are
bitwise equal, the max |d|, and the eager time per wrap of each path.

    pytest models/demos/glm53_flash/tests/test_mhc_fast_exact.py -s
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
from models.demos.glm53_flash.tt.mhc import TtMHCFast, TtMHCWrap, mhc_kernel_config

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
TS = [int(t) for t in os.environ.get("MHCX_T", "2048,1024,64").split(",")]
N = int(os.environ.get("MHCX_N", "10"))


def _chip0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mhc_fast_exact(mesh_device):
    cfg = GLM53FlashConfig()
    kcfg = mhc_kernel_config(cfg)
    n, C, eps = cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    g = torch.Generator().manual_seed(7)
    fn = (torch.randn(kcfg.mix_hc, n * C, generator=g) * 0.02).bfloat16().float()
    base = torch.randn(kcfg.mix_hc, generator=g) * 0.5
    scale = 0.5 + torch.rand(3, generator=g)
    comp = TtMHCWrap(mesh_device, kcfg, fn, base, scale, dtype=ttnn.float32)
    fast = TtMHCFast(mesh_device, kcfg, fn, base, scale, dtype=ttnn.float32)
    up = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    w = up((1.0 + 0.1 * torch.randn(1, 1, 1, C, generator=g)), ttnn.bfloat16)
    gain = up(torch.full((1, 1, 1, C), 0.75), ttnn.bfloat16)
    norm_compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    seen = {}

    def F(t):
        if "h" not in seen:
            seen["h"] = ttnn.clone(t)
        return ttnn.multiply(t, gain)

    def normed(h):
        return F(ttnn.rms_norm(h, epsilon=eps, weight=w, compute_kernel_config=norm_compute))

    def run_comp(x):
        return comp.forward(x, normed, sublayer_dtype=ttnn.bfloat16)

    def run_fast(x):
        return fast.forward(x, F, sublayer_dtype=ttnn.bfloat16, norm=(w, eps))

    def once(fn_, x):
        seen.clear()
        y = fn_(x)
        h = _chip0(seen.pop("h"))
        return _chip0(y), h, y

    def timed(fn_, x):
        for _ in range(2):
            ttnn.deallocate(fn_(x))
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(N):
            ttnn.deallocate(fn_(x))
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / N * 1e3

    ok = True
    for T in TS:
        x = up(torch.randn(1, 1, T, n * C, generator=g), ttnn.float32)
        ref, ref_h, yr = once(run_comp, x)
        ttnn.deallocate(yr)
        t_comp = timed(run_comp, x)
        for order, hmix in (("composite", "0"), ("composite", "1"), ("kernel", "0")):
            os.environ["GLM53_MHC_POST_ORDER"] = order
            os.environ["GLM53_MHC_PREFILL_HMIX"] = hmix  # G53F-0197: the pre-mix kernel for the prefill h
            got, got_h, yg = once(run_fast, x)
            ttnn.deallocate(yg)
            t_fast = timed(run_fast, x)
            h_eq, y_eq = torch.equal(got_h, ref_h), torch.equal(got, ref)
            d = (got.float() - ref.float()).abs()
            logger.info(
                f"[mhc-exact] T={T} order={order} hmix={hmix}: sublayer input equal {h_eq}, output equal {y_eq} "
                f"(max|d| {d.max().item():.3e}, {int((d > 0).sum())} / {d.numel()} differ); "
                f"composite {t_comp:.3f} ms, fused {t_fast:.3f} ms per wrap"
            )
            if order == "composite" and T > 32:
                ok = ok and h_eq and y_eq
        os.environ.pop("GLM53_MHC_POST_ORDER", None)
        os.environ.pop("GLM53_MHC_PREFILL_HMIX", None)
        ttnn.deallocate(x)
    assert ok, "the fused prefill path in the composite order is not bit-identical"
