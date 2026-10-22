# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the decode-ledger items A3 / A4 in ``TtMHCFast`` on one chip, real layer-4 streams and
parameters, decode shapes T=4 (4 users per row on the 8x4) and T=32:

* A3 ``norm_mode``: ``single`` (typecast + one-core interleaved rms_norm, the pre-A3 path) vs ``sharded`` (typecast,
  interleaved_to_sharded, sharded multicore rms_norm, sharded_to_interleaved) vs ``sharded_cast`` (the cast folded
  into the reshard). The normalised bf16 sublayer input and the wrap output are compared mode vs ``single``
  (bar PCC >= 0.9999: the sharded norm's reduction order may flip bf16 ulps) and vs the float64 math.
* A4 ``sinkhorn_mode``: ``kernel`` (the fused mhc_split_sinkhorn program) vs ``batched`` (its tile sequence as 131
  composite ops): pre / post / comb must agree to fp32 rounding (max |d| <= 1e-5 on values in [0, 2]).
* Each mode's ``forward(norm=...)`` is captured in a trace and replayed ITERS times: the per-wrap wall time table
  (the device-time split needs tracy: ``python -m tracy -r -p -v -m pytest ...`` and the ``LayerNorm`` /
  ``MhcSplitSinkhorn`` rows). Expected: LayerNorm 46 -> ~7 us per wrap; the batched Sinkhorn slower than the kernel.

    TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/test_mhc_a3a4_device.py -s
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_mhc import LAYER, _cmp, _down, _streams, _up, torch_reference
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import TtMHCFast, mhc_kernel_config

ITERS = 20
PCC_BAR = 0.9999
SINKHORN_TOL = 1e-5
MODES = [("single", "kernel"), ("sharded", "kernel"), ("sharded_cast", "kernel"), ("sharded", "batched")]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T", [4, 32])
def test_mhc_a3a4_modes(mesh_device, T):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    fn, base, scale = raw["hc_attn_fn"].float(), raw["hc_attn_base"].float(), raw["hc_attn_scale"].float()
    n, C = kcfg.n, kcfg.dim
    g = torch.Generator().manual_seed(0)
    x = _streams(T)  # packed [T, n*C]
    xd = _up(mesh_device, x.reshape(1, 1, T, n * C))
    w_norm_t = 0.8 + 0.4 * torch.rand(1, 1, 1, C, generator=g)
    w_norm = ttnn.from_torch(w_norm_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    gain_bf = ttnn.typecast(_up(mesh_device, 0.5 + torch.rand(1, 1, 1, C, generator=g)), ttnn.bfloat16)
    eps = float(cfg.rms_norm_eps)
    r_mixes, r_h, r_post, r_comb = torch_reference(x, fn, base, scale, kcfg)
    r_hn = r_h * torch.rsqrt(r_h.pow(2).mean(-1, keepdim=True) + eps) * _down(w_norm).reshape(1, C)

    hidden, outs, coefs, rows = {}, {}, {}, []
    for norm_mode, sk_mode in MODES:
        key = f"{norm_mode}/{sk_mode}"
        hc = TtMHCFast(mesh_device, kcfg, fn, base, scale, norm_mode=norm_mode, sinkhorn_mode=sk_mode)

        def capturing(t, _key=key):  # eager only: reads the normalised sublayer input back
            hidden[_key] = _down(t)[0, 0]
            return ttnn.multiply(t, gain_bf)

        def plain(t):
            return ttnn.multiply(t, gain_bf)

        y = hc.forward(xd, capturing, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, eps))
        outs[key] = _down(y)[0, 0]
        ttnn.deallocate(y)
        h, post, comb = hc.hc_pre(xd)
        coefs[key] = (_down(h)[0, 0], _down(post)[0, 0, :, :n], _down(comb)[0, 0, :, : n * n])
        for t in (h, post, comb):
            ttnn.deallocate(t)
        logger.info(f"[{key}] T={T}")
        _cmp(f"  normed hidden vs float64", r_hn, hidden[key], 0.999)
        # traced wall time per wrap (program cache warm from the eager call above)
        y = hc.forward(xd, plain, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, eps))
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(y)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        y = hc.forward(xd, plain, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, eps))
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        traced = _down(y)[0, 0]
        assert pcc(outs[key], traced) > 0.99999, f"{key}: traced replay differs from the eager call"
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / ITERS
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(y)
        rows.append((key, dt * 1e6))
        logger.info(f"  traced wrap: {dt * 1e6:.1f} us per call ({ITERS} replays)")

    base_key = "single/kernel"
    for key in outs:
        if key == base_key:
            continue
        logger.info(f"[{key}] vs [{base_key}]")
        _cmp("  normed hidden", hidden[base_key], hidden[key], PCC_BAR)
        _cmp("  wrap output", outs[base_key], outs[key], PCC_BAR)
        if key.endswith("/batched"):
            for name, ref, got in zip(("h", "post", "comb"), coefs[base_key], coefs[key]):
                d = (ref.double() - got.double()).abs().max().item()
                logger.info(f"  {name}: max |batched - kernel| {d:.3e}")
                if name != "h":
                    assert d <= SINKHORN_TOL, f"{name}: {d} > {SINKHORN_TOL}"
    logger.info(
        f"| T={T} mode (norm/sinkhorn) | us per traced wrap |\n|---|---|\n"
        + "\n".join(f"| {k} | {us:.1f} |" for k, us in rows)
    )
    ttnn.deallocate(xd)
