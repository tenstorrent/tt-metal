# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark of one mHC wrap (hc_pre + bf16 casts + hc_post around a trivial sublayer):
``TtMHCFast`` vs ``TtMHCWrap``, traced, at the decode (T=32 = 32 users) and prefill (T=1024)
shapes, real layer-4 parameters and streams.

    TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/test_mhc_perf.py -s
    # op table: python -m tracy -r -p -v -m pytest ... (signposts MHC_<impl>_T<T> .. _END around ONE
    # replay), then tt-perf-report <csv> --start-signpost MHC_fast_T32 --end-signpost MHC_fast_T32_END

Knobs: GLM53_MHC_PERF_T (comma list, default "32,1024"), GLM53_MHC_PERF_ITERS (default 20),
GLM53_MHC_PERF_IMPLS (default "wrap,fast").
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import build_mhc

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(header, message=None):
        pass


DUMP = Path("bringup/references/glm53_flash/R4097/layer4_prefill.pt")
LAYER = 4
SHAPES = [int(t) for t in os.environ.get("GLM53_MHC_PERF_T", "32,1024").split(",")]
ITERS = int(os.environ.get("GLM53_MHC_PERF_ITERS", "20"))
IMPLS = os.environ.get("GLM53_MHC_PERF_IMPLS", "wrap,fast").split(",")


def _read_profiler(device):
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mhc_wrap_perf(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    n, C = cfg.hc_mult, cfg.hidden_size
    x_in = torch.load(DUMP)["x_in"][0]  # [4097, 4, 4096]
    gain = ttnn.from_torch(
        torch.full((1, 1, 1, C), 0.75), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )

    def sublayer(t):  # one bf16 op standing in for attention / MoE
        return ttnn.multiply(t, gain)

    rows = []
    for T in SHAPES:
        x = x_in[:T].reshape(1, 1, T, n * C).float()
        outs = {}
        for impl in IMPLS:
            hc = build_mhc(mesh_device, cfg, raw["hc_attn_fn"], raw["hc_attn_base"], raw["hc_attn_scale"], impl=impl)
            xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
            # compile + warm, then capture one wrap call
            y = hc.forward(xd, sublayer, sublayer_dtype=ttnn.bfloat16)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(y)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            y = hc.forward(xd, sublayer, sublayer_dtype=ttnn.bfloat16)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            _read_profiler(mesh_device)
            signpost(f"MHC_{impl}_T{T}")
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            signpost(f"MHC_{impl}_T{T}_END")
            _read_profiler(mesh_device)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / ITERS
            outs[impl] = ttnn.to_torch(y).float()
            ttnn.release_trace(mesh_device, tid)
            ttnn.deallocate(y)
            ttnn.deallocate(xd)
            rows.append((T, impl, dt * 1e3))
            logger.info(f"mHC wrap {impl} T={T}: {dt * 1e3:.3f} ms per wrap call (traced, {ITERS} replays)")
        if len(outs) == 2:
            p = pcc(outs["wrap"], outs["fast"])
            logger.info(f"  T={T}: fast vs wrap output pcc {p:.7f}")
            assert p > 0.99999
    logger.info(
        "| T | impl | ms per wrap |\n|---|---|---|\n" + "\n".join(f"| {T} | {i} | {ms:.3f} |" for T, i, ms in rows)
    )
