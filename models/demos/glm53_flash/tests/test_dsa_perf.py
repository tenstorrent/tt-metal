# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced decode performance of one DSA layer alone (stems, state maintenance, absorbed flash-MLA decode, o_proj),
with Tracy signposts: the composite vs the fused decode block.

    TT_VISIBLE_DEVICES=0 GLM53_PERF_B=32 [GLM53_DSA_FUSED=1] pytest models/demos/glm53_flash/tests/test_dsa_perf.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache, decode_position_tensors

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(header, message=None):
        pass


LAYER = int(os.environ.get("GLM53_PERF_LAYER", "3"))
B = int(os.environ.get("GLM53_PERF_B", "4"))
STEPS = int(os.environ.get("GLM53_PERF_DECODE_STEPS", "10"))
POS = int(os.environ.get("GLM53_PERF_POS", "1024"))


def _read_profiler(device):
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dsa_decode_perf(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=lambda n: n.startswith("self_attn."))
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    tt = TtDSA(mesh_device, cfg, attn, layer_idx=LAYER)
    mode = tt.fused_decode
    cache = TtLatentCache(mesh_device, cfg, max_batch=B, max_seq_len=POS + 64, block_size=32)
    torch.manual_seed(0)
    x = ttnn.from_torch(
        (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    pos, pool_ids, ring_slots = decode_position_tensors(mesh_device, [POS] * B, cfg.index_kpool)
    tt.decode_forward(x, cache, pos, pool_ids, ring_slots)  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y = tt.decode_forward(x, cache, pos, pool_ids, ring_slots)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    _read_profiler(mesh_device)
    signpost("PERF_DSA")
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) / STEPS
    signpost("PERF_DSA_END")
    _read_profiler(mesh_device)
    assert torch.isfinite(ttnn.to_torch(y).float()).all()
    logger.info(
        f"DSA layer {LAYER} {mode} traced decode B={B} pos={POS}: {dt*1e3:.3f} ms/step ({B/dt:.0f} tok/s aggregate)"
    )
    ttnn.release_trace(mesh_device, trace_id)
