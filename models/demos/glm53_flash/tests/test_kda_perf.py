# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced decode performance of one KDA layer alone (projections, conv, gates, recurrence, gated norm,
o_proj), with Tracy signposts: the per-layer cost of the decode recurrence composite vs the fused kernel.

    TT_VISIBLE_DEVICES=0 GLM53_PERF_B=32 [GLM53_KDA_FUSED=1] pytest models/demos/glm53_flash/tests/test_kda_perf.py -s
    python -m tracy -r -p -v -m pytest ... ; tt-perf-report <csv> --start-signpost PERF_KDA --end-signpost PERF_KDA_END

Pattern: compile pass, synchronize, capture one decode step, warm replay, synchronize, signpost,
replays, synchronize, signpost. The state is a random fp32 carry (the cost does not depend on it).
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
from models.demos.glm53_flash.tt.kda import TtKDA

try:
    from tracy import signpost
except ImportError:  # pragma: no cover

    def signpost(header, message=None):
        pass


LAYER = int(os.environ.get("GLM53_PERF_LAYER", "4"))
B = int(os.environ.get("GLM53_PERF_B", "1"))
STEPS = int(os.environ.get("GLM53_PERF_DECODE_STEPS", "10"))
# local head count: 64 = the whole layer on one chip; 16 emulates the TP4 per-chip shard (the first 16 heads' weights)
H_LOCAL = int(os.environ.get("GLM53_PERF_H", "64"))


def slice_heads(attn: dict, cfg, h: int):
    """The first ``h`` heads' KDA weights (head-major rows / columns) and a config with ``kda_num_heads = h``."""
    import dataclasses

    K = cfg.kda_head_dim
    hk = h * K
    out = dict(attn)
    for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "f_b_proj.weight", "g_b_proj.weight"):
        out[n] = attn[n][:hk]
    for n in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
        out[n] = attn[n][:hk]
    out["b_proj.weight"] = attn["b_proj.weight"][:h]
    out["A_log"] = attn["A_log"][:h]
    out["dt_bias"] = attn["dt_bias"][:hk]
    out["o_proj.weight"] = attn["o_proj.weight"][:, :hk]
    return out, dataclasses.replace(cfg, kda_num_heads=h)


def _read_profiler(device):
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_kda_decode_perf(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=lambda n: n.startswith("self_attn."))
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    if H_LOCAL != cfg.kda_num_heads:
        attn, cfg = slice_heads(attn, cfg, H_LOCAL)
    tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER)
    mode = tt.fused_decode

    torch.manual_seed(0)
    state = tt.allocate_state(B)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(0.05 * torch.randn(B, tt.H, tt.K, tt.V), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
        state.recurrent,
    )
    x = ttnn.from_torch(
        (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )

    tt.decode_forward(x, state)  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y, _ = tt.decode_forward(x, state)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    _read_profiler(mesh_device)
    signpost("PERF_KDA")
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) / STEPS
    signpost("PERF_KDA_END")
    _read_profiler(mesh_device)
    assert torch.isfinite(ttnn.to_torch(y).float()).all()
    logger.info(
        f"KDA layer {LAYER} {mode} H={tt.H} traced decode B={B}: {dt*1e3:.3f} ms/step "
        f"({1/dt:.0f} tok/s/user single-layer rate, {B/dt:.0f} tok/s aggregate)"
    )
    ttnn.release_trace(mesh_device, trace_id)
