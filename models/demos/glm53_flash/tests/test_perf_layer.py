# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed prefill and traced decode performance of one decoder layer, with Tracy signposts.

Collect (one layer kind at a time, device profiler only, no watcher):

    TT_VISIBLE_DEVICES=0 GLM53_PERF_LAYER=3 python -m tracy -r -p -v -m pytest \
        models/demos/glm53_flash/tests/test_perf_layer.py -s
    tt-perf-report <ops_perf_results_*.csv> --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END ...

Pattern: compile/warm-up, synchronize, signpost start, measured warmed pass (traced for decode),
synchronize, signpost end. Wall-clock per step is also logged (ms and tok/s per user).
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
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder, host_streams_to_device

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy module ships with the source tree

    def signpost(header, message=None):
        pass


def _read_profiler(device):
    """Drain the device profiler buffers (they fill during long warm-ups and later ops are dropped)."""
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:  # not built with the profiler
        pass


LAYER = int(os.environ.get("GLM53_PERF_LAYER", "3"))
PREFILL_LEN = int(os.environ.get("GLM53_PERF_PREFILL", "1024"))
DECODE_STEPS = int(os.environ.get("GLM53_PERF_DECODE_STEPS", "3"))
# "both" | "prefill" | "decode": the Tracy device-zone hash table collides for the full prefill+decode kernel
# set of a MoE layer ("Source location hashes are colliding"), so each phase can be profiled in its own run
PHASE = os.environ.get("GLM53_PERF_PHASE", "both")
# 1: skip the prefill entirely and seed the DSA latent cache with random rows (the decode profile does
# not depend on the cache content), so a decode-only profile compiles no SDPA prefill kernel at all:
# both the dense and the sparse prefill kernels share compute_streaming.hpp, whose Tracy zones
# collide on this commit (profiler.cpp "Source location hashes are colliding")
SEED_ONLY = os.environ.get("GLM53_PERF_SEED_ONLY", "0") == "1"
DUMP = Path("bringup/references/glm53_flash/T4097")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_layer_perf(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{LAYER}_prefill.pt")
    S = PREFILL_LEN
    x_in = pre["x_in"][:, :S]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )
    kind = f"{'KDA' if dec.layer.is_kda else 'DSA'}+{'MoE' if dec.layer.is_moe else 'dense'}"
    if SEED_ONLY:
        if dec.cache is not None:
            g = torch.Generator().manual_seed(0)
            rows = -(-S // 32) * 32
            latent = torch.randn(1, 1, rows, cfg.kv_lora_rank, generator=g) * 0.5
            dec.cache.fill(
                ttnn.from_torch(latent, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device), user=0
            )
        logger.info(f"layer {LAYER} ({kind}) seed-only: prefill skipped, cache seeded with {S} random rows")

    # ---- prefill: compile pass, then the measured warmed pass
    x_dev = host_streams_to_device(x_in, mesh_device)
    if not SEED_ONLY:
        dec.prefill_forward(x_dev, user=0, seq_len=S)
        ttnn.synchronize_device(mesh_device)
        _read_profiler(mesh_device)
    if PHASE in ("both", "prefill") and not SEED_ONLY:
        signpost("PERF_PREFILL")
        t0 = time.perf_counter()
        y = dec.prefill_forward(x_dev, user=0, seq_len=S)
        ttnn.synchronize_device(mesh_device)
        t_prefill = time.perf_counter() - t0
        signpost("PERF_PREFILL_END")
        _read_profiler(mesh_device)
        ttnn.deallocate(y)
        logger.info(
            f"layer {LAYER} ({kind}) warmed prefill S={S}: {t_prefill*1e3:.1f} ms wall ({S / t_prefill:.0f} tok/s)"
        )
    if PHASE == "prefill":
        return

    # ---- decode: capture trace (warm-up inside), then measured replays
    streams_dev = ttnn.from_torch(
        torch.zeros(1, 1, 1, cfg.hc_mult * cfg.hidden_size),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            pre["x_in"][:, S : S + 1].reshape(1, 1, 1, -1).float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
        ),
        streams_dev,
    )
    dp = dec.make_decode_positions([S])
    trace_id, out_dev, _ = dec.capture_decode_trace(streams_dev, dp)
    dec.replay_decode(trace_id)  # one warm replay
    ttnn.synchronize_device(mesh_device)
    _read_profiler(mesh_device)
    signpost("PERF_DECODE")
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        dec.replay_decode(trace_id, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t_decode = (time.perf_counter() - t0) / DECODE_STEPS
    signpost("PERF_DECODE_END")
    _read_profiler(mesh_device)
    logger.info(
        f"layer {LAYER} ({kind}) traced decode B=1: {t_decode*1e3:.2f} ms/step wall "
        f"(single-layer rate {1/t_decode:.0f} tok/s/user; x45 layers -> {1/(45*t_decode):.1f} tok/s/user)"
    )
    ttnn.release_trace(mesh_device, trace_id)
