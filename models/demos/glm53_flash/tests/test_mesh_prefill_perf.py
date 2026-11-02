# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel prefill timing per layer on a single mesh row (1, 4): warmed single-shot prefill wall
time at S tokens for the three layer kinds (the per-row TTFT building block until the multi-row
sequence-parallel engine exists), plus the eager decode step time.

    TT_VISIBLE_DEVICES=2,6,14,10 TT_MESH_GRAPH_DESC_PATH=~/g4cov/p150_x4_galaxy_2ch.textproto MESH_DEVICE=P150x4 \
    GLM53_PREFILL_S=1024 pytest models/demos/glm53_flash/tests/test_mesh_prefill_perf.py -s
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
from models.demos.glm53_flash.tt.mesh import mesh_device_params

DUMP = Path("bringup/references/glm53_flash/R4097")
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,4").split(","))
S = int(os.environ.get("GLM53_PREFILL_S", "1024"))
LAYERS = [int(v) for v in os.environ.get("GLM53_PREFILL_LAYERS", "0,3,4").split(",")]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("layer_idx", LAYERS)
def test_tp_prefill_wall(mesh_device, layer_idx):
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    raw = load_layer_raw(snapshot, layer_idx, headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    x = torch.load(DUMP / f"layer{layer_idx}_prefill.pt")["x_in"][:, :S]
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=1, max_seq_len=S + 64
    )
    x_dev = host_streams_to_device(x, mesh_device)
    walls = []
    for it in range(3):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        y = dec.prefill_forward(x_dev, user=0, seq_len=S)
        ttnn.synchronize_device(mesh_device)
        walls.append(time.perf_counter() - t0)
        ttnn.deallocate(y)
    kind = "KDA+dense" if layer_idx < 3 else ("DSA+MoE" if layer_idx % 4 == 3 else "KDA+MoE")
    logger.info(
        f"PREFILL mesh {MESH} layer {layer_idx} ({kind}) S={S}: wall first {walls[0] * 1e3:.1f} ms, warmed "
        f"{min(walls[1:]) * 1e3:.1f} ms ({S / min(walls[1:]):.0f} tok/s)"
    )
    # eager decode step at position S
    dp = dec.make_decode_positions([S])
    xd = host_streams_to_device(x[:, :1], mesh_device, pad_to=1)
    dec.decode_forward(xd, positions=dp)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(5):
        dec.decode_forward(xd, positions=dp)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"DECODE (eager) mesh {MESH} layer {layer_idx} at position {S}: {(time.perf_counter() - t0) / 5 * 1e3:.2f} ms/step"
    )
