# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer traced decode step of the DSA layer (layer 3: mHC + attention + MoE) at 4 K / 128 K / 1 M positions
for 1, 4 and 8 users on one chip through the traced multi-user sparse decode, plus the op count of the DSA block
(graph capture of one eager attention step). The caches are allocated for 1 M positions per user and left unseeded
(the step's cost does not depend on the cache content; correctness is test_sparse_1m / test_dsa_sparse_traced).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_sparse_perf.py -s --timeout 3600

Knobs: GLM53_SPARSE1M_USERS (default 1,4,8), GLM53_SPARSE1M_POS (default 4096,131072,1048576).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_moe_block import count_ops
from models.demos.glm53_flash.tests.utils import log_dram
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder

LAYER = 3
USERS = [int(v) for v in os.environ.get("GLM53_SPARSE1M_USERS", "1,4,8").split(",")]
POSITIONS = [int(v) for v in os.environ.get("GLM53_SPARSE1M_POS", "4096,131072,1048576").split(",")]
REPLAYS = 20


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200_000_000, "l1_small_size": 32768}], indirect=True)
def test_layer_step_sweep(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    n_max = max(POSITIONS) + 64
    table = []
    for B in USERS:
        t0 = time.perf_counter()
        dec = FunctionalDecoder.from_state_dict(
            raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=B, max_seq_len=n_max
        )
        log_dram(
            mesh_device,
            f"layer {LAYER} decoder for {B} users x {n_max} positions built ({time.perf_counter() - t0:.0f} s)",
        )
        streams = ttnn.from_torch(
            torch.randn(1, 1, B, 4 * cfg.hidden_size) * 0.5,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        x_attn = ttnn.from_torch(
            torch.randn(1, 1, B, cfg.hidden_size) * 0.5,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        dp = dec.make_decode_positions([POSITIONS[0]] * B)
        for N in POSITIONS:
            dec.update_decode_positions(dp, [N] * B)
            # op count of the DSA attention block (one eager step on the persistent inputs)
            attn, cache = dec.layer.attn, dec.cache
            top, dev = count_ops(
                mesh_device,
                lambda: attn.decode_forward(
                    x_attn, cache, dp.positions, dp.pool_ids, dp.ring_slots, sparse_meta=dp.sparse
                ),
            )
            trace_id, out, warm = dec.capture_decode_trace(streams, dp)
            ttnn.deallocate(warm)
            dec.replay_decode(trace_id)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(REPLAYS):
                dec.replay_decode(trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
            ms = (time.perf_counter() - t0) / REPLAYS * 1e3
            ttnn.release_trace(mesh_device, trace_id)
            table.append((N, B, ms, len(top), len(dev)))
            logger.info(
                f"layer {LAYER} traced decode step: {B} users at position {N}: {ms:.3f} ms/step; DSA block {len(top)} ttnn ops / {len(dev)} device ops"
            )
        ttnn.deallocate(streams)
        ttnn.deallocate(x_attn)
        dec.release_weights()
        del dec
    logger.info("| positions | users | layer-3 traced step ms | DSA block ttnn ops | device ops |")
    for N, B, ms, nt, nd in table:
        logger.info(f"| {N} | {B} | {ms:.3f} | {nt} | {nd} |")
