# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stress the serving path for hangs: the model (MIMO_STRESS_LAYERS, default 48, + lm_head) runs MIMO_STRESS_ITERS
eager chunks back to back, each with fresh random token ids (routing changes every chunk) at a chunk position cycling
over MIMO_STRESS_POSITIONS (default 16) like a growing conversation, the next-token logits read back each chunk (as the
server does). Progress (iteration, position, ms) is logged every MIMO_STRESS_LOG chunks, so a hang names where it hit.

    MIMO_STRESS_ITERS (2000), MIMO_STRESS_CHUNK (1024), MIMO_STRESS_SEED (0), MIMO_STRESS_SAME_TOKENS=1 (repeat one input)
"""

import os
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

ITERS = int(os.environ.get("MIMO_STRESS_ITERS", "2000"))
CHUNK = int(os.environ.get("MIMO_STRESS_CHUNK", "1024"))
N_LAYERS = int(os.environ.get("MIMO_STRESS_LAYERS", "48"))
POSITIONS = int(os.environ.get("MIMO_STRESS_POSITIONS", "16"))
LOG_EVERY = int(os.environ.get("MIMO_STRESS_LOG", "50"))
SEED = int(os.environ.get("MIMO_STRESS_SEED", "0"))
SAME = os.environ.get("MIMO_STRESS_SAME_TOKENS") == "1"


@pytest.mark.timeout(86400)
@MESH_PARAMS
def test_model_stress(mesh_device, device_params):
    cfg = MiMoTextConfig.from_json()
    model = TtMiMoModel(
        mesh_device,
        cfg,
        lambda i: layer_state(i, cfg),
        fabric_config=device_params["fabric_config"],
        max_seq_len=(POSITIONS + 1) * CHUNK,
        chunk_size=CHUNK,
        layers=list(range(N_LAYERS)),
        global_state=global_state,
        lm_head=N_LAYERS == cfg.num_hidden_layers,
        options=MiMoRuntimeOptions.from_env(),
    )
    g = torch.Generator().manual_seed(SEED)
    fixed = torch.randint(0, cfg.vocab_size, (CHUNK,), generator=g)
    t_start = time.perf_counter()
    ms = []
    for it in range(ITERS):
        pos = it % POSITIONS
        ids = fixed if SAME else torch.randint(0, cfg.vocab_size, (CHUNK,), generator=g)
        t0 = time.perf_counter()
        x = model.prefill_chunk(ids, pos * CHUNK)
        if model.lm_head is not None:
            logits = model.next_token_logits(x, pos * CHUNK, pos * CHUNK + CHUNK - 1)
            assert logits.isfinite().all(), (it, pos)
        else:
            ttnn.synchronize_device(mesh_device)
        x.deallocate(True)
        ms.append((time.perf_counter() - t0) * 1e3)
        if (it + 1) % LOG_EVERY == 0 or it < 3:
            s = sorted(ms[-LOG_EVERY:])
            logger.info(
                f"STRESS {mesh_id(mesh_device)} iter {it + 1}/{ITERS} pos {pos}: median {s[len(s) // 2]:.1f} ms, "
                f"max {s[-1]:.1f} ms, elapsed {time.perf_counter() - t_start:.0f} s"
            )
            sys.stderr.flush()
    logger.info(f"STRESS done: {ITERS} chunks, no hang, {time.perf_counter() - t_start:.0f} s")
