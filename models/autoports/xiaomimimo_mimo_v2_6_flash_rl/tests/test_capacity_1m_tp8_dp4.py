# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 sizing point on the Galaxy: TP=8 x DP=4, 4 users per replica, 1M context each.

Builds all four replicas with max_seq_len = 1M, reports DRAM per chip after the weights and after
the KV caches, checks greedy text at short positions, then times traced decode with every user at
4K, 128K and ~1M positions. Only the prompt and the decoded tokens hold real KV; the positions in
between are zero, so the long-position logits are only checked for being finite.
"""

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.common import (
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    load_config_and_ckpt,
    model_path,
)
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.generator import MiMoGalaxyGenerator
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.model import TtMiMoModel

MAX_SEQ = 1 << 20
USERS_PER_REPLICA = 4
TRACE_DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": 150_000_000}
TIMED_STEPS = 5


def dram_gb(device):
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    gb = mv.num_banks / 1e9
    return mv.total_bytes_allocated_per_bank * gb, mv.total_bytes_per_bank * gb


@pytest.mark.parametrize("device_params", [TRACE_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_capacity_1m_tp8_dp4(mesh_device, monkeypatch):
    golden = os.environ.get("MIMO_GOLDEN", "/home/zni/models/mimo-golden/golden_48l.pt")
    if not os.path.exists(golden):
        pytest.skip(f"golden file {golden} not found")
    prompts_all = torch.load(golden)["token_ids"]
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())

    # Measure weights alone, then with the 1M KV caches.
    setup_kv = TtMiMoModel.setup_kv_cache
    after_weights = {}

    def measured_setup(model, num_users):
        after_weights[id(model)] = dram_gb(model.mesh_device)
        return setup_kv(model, num_users)

    monkeypatch.setattr(TtMiMoModel, "setup_kv_cache", measured_setup)
    gen = MiMoGalaxyGenerator(
        mesh_device,
        cfg,
        ckpt,
        cache_dir("model_bfp4experts"),
        dp=DP,
        users_per_replica=USERS_PER_REPLICA,
        max_seq_len=MAX_SEQ,
    )
    for r, model in enumerate(gen.models):
        w_used, total = after_weights[id(model)]
        used, _ = dram_gb(model.mesh_device)
        logger.info(
            f"replica {r} DRAM per chip: weights {w_used:.2f} GB, + KV for {USERS_PER_REPLICA} x 1M -> "
            f"{used:.2f} GB of {total:.2f} GB ({total - used:.2f} GB free)"
        )

    prompts = {u: prompts_all[u % len(prompts_all)] for u in range(gen.num_users)}
    first = gen.prefill(prompts)
    gen.capture_decode_traces()

    cur = {u: int(first[u].argmax()) for u in prompts}
    pos = {u: prompts[u].numel() for u in prompts}
    text = {u: [cur[u]] for u in prompts}
    for _ in range(7):
        out = gen.decode(cur, pos)
        for u, lg in out.items():
            cur[u] = int(lg.argmax())
            text[u].append(cur[u])
            pos[u] += 1
    for u in range(0, gen.num_users, USERS_PER_REPLICA + 1):
        logger.info(f"user {u} (replica {gen.locate(u)[0]}): {tok.decode(prompts[u])!r} -> {tok.decode(text[u])!r}")

    timings = {}
    for label, p in (("4K", 4096), ("128K", 128 * 1024), ("1M", MAX_SEQ - TIMED_STEPS - 1)):
        step_pos = {u: p for u in prompts}
        t0 = time.time()
        for _ in range(TIMED_STEPS):
            out = gen.decode(cur, step_pos)
            for u in step_pos:
                step_pos[u] += 1
        timings[label] = (time.time() - t0) / TIMED_STEPS
        finite = all(torch.isfinite(lg).all() for lg in out.values())
        logger.info(
            f"traced decode, 16 users at position {label}: {timings[label] * 1e3:.1f} ms/step "
            f"({gen.num_users / timings[label]:.1f} tok/s aggregate), logits finite: {finite}"
        )
        assert finite, label
