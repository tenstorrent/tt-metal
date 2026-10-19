# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced decode on TP=8 x DP=4: trace == eager, and the four rows run concurrently.

Each replica serves two golden prompts twice: slots 0-1 decode eagerly, slots 2-3 replay the
captured decode trace. Both greedy streams must produce the same tokens. Then we time traced
decode steps on one row alone and on all four rows; with independent replicas the four-row step
should cost about the same as the one-row step (4x aggregate throughput). Last, the device-resident
greedy loop (greedy-only trace: GreedyHead, token and position fed back on device) must produce the
host loop's tokens.
"""

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.tests.common import (
    golden_path,
    DP,
    GALAXY_DEVICE_PARAMS,
    cache_dir,
    load_config_and_ckpt,
    model_path,
    pcc,
)
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

TRACE_DEVICE_PARAMS = {**GALAXY_DEVICE_PARAMS, "trace_region_size": 150_000_000}
GREEDY = 12
TIMED_STEPS = 10
DEVICE_LOOP_STEPS = 16


def golden_prompts():
    path = golden_path()
    if not os.path.exists(path):
        pytest.skip(f"golden file {path} not found")
    return torch.load(path)["token_ids"]


@pytest.mark.parametrize("device_params", [TRACE_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_model_traced_tp8_dp4(mesh_device):
    ids = golden_prompts()
    cfg, ckpt = load_config_and_ckpt()
    tok = AutoTokenizer.from_pretrained(model_path())
    gen = MiMoGalaxyGenerator(
        mesh_device, cfg, ckpt, cache_dir("model_bfp4experts"), dp=DP, users_per_replica=4, max_seq_len=1024
    )

    # replica r: slots 0/1 hold golden prompts 2r/2r+1 (eager); slots 2/3 hold the same prompts (traced)
    prompts, twin = {}, {}
    for r in range(DP):
        for s in range(2):
            prompts[r * 4 + s] = prompts[r * 4 + s + 2] = ids[2 * r + s]
            twin[r * 4 + s] = r * 4 + s + 2
    first = gen.prefill(prompts)
    for a, b in twin.items():
        assert pcc(first[a], first[b]) > 0.9999

    def greedy(users, steps, record):
        cur = {u: int(first[u].argmax()) for u in users}
        pos = {u: prompts[u].numel() for u in users}
        out = {u: [cur[u]] for u in users}
        for _ in range(steps - 1):
            logits = gen.decode(cur, pos)
            for u, lg in logits.items():
                record.setdefault(u, []).append(lg)
                cur[u] = int(lg.argmax())
                out[u].append(cur[u])
                pos[u] += 1
        return out, cur, pos

    eager_logits, traced_logits = {}, {}
    t0 = time.time()
    eager_tokens, _, _ = greedy(list(twin), GREEDY, eager_logits)
    eager_step = (time.time() - t0) / (GREEDY - 1)

    gen.capture_decode_traces()
    t0 = time.time()
    traced_tokens, cur, pos = greedy(list(twin.values()), GREEDY, traced_logits)
    traced_step = (time.time() - t0) / (GREEDY - 1)

    worst = 1.0
    for a, b in twin.items():
        r, s = gen.locate(a)
        logger.info(f"replica {r} prompt {tok.decode(prompts[a])!r}")
        logger.info(f"  eager : {tok.decode(eager_tokens[a])!r}")
        logger.info(f"  traced: {tok.decode(traced_tokens[b])!r}")
        for la, lb in zip(eager_logits[a], traced_logits[b]):
            worst = min(worst, pcc(la, lb))
    logger.info(f"eager vs traced logits: worst PCC {worst:.6f}")

    # ---- the per-chip top-k readback picks the same greedy token as the full logits (same position twice
    # rewrites the same KV, so the probe does not disturb the users)
    full = gen.decode(cur, pos)
    cands = gen.decode(cur, pos, mode="topk")
    for u in cur:
        assert int(full[u].argmax()) == gen.greedy_token(cands[u]), u

    # ---- throughput: one row vs all four rows, traced
    def timed(replicas):
        users = [u for u in cur if gen.locate(u)[0] in replicas]
        t0 = time.time()
        for _ in range(TIMED_STEPS):
            cands = gen.decode({u: cur[u] for u in users}, {u: pos[u] for u in users}, replicas=replicas, mode="topk")
            for u, c in cands.items():
                cur[u] = gen.greedy_token(c)
                pos[u] += 1
        return (time.time() - t0) / TIMED_STEPS

    one_row = timed([0])
    four_rows = timed(list(range(DP)))
    users_active = 2
    logger.info(
        f"decode step (2 users per row): eager {eager_step * 1e3:.0f} ms (4 rows); traced 1 row {one_row * 1e3:.1f} ms, "
        f"4 rows {four_rows * 1e3:.1f} ms -> {DP * users_active / four_rows:.1f} tok/s aggregate vs "
        f"{users_active / one_row:.1f} tok/s on one row"
    )

    # ---- device-resident greedy loop vs the host loop, each from a fresh prefill of the prompts (decode releases the
    # SWA blocks behind the window, so a run cannot restart from an earlier position without its prefill)
    first = gen.prefill(prompts)
    start_tok = {u: int(first[u].argmax()) for u in prompts}
    start_pos = {u: prompts[u].numel() for u in prompts}
    cur, pos = dict(start_tok), dict(start_pos)
    host_tokens = {u: [] for u in cur}
    for _ in range(DEVICE_LOOP_STEPS):
        cands = gen.decode(cur, pos, mode="topk")
        for u, c in cands.items():
            cur[u] = gen.greedy_token(c)
            pos[u] += 1
            host_tokens[u].append(cur[u])
    gen.capture_decode_traces(greedy_only=True)
    gen.prefill(prompts)
    device_tokens = gen.decode_steps(start_tok, start_pos, DEVICE_LOOP_STEPS)
    for u in host_tokens:
        assert device_tokens[u] == host_tokens[u], (u, host_tokens[u], device_tokens[u])

    for a, b in twin.items():
        assert eager_tokens[a] == traced_tokens[b], (a, eager_tokens[a], traced_tokens[b])
    assert worst > 0.999, worst
    assert four_rows < 2.0 * one_row, (one_row, four_rows)
