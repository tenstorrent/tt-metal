# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM bandwidth utilization of traced MiMo decode on the Galaxy (TP=8 x DP=4, 4 users per replica).

Utilization = useful bytes each chip must read per decode step / step time / 512 GB/s (Blackhole peak).
Useful bytes count the weights a step needs (attention, router, the selected experts' slices, dense MLP,
LM head) and the KV it attends to, without tile/zero padding. The step time is measured on one row
alone and with all four rows running concurrently (each row is one replica), for the host greedy loop
(the host picks every token) and the device-resident loop (``MiMoGalaxyGenerator.decode_steps``).

    python models/autoports/xiaomimimo_mimo_v2_6_flash_rl/tests/perf/decode_bandwidth.py [--steps 32]
"""

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference.checkpoint import MiMoCheckpoint
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference.config import MiMoConfig
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.generator import MiMoGalaxyGenerator

PEAK_GBPS = 512.0
BFP8, BFP4, BF16 = 1.0625, 0.5625, 2.0
USERS = 4


def useful_bytes_per_chip(cfg, tp, users, positions):
    """Bytes one chip of a TP replica must read for one decode step of ``users`` users at ``positions``."""
    H, k = cfg.hidden_size, cfg.num_experts_per_tok
    i_local = cfg.moe_intermediate_size // tp
    total = 0.0
    for layer in range(cfg.num_hidden_layers):
        d = cfg.attention_dims(layer)
        if d.num_kv_heads >= tp:  # one KV head per chip, 8 Q heads
            heads, kv_users = d.num_heads // tp, users
        else:  # chip pair splits the users; all Q heads of the group
            heads, kv_users = d.num_heads // d.num_kv_heads, users * d.num_kv_heads // tp
        total += H * (heads * d.head_dim + d.head_dim + d.v_head_dim) * BFP8  # qkv
        total += heads * d.v_head_dim * H * BFP8  # o_proj
        ctx = [min(p + 1, d.sliding_window) if d.sliding_window else p + 1 for p in positions]
        total += sum(ctx[:kv_users]) * (d.head_dim + d.v_head_dim) * BFP8  # KV it attends to
        if cfg.is_moe(layer):
            total += H * cfg.n_routed_experts * BF16  # router
            total += users * k * (H * 2 * i_local + i_local * H) * BFP4  # selected experts' slices
        else:
            total += 3 * H * (cfg.intermediate_size // tp) * BFP8
    total += H * (cfg.vocab_size // tp) * BFP8  # LM head
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536, trace_region_size=200_000_000)
    try:
        cfg = MiMoConfig.from_model_path(common.MODEL_PATH)
        ckpt = MiMoCheckpoint(common.MODEL_PATH, cfg)
        gen = MiMoGalaxyGenerator(
            mesh,
            cfg,
            ckpt,
            os.path.join(common.CACHE_ROOT, "model_bfp4experts"),
            dp=4,
            users_per_replica=USERS,
            max_seq_len=4096,
        )
        golden = torch.load(common.GOLDEN)["token_ids"]
        prompts = {u: golden[u % len(golden)] for u in range(gen.num_users)}
        first = gen.prefill(prompts)
        gen.capture_decode_traces()
        start_tok = {u: int(first[u].argmax()) for u in prompts}
        start_pos = {u: prompts[u].numel() for u in prompts}

        def host_loop(replicas, steps):
            """Host greedy: every step reads the per-chip candidates and picks the token on the host."""
            users = [u for u in prompts if gen.locate(u)[0] in replicas]
            cur, pos = {u: start_tok[u] for u in users}, {u: start_pos[u] for u in users}
            out = {u: [] for u in users}
            t0 = time.time()
            for _ in range(steps):
                cands = gen.decode(cur, pos, replicas=replicas, mode="topk")
                for u, c in cands.items():
                    cur[u] = gen.greedy_token(c)
                    pos[u] += 1
                    out[u].append(cur[u])
            return out, (time.time() - t0) / steps

        def device_loop(replicas, steps):
            """Device-resident greedy: argmax, token feedback and positions on device; tokens read in bursts."""
            users = [u for u in prompts if gen.locate(u)[0] in replicas]
            t0 = time.time()
            out = gen.decode_steps({u: start_tok[u] for u in users}, {u: start_pos[u] for u in users}, steps, replicas)
            return out, (time.time() - t0) / steps

        # Same greedy tokens either way (both runs start from the prefill and rewrite the same KV positions).
        check = 8
        ref_tokens, _ = host_loop(list(range(4)), check)
        dev_tokens, _ = device_loop(list(range(4)), check)
        agree = sum(a == b for u in ref_tokens for a, b in zip(ref_tokens[u], dev_tokens[u]))
        total = sum(len(t) for t in ref_tokens.values())
        logger.info(f"device-loop greedy tokens vs host loop: {agree}/{total} equal")
        print(f"DECODE_DEVICE_LOOP_TOKENS_MATCH {agree}/{total}", flush=True)

        results = {}
        for name, loop in (("host loop", host_loop), ("device loop", device_loop)):
            loop([0], 4)  # warm
            results[(name, "1 row")] = loop([0], args.steps)[1]
            results[(name, "4 rows")] = loop(list(range(4)), args.steps)[1]
        positions = [start_pos[u] + args.steps for u in range(USERS)]
        nbytes = useful_bytes_per_chip(cfg, 8, USERS, positions)
        for (name, label), t in results.items():
            gbps = nbytes / t / 1e9
            logger.info(
                f"{name}, {label}: {t * 1e3:.2f} ms/step, {nbytes / 1e9:.3f} GB/chip/step -> {gbps:.1f} GB/s "
                f"per chip, DRAM utilization {100 * gbps / PEAK_GBPS:.1f}% of {PEAK_GBPS:.0f} GB/s"
            )
        for name, tag in (("host loop", "HOSTLOOP"), ("device loop", "DEVLOOP")):
            for label, rows in (("1 row", "1ROW"), ("4 rows", "4ROWS")):
                t = results[(name, label)]
                pct = 100 * nbytes / t / 1e9 / PEAK_GBPS
                print(f"DECODE_BW_UTIL_{tag}_{rows} pct={pct:.2f} ms={t * 1e3:.2f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
