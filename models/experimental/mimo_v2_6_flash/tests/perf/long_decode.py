# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced decode step at long context: ``--users`` users per replica (default 2: one per parity of each chip pair)
decoding at positions up to ~1M, on one row and on all four rows (device-resident greedy loop).

Only the prompt's KV is real; the positions up to the decode position hold whatever the cache had, so this times
the step (every global layer attends over the whole context) without a 1M-token prefill. Reports ms per step and
the per-user decode speed (one token per user per step).

    python models/experimental/mimo_v2_6_flash/tests/perf/long_decode.py [--users 2] [--positions 4096,131072,...]
"""

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.generator import MiMoGalaxyGenerator

MAX_SEQ = (1 << 20) + 4096


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--positions", default="4096,131072,524288,1040384")
    args = parser.parse_args()
    positions = [int(p) for p in args.positions.split(",")]

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
        gen = MiMoGalaxyGenerator(
            mesh,
            cfg,
            MiMoCheckpoint(common.MODEL_PATH, cfg),
            os.path.join(common.CACHE_ROOT, "model_bfp4experts"),
            dp=4,
            users_per_replica=args.users,
            max_seq_len=MAX_SEQ,
        )
        golden = torch.load(common.GOLDEN)["token_ids"]
        prompts = {u: golden[u % len(golden)] for u in range(gen.num_users)}
        first = gen.prefill(prompts)
        gen.capture_decode_traces(greedy_only=True)
        tok = {u: int(first[u].argmax()) for u in prompts}

        def loop(replicas, pos, steps):
            users = [u for u in prompts if gen.locate(u)[0] in replicas]
            t0 = time.time()
            gen.decode_steps({u: tok[u] for u in users}, {u: pos for u in users}, steps, replicas)
            return (time.time() - t0) / steps * 1e3

        loop([0], positions[0], 4)  # warm
        for pos in positions:
            one = sorted(loop([0], pos, args.steps) for _ in range(3))[1]
            four = sorted(loop(list(range(4)), pos, args.steps) for _ in range(3))[1]
            logger.info(
                f"position {pos}: {args.users} users per replica, 1 row {one:.2f} ms/step ({1e3 / one:.1f} tok/s per "
                f"user), 4 rows {four:.2f} ms/step ({1e3 / four:.1f} tok/s per user, {4 * args.users * 1e3 / four:.0f} "
                f"tok/s total)"
            )
            print(f"LONG_DECODE pos={pos} users={args.users} ms_1row={one:.2f} ms_4rows={four:.2f}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
