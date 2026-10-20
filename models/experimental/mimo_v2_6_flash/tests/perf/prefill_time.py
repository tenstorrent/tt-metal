# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill time of the 48-layer MiMo on one Galaxy row (one TP=8 replica), one user at a time.

* A ``--short``-token prompt in one chunk and in smaller chunks (whole prefill, logits read back).
* A ``--long``-token prompt in chunks of ``--chunk``, each chunk timed on its own (device synchronized
  after it), so the cost of attending to a growing cached context shows per chunk.

Prompts are random token ids; the second run of every configuration is reported (the first compiles).

    python models/experimental/mimo_v2_6_flash/tests/perf/prefill_time.py [--short 4096] [--long 32768]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=int, default=4096)
    parser.add_argument("--long", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=4096)
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
            users_per_replica=4,
            max_seq_len=max(args.short, args.long),
        )
        model = gen.models[0]
        tokens = torch.randint(
            0, cfg.vocab_size, (max(args.short, args.long),), generator=torch.Generator().manual_seed(0)
        )

        def whole(n, chunk):
            t0 = time.time()
            model.prefill_read(model.prefill_enqueue(tokens[:n], 0, chunk_size=chunk))
            return time.time() - t0

        for chunk in (args.short, args.short // 2, args.short // 4):
            whole(args.short, chunk)
            t = whole(args.short, chunk)
            logger.info(f"{args.short} tokens in chunks of {chunk}: {t * 1e3:.0f} ms, {args.short / t:.0f} tokens/s")

        def per_chunk(n, chunk):
            times = []
            for s in range(0, n, chunk):
                t0 = time.time()
                x, _ = model._prefill_chunk(tokens[s : s + chunk], 0, s, None)
                ttnn.synchronize_device(model.mesh_device)
                times.append(time.time() - t0)
                x.deallocate(True)
            return times

        per_chunk(args.long, args.chunk)
        times = per_chunk(args.long, args.chunk)
        for i, t in enumerate(times):
            logger.info(f"chunk at {i * args.chunk}: {t * 1e3:.0f} ms")
        total = sum(times)
        logger.info(
            f"{args.long} tokens in chunks of {args.chunk}: {total * 1e3:.0f} ms, {args.long / total:.0f} tokens/s"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
