# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time one MiMo MoE prefill layer (``TtMiMoMoE.forward``: router, top-8, experts, row all-reduce) on Galaxy row 0.

TP=8 over the row, BFP4 experts from the tensor cache, one 4096-token chunk of normalized random tokens (routing
spreads them over all 256 experts, ~128 tokens per expert). Each call is timed between device syncs (prefill runs
eagerly, so host dispatch counts). Prints one sample per timed call and the median last:

    MOE_PREFILL_SAMPLE ms=<per call>
    MOE_PREFILL_METRIC ms=<median>
"""

import argparse
import statistics
import time

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common

TOKENS = 4096
CALLS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.parse_args()

    mesh, row = common.open_row()
    try:
        cfg = common.config()
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b)
        g = torch.Generator().manual_seed(0)
        rows = torch.nn.functional.normalize(torch.randn(TOKENS, cfg.hidden_size, generator=g), dim=-1)
        x = ttnn.from_torch(
            (rows * cfg.hidden_size**0.5).reshape(1, 1, TOKENS, -1).bfloat16(),
            device=row,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(row),
        )
        moe.forward(x).deallocate(True)  # compile
        samples = []
        for _ in range(CALLS):
            ttnn.synchronize_device(row)
            t0 = time.perf_counter()
            out = moe.forward(x)
            ttnn.synchronize_device(row)
            samples.append((time.perf_counter() - t0) * 1e3)
            out.deallocate(True)
        for s in samples:
            print(f"MOE_PREFILL_SAMPLE ms={s:.3f}", flush=True)
        print(f"MOE_PREFILL_METRIC ms={statistics.median(samples):.3f}", flush=True)
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
