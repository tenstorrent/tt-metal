# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced token feedback experiment, with real terminal weights."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cross-request", action="store_true")
    ap.add_argument("--warm-prefills", action="store_true")
    ap.add_argument("--context", type=int, default=262144)
    args = ap.parse_args()
    torch.set_num_threads(8)
    start = time.perf_counter()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=None if args.all_layers else [0, 3], max_seq_len=args.context)
        print("GENERATOR_READY", flush=True)
        if args.warm_prefills:
            for length in (33, 34, 31, 65):
                gen.prefill_logits(list(range(200, 200 + length)))
            print("PREFILL_VARIANTS_PREPARED", flush=True)
        result = gen.generate(list(range(100, 133)), args.steps)
        print("TOKENS", result, gen.counters, flush=True)
        if args.cross_request:
            for length in (33, 34, 31, 65):
                print("NEXT_REQUEST", length, flush=True)
                result = gen.generate(list(range(200, 200 + length)), args.steps)
                print("NEXT_RESULT", result, gen.counters, flush=True)
        Path(args.output).write_text(
            json.dumps(
                {"tokens": result, "counters": gen.counters, "elapsed_seconds": time.perf_counter() - start}, indent=2
            )
            + "\n"
        )
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
