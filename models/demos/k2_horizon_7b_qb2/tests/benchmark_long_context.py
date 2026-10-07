# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-user long-context TTFT/TPOT on the full TP4 model (selected precision policy).

Per ISL, K2Generator.generate runs its untimed warm eager prefill (it compiles every chunk-offset
program, since misses are forbidden while a decode trace is live), then the timed request: warmed TTFT
and the steady-state traced decode loop. Prompts are real text, repeated to ISL.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator
from ..tt.multichip_decoder import MultichipDecoder

MODEL_DIR = Path(__file__).resolve().parents[1]


def prompt_tokens(tokenizer, length):
    text = "\n".join(path.read_text(errors="ignore") for path in sorted((MODEL_DIR / "doc").rglob("*.md"))[:200])
    base = tokenizer.encode(text)
    reps = -(-length // len(base))
    return (base * reps)[:length]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--isl", nargs="+", type=int, default=[32767, 65535, 131071, 262015, 524159])
    p.add_argument("--osl", type=int, default=128)
    p.add_argument("--kernels", choices=["new", "old"], default="new")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.kernels == "old":
        MultichipDecoder.accurate_decode_kernel = "chunked"
        MultichipDecoder.accurate_prefill_grid = (8, 8)
        MultichipDecoder.accurate_prefill_q_chunk = 128
        MultichipDecoder.accurate_prefill_max_k_chunk = 128
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    results = {"kernels": a.kernels, "osl": a.osl, "runs": []}
    try:
        start = time.perf_counter()
        gen = build_generator(MODEL_DIR, mesh)
        results["model_load_seconds"] = time.perf_counter() - start
        for isl in sorted(a.isl):
            prompt = prompt_tokens(gen.tokenizer, isl)
            wall = time.perf_counter()
            tokens = gen.generate(prompt, a.osl, trace_prefill=False)
            perf = dict(gen.last_perf)
            row = {
                "isl": isl,
                "osl": a.osl,
                "capacity_tokens": gen.capacity,
                "ttft_seconds": perf["ttft_seconds"],
                "tpot_ms": perf["decode_seconds"] / (a.osl - 1) * 1000,
                "decode_tokens_per_second_per_user": perf["decode_tokens_per_second_per_user"],
                "preparation_seconds": perf["preparation_seconds"],
                "wall_seconds": time.perf_counter() - wall,
                "tokens": tokens,
                "text": gen.tokenizer.decode(tokens[:48]),
            }
            print("LONG_BENCH", json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
            results["runs"].append(row)
            a.output.write_text(json.dumps(results, indent=1) + "\n")
    finally:
        a.output.write_text(json.dumps(results, indent=1) + "\n")
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
