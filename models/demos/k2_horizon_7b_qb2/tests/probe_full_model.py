"""Fast real-shape full-model probe; defaults to one real dense layer."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--warm-runs", type=int, default=0)
    p.add_argument("--head-dtype", default="bfloat8_b")
    p.add_argument("--head-fidelity", default="HiFi2")
    p.add_argument("--head-split-size", type=int, default=16384)
    p.add_argument("--head-workers", type=int, default=1)
    p.add_argument("--head-k", type=int, default=2)
    p.add_argument("--prompt", default="Explain why the sky appears blue.")
    p.add_argument("--strategy", choices=["split", "argmax"], default="split")
    p.add_argument("--output", default="models/demos/k2_horizon_7b_qb2/doc/full_model/probe.json")
    args = p.parse_args()
    torch.set_num_threads(16)
    start = time.perf_counter()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(
            mesh,
            override_num_layers=args.layers,
            head_dtype=args.head_dtype,
            head_fidelity=args.head_fidelity,
            head_split_size=args.head_split_size,
            head_workers=args.head_workers,
            head_k=args.head_k,
        )
        prompt = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        print("PROMPT", len(prompt), flush=True)
        out = gen.generate(prompt, args.steps, strategy=args.strategy)
        warm_runs = []
        for _ in range(args.warm_runs):
            repeated = gen.generate(prompt, args.steps, strategy=args.strategy)
            assert repeated == out
            warm_runs.append(gen.last_perf.copy())
        result = {
            "layers": args.layers,
            "prompt_tokens": prompt,
            "tokens": out,
            "head_policy": {
                "dtype": args.head_dtype,
                "fidelity": args.head_fidelity,
                "split_size": args.head_split_size,
                "workers_per_bank": args.head_workers,
                "k_block": args.head_k,
            },
            "text": gen.tokenizer.decode(out),
            "perf": gen.last_perf,
            "warm_runs": warm_runs,
            "wall_seconds": time.perf_counter() - start,
        }
        print(json.dumps(result, indent=2), flush=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
