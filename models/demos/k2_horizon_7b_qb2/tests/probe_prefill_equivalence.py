"""Isolate eager/trace and per-token/buffered generation for one prompt shape."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=4095)
    p.add_argument("--layers", type=int, default=36)
    p.add_argument("--warm-transition", action="store_true")
    p.add_argument("--cache-capacity", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {"length": args.length, "layers": args.layers, "runs": []}
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers)
        if args.cache_capacity:
            gen._ensure_owned_cache(1, args.cache_capacity)
            result["cache_capacity"] = gen.capacity
        prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 600)[
            : args.length
        ]
        if args.warm_transition:
            from .probe_optimized_generator import run

            # Preserve the exact history/program/cache preconditions from the
            # full boundary suite, then isolate the next long request.
            run(
                gen,
                str(args.output.with_name(args.output.stem + "_warm.json")),
                lengths=[1, 31, 32, 127, 129, 223, 224, 225, 255, 256, 257],
            )
            result["transition_history_capacity"] = gen.token_history.shape[2]
            result["transition_cache_capacity"] = gen.capacity
        for traced, output in [
            (False, "per_token"),
            (True, "buffered"),
            (False, "buffered"),
            (True, "per_token"),
            (False, "per_token"),
            (True, "buffered"),
        ]:
            tokens = gen.generate(prompt, 4, trace_prefill=traced, token_output=output)
            result["runs"].append(
                {"traced_prefill": traced, "output_mode": output, "tokens": tokens, "perf": gen.last_perf.copy()}
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("MODE", traced, output, tokens, flush=True)
        gen.reset()
        trace_logits = gen._read_logits(gen._replay_prefill(prompt)).float().flatten()
        gen.reset()
        eager_logits = (
            gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[len(prompt)],
                sampling_mode="host",
            )
            .float()
            .flatten()
        )
        result["logits"] = {
            "exact": torch.equal(trace_logits, eager_logits),
            "pcc": float(torch.corrcoef(torch.stack([trace_logits, eager_logits]))[0, 1]),
            "max_abs": float((trace_logits - eager_logits).abs().max()),
            "trace_top10": {
                "values": torch.topk(trace_logits, 10).values,
                "indices": torch.topk(trace_logits, 10).indices,
            },
            "eager_top10": {
                "values": torch.topk(eager_logits, 10).values,
                "indices": torch.topk(eager_logits, 10).indices,
            },
        }
        for key in ["trace_top10", "eager_top10"]:
            result["logits"][key] = {k: v.tolist() for k, v in result["logits"][key].items()}
        result["complete"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
