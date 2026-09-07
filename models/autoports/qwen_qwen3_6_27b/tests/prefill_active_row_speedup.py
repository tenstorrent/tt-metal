# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Time one-active-slot prefill at the serving batch, narrow vs full width.

This is the shape vLLM actually issues: chunked prefill is disabled for
model_type=qwen3_5, so each request is prefilled on its own while the other
slots are padding. The full-width path runs the layer stack over all of them.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def _time_prefill(generator, slot, batch, length, iters):
    vocab = generator.model.vocab_size
    tokens = torch.zeros((batch, length), dtype=torch.long)
    tokens[slot, :length] = (torch.arange(length) * 7 + slot * 101 + 3) % vocab
    prompt_lens = [0] * batch
    prompt_lens[slot] = length
    samples = []
    for index in range(iters + 1):  # first iteration compiles
        generator.model.reset_cache()
        generator._slots_requiring_prefill = set(range(batch))
        ttnn.synchronize_device(generator.mesh_device)
        start = time.perf_counter()
        generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=prompt_lens,
        )
        ttnn.synchronize_device(generator.mesh_device)
        elapsed = (time.perf_counter() - start) * 1000.0
        if index:
            samples.append(elapsed)
        else:
            print(f"    (compile+first: {elapsed:.1f} ms)", flush=True)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    result = {}
    try:
        kwargs = {"max_context": max(512, args.length + 128), "batch": args.batch}
        if args.layers:
            kwargs["num_layers"] = args.layers
        generator = build_generator(Path("models/autoports/qwen_qwen3_6_27b"), mesh, **kwargs)
        for label, mode in (("full_width", "0"), ("narrow", "1")):
            os.environ["QWEN36_PREFILL_NARROW"] = mode
            print(f"  {label}:", flush=True)
            result[label] = _time_prefill(generator, args.slot, args.batch, args.length, args.iters)
            print(f"    median {result[label]:.1f} ms", flush=True)
        os.environ.pop("QWEN36_PREFILL_NARROW", None)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    result["speedup"] = result["full_width"] / result["narrow"]
    result.update(batch=args.batch, length=args.length, layers=args.layers)
    print(
        f"ACTIVE_ROW_SPEEDUP batch={args.batch} length={args.length} "
        f"full_width={result['full_width']:.1f} ms narrow={result['narrow']:.1f} ms "
        f"speedup={result['speedup']:.2f}x"
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
