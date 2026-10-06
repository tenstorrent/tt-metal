# SPDX-License-Identifier: Apache-2.0
"""Full-stack smoke / latency evidence. Profiling is restricted to reduced layers."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn

from ..tt.checkpoint import SNAPSHOT
from ..tt.generator import build_generator
from .full_memory import memory_views, persistent_tensors
from .full_provenance import provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=int, default=1048576)
    p.add_argument("--reduced", action="store_true")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--tag", default="smoke")
    a = p.parse_args()
    source = provenance()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    out = root / "doc/full_model"
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content="What is the capital of France? Answer in one sentence.")],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        start = time.monotonic()
        gen = build_generator(root, mesh, capacity=a.capacity, layer_indices=[0, 4] if a.reduced else None)
        print("FULL_LOAD_DONE", time.monotonic() - start, flush=True)
        gen.prepare()
        print("FULL_PREPARE_DONE", time.monotonic() - start, flush=True)
        (out / f"{a.tag}_memory.json").write_text(
            json.dumps(
                dict(provenance=source, allocator=memory_views(mesh), persistent_tensors=persistent_tensors(gen)),
                indent=2,
            )
            + "\n"
        )
        ids = gen.generate(prompt, a.tokens)
        result = dict(
            provenance=source,
            prompt=tokenizer.decode(prompt),
            prompt_ids=prompt,
            completion_ids=ids,
            completion=tokenizer.decode(ids),
            counters=dict(gen.counters),
            setup_and_smoke_seconds=time.monotonic() - start,
        )
        print("FULL_COMPLETION", json.dumps(result), flush=True)
        # Matched sampling-inclusive versus logits-only timing, no host token feedback.
        times = {}
        for mode in ("split", "argmax", "logits"):
            gen.bind([42], [128])
            gen.replay(sample=mode != "logits", mode="split" if mode == "logits" else mode)
            ttnn.synchronize_device(mesh)
            tick = time.monotonic()
            for step in range(128):
                gen.replay(sample=mode != "logits", mode="split" if mode == "logits" else mode)
            ttnn.synchronize_device(mesh)
            times[mode] = (time.monotonic() - tick) * 1000 / 128
        result["traced_ms_per_token"] = times
        result["layer_count"] = len(gen.model.layers)
        result["capacity"] = a.capacity
        (out / f"{a.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("FULL_DONE", json.dumps(times), flush=True)
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
