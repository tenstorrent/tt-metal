"""Reduced real-layer profile with the complete terminal and split-sampling path."""

import argparse
import json
import time

import torch
from tracy import signpost

import ttnn

from ..tt.generator import K2Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh, override_num_layers=args.layers)
        prompt = gen.tokenizer.encode(
            "The sky appears blue because sunlight scatters in the atmosphere. " * args.prompt_len
        )[: args.prompt_len]
        assert len(prompt) == args.prompt_len
        gen.generate(prompt, 4)
        # Every captured graph must have a replay for Tracy's trace-op/device
        # enrichment, including the optional argmax comparison outside windows.
        gen.generate(prompt, 4, strategy="argmax")
        for trace in gen.state["sample_traces"].values():
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        gen.generate(prompt, 4)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL")
        begin = time.perf_counter()
        logits = gen._replay_prefill(prompt)
        gen._sample(gen.model.sampler_logits(logits), strategy="split", record_history=True)
        ttnn.synchronize_device(mesh)
        del logits
        prefill = time.perf_counter() - begin
        signpost("PERF_PREFILL_END")
        signpost("PERF_MODEL")
        begin = time.perf_counter()
        gen.replay(sample=False)
        ttnn.synchronize_device(mesh)
        model = time.perf_counter() - begin
        signpost("PERF_MODEL_END")
        signpost("PERF_SAMPLING")
        begin = time.perf_counter()
        ttnn.execute_trace(mesh, gen.state["sample_traces"]["split_history"], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        sampling = time.perf_counter() - begin
        signpost("PERF_SAMPLING_END")
        signpost("PERF_TOKEN_OUT")
        begin = time.perf_counter()
        gen.replay(record_history=True)
        ttnn.synchronize_device(mesh)
        token_out = time.perf_counter() - begin
        signpost("PERF_TOKEN_OUT_END")
        print(
            "PROFILE_HOST",
            json.dumps({"prefill_s": prefill, "model_s": model, "sampling_s": sampling, "token_out_s": token_out}),
            flush=True,
        )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
