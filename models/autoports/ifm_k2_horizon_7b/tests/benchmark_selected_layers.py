"""Unprofiled traced latency of both selected multichip layer classes.

Inputs come from the real two-layer model after a normal request. Each layer
uses the stack's persistent CCL buffers and width-sharded residual boundary.
Position and RoPE are fixed during this layer-only measurement; terminal and
position-advance work belong to the separate complete-model accounting.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

import ttnn

from ..tt.generator import K2Generator


def read(tensor):
    host = tensor.cpu(blocking=True)
    return [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(host)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.trials < 1:
        parser.error("iterations and trials must be positive")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") or os.environ.get("TT_METAL_WATCHER"):
        raise RuntimeError("Run this timing control without profiler or watcher")
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        raise RuntimeError("This control requires strict trace allocation tracking")
    torch.set_num_threads(16)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    trace = None
    result = {
        "pass": False,
        "mesh_shape": [1, 4],
        "prompt_len": 128,
        "iterations_per_trial": args.iterations,
        "trials": args.trials,
        "boundary": "one real selected layer, persistent CCL, width-sharded residual, fixed device position/RoPE/cache/table; nonblocking trace loop and one final synchronization",
        "excludes": ["embedding", "terminal norm/head", "sampling", "position/RoPE advance", "host output"],
        "allocation_tracking": True,
        "diagnostic_acknowledgments": False,
        "records": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        gen = K2Generator(mesh, override_num_layers=2)
        prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 30)[:128]
        gen.generate(prompt, 4)
        gen._release_traces(drop_state=False)
        captured = []
        forwards = [layer.decode_forward for layer in gen.model.layers]

        def retain(forward):
            def call(x, **kwargs):
                output = forward(x, **kwargs)
                captured.append((x, kwargs.copy(), output))
                return output

            return call

        for layer, forward in zip(gen.model.layers, forwards):
            layer.decode_forward = retain(forward)
        state = gen.state
        # Direct model.decode leaves position/RoPE unchanged, so retained
        # inputs and their cache updates describe the same logical token.
        logits = gen.model.decode(
            state["tokens"],
            state["positions"],
            state["rope"],
            page_table=state["table"],
            kv_cache=state["cache"],
            batch_size=1,
        )
        ttnn.synchronize_device(mesh)
        del logits
        for layer, forward in zip(gen.model.layers, forwards):
            layer.decode_forward = forward
        assert len(captured) == 2

        for index, (layer, (raw_input, kwargs, eager_output)) in enumerate(zip(gen.model.layers, captured)):
            # Layer0 normally follows embedding. Use the inter-layer boundary
            # when weighting this class nine times; its one embedding reshard
            # remains in complete-path terminal accounting.
            x = ttnn.to_memory_config(raw_input, layer.decode_residual_memory)
            reference = read(eager_output)
            warm = layer.decode_forward(x, **kwargs)
            assert all(torch.equal(a, b) for a, b in zip(read(warm), reference))
            del warm
            with gen._capture() as captured_trace:
                output = layer.decode_forward(x, **kwargs)
            trace = captured_trace
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            actual = read(output)
            assert len(actual) == 4 and all(torch.equal(a, b) for a, b in zip(actual, reference))
            timings = []
            for _ in range(args.trials):
                ttnn.synchronize_device(mesh)
                started = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                timings.append((time.perf_counter() - started) * 1e6 / args.iterations)
            final = read(output)
            assert all(torch.equal(a, b) for a, b in zip(final, reference))
            row = {
                "layer": index,
                "policy": asdict(layer.policy),
                "position": read(kwargs["current_pos"])[0].reshape(-1).tolist(),
                "input_shape": list(x.shape),
                "input_memory": str(x.memory_config()),
                "cache_shape": list(kwargs["kv_cache"][0].shape),
                "trace_id": str(trace),
                "traced_us_per_layer": timings,
                "median_us": statistics.median(timings),
                "minimum_us": min(timings),
                "eager_and_replay_exact_all_ranks": True,
            }
            result["records"].append(row)
            print("SELECTED_LAYER", json.dumps(row), flush=True)
            ttnn.release_trace(mesh, trace)
            trace = None
            del output, x
        result["class_counts"] = [9, 27]
        result["weighted_stack_median_ms"] = (
            sum(count * row["median_us"] for count, row in zip(result["class_counts"], result["records"])) / 1000
        )
        result["weighted_stack_minimum_ms"] = (
            sum(count * row["minimum_us"] for count, row in zip(result["class_counts"], result["records"])) / 1000
        )

        # Independent terminal work from the real stack residual. This uses
        # the production final norm/gather/head/logit preparation and split
        # greedy sampler. Input and output buffers persist across replays;
        # cache/positions do not participate in this terminal-only trace.
        terminal_input = captured[-1][2]

        def terminal():
            logits = gen.model.sampler_logits(gen.model.final_logits(terminal_input, decode=True))
            return gen._sample(logits, strategy="split")

        terminal()
        terminal_reference = read(state["tokens"])
        with gen._capture() as captured_trace:
            terminal()
        trace = captured_trace
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        assert all(torch.equal(a, b) for a, b in zip(read(state["tokens"]), terminal_reference))
        timings = []
        for _ in range(args.trials):
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for _ in range(args.iterations):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - started) * 1e6 / args.iterations)
        assert all(torch.equal(a, b) for a, b in zip(read(state["tokens"]), terminal_reference))
        result["terminal"] = {
            "scope": "final norm, hidden gather, head, sampler-ready padding, split greedy and device seed advance",
            "excludes": ["embedding/RoPE lookup", "position/RoPE advance", "token history", "host output"],
            "traced_us": timings,
            "median_us": statistics.median(timings),
            "minimum_us": min(timings),
            "trace_id": str(trace),
            "eager_and_replay_exact_all_ranks": True,
        }
        print("SELECTED_TERMINAL", json.dumps(result["terminal"]), flush=True)
        ttnn.release_trace(mesh, trace)
        trace = None
        result["pass"] = True
    finally:
        if trace is not None:
            ttnn.synchronize_device(mesh)
            ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
