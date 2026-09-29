"""Warmed 128/128/1 end-to-end token-out and separately labeled model-only timing."""

import argparse
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")


def run(gen, output, *, model_load_seconds=None, token_output="buffered"):
    mesh = gen.mesh
    ttnn.CONFIG.throw_exception_on_fallback = True
    result = {"layers": gen.model.num_layers, "prompt_len": 128, "generation_len": 128, "batch_size": 1}
    result["precision_config"] = gen.model.precision_config
    result["runtime_policies"] = [asdict(layer.policy) for layer in gen.model.layers]
    result["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((DOC.parent.parent / "tt").glob("*.py"))
    }
    result["command"] = shlex.join(sys.orig_argv)
    result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if model_load_seconds is not None:
        result["model_load_seconds"] = model_load_seconds
    # Shape benchmark, not an instruction-following quality prompt.
    prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 20)[:128]
    assert len(prompt) == 128
    gen.generate(prompt, 128, token_output="per_token", trace_prefill=False)
    legacy_runs = []
    for _ in range(3):
        legacy_tokens = gen.generate(prompt, 128, token_output="per_token", trace_prefill=False)
        legacy_runs.append(gen.last_perf.copy())
    result["before_optimization_control"] = {
        "boundary": "same current model and cache, eager prefill and per-token async output reads",
        "runs": legacy_runs,
        "ttft_seconds": statistics.median(row["ttft_seconds"] for row in legacy_runs),
        "decode_ms_per_token": statistics.median(row["decode_seconds"] * 1000 / 127 for row in legacy_runs),
    }
    result["capacity_tokens_at_128_benchmark"] = gen.capacity
    gen.generate(prompt, 128, token_output=token_output)
    runs = {}
    outputs = {}
    for strategy in ["split", "argmax"]:
        runs[strategy] = []
        for repetition in range(3):
            outputs[strategy] = gen.generate(prompt, 128, strategy=strategy, token_output=token_output)
            runs[strategy].append(gen.last_perf.copy())
        if strategy == "argmax":
            assert outputs["split"] == outputs["argmax"], "Both comparisons must be greedy"
    result["runs"] = runs
    result["greedy_strategies_same_tokens"] = True
    assert legacy_tokens == outputs["split"]
    result["legacy_and_optimized_same_tokens"] = True
    result["selected"] = {
        "strategy": "split",
        "ttft_seconds": statistics.median(r["ttft_seconds"] for r in runs["split"]),
        "decode_tokens_per_second_per_user": statistics.median(
            r["decode_tokens_per_second_per_user"] for r in runs["split"]
        ),
        "decode_ms_per_token": statistics.median(r["decode_seconds"] / 127 * 1000 for r in runs["split"]),
    }
    for run in runs["split"]:
        counters = run["steady_state_counters"]
        assert counters["model_replays"] == counters["sampling_replays"] == 127
        if token_output == "buffered":
            assert counters.get("token_readbacks", 0) == 0
            assert counters["history_readbacks"] == 1 and counters["history_writes"] == 127
        else:
            assert counters["token_readbacks"] == 127 and counters["output_synchronizations"] == 1
        for key in [
            "token_refreshes",
            "position_refreshes",
            "rope_refreshes",
            "page_table_refreshes",
            "seed_refreshes",
            "logit_readbacks",
        ]:
            assert counters.get(key, 0) == 0
    gen.generate(prompt, 1)
    ttnn.synchronize_device(mesh)
    before = gen.counters.copy()
    start = time.perf_counter()
    for _ in range(128):
        gen.replay(sample=False)
    ttnn.synchronize_device(mesh)
    elapsed = time.perf_counter() - start
    result["model_only"] = {
        "boundary": "fixed-input traced model including head, no sampler or token output; positions advance",
        "iterations": 128,
        "seconds": elapsed,
        "ms_per_token": elapsed / 128 * 1000,
        "tokens_per_second_per_user": 128 / elapsed,
        "counters": dict(gen.counters - before),
    }
    # The same greedy request without caller output reads, including sampler
    # and device feedback. Only the final token is read after the timed window.
    gen.generate(prompt, 1)
    ttnn.synchronize_device(mesh)
    before = gen.counters.copy()
    start = time.perf_counter()
    for _ in range(127):
        gen.replay()
    ttnn.synchronize_device(mesh)
    elapsed = time.perf_counter() - start
    result["token_out_no_readback"] = {
        "boundary": "127 nonblocking model+split-sampler replays; one end synchronization; final read outside timing",
        "ms_per_token": elapsed / 127 * 1000,
        "tokens_per_second_per_user": 127 / elapsed,
        "counters": dict(gen.counters - before),
        "final_token_matches_generation": int(gen._read_tokens(batch=1)[0]) == outputs["split"][-1],
    }
    assert result["token_out_no_readback"]["final_token_matches_generation"]
    device = gen.generate(prompt, 8)
    host = gen.generate(prompt, 8, sampling_mode="host")
    assert host == device
    result["host_sampling_compatibility"] = {
        "tokens": host,
        "matches_device_greedy": True,
        "perf_not_used_for_headline": gen.last_perf,
    }
    long_prompt = (prompt * 32)[:4096]
    gen.generate(long_prompt, 128, token_output=token_output)
    gen.generate(long_prompt, 128, token_output=token_output)
    result["context4096_comparison"] = {
        "perf": gen.last_perf.copy(),
        "layer_stack_accounting": "Use current-policy account_full_model profile evidence; historical stage5 precision is rejected",
        "full_model_token_out_ms": gen.last_perf["decode_seconds"] / 127 * 1000,
    }
    result["pass"] = True
    result["throw_exception_on_fallback"] = ttnn.CONFIG.throw_exception_on_fallback
    result["persistent_collective_buffers"] = gen.model.pool.inventory()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--output", default=str(DOC / "performance.json"))
    parser.add_argument("--autoregressive", action="store_true")
    parser.add_argument("--token-output", choices=["buffered", "per_token"], default="buffered")
    parser.add_argument("--precision-config", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {"layers": args.layers, "prompt_len": 128, "generation_len": 128, "batch_size": 1}
    try:
        start = time.perf_counter()
        gen = build_generator(
            DOC.parent.parent, mesh, override_num_layers=args.layers, precision_config=args.precision_config
        )
        result["model_load_seconds"] = time.perf_counter() - start
        run(gen, args.output, model_load_seconds=result["model_load_seconds"], token_output=args.token_output)
        if args.autoregressive:
            from .run_full_quality import run as run_quality

            run_quality(gen, shared_suite=False)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
