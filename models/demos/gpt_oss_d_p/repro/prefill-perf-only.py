#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Time the pinned GPT-OSS prefill runtime without loading or checking goldens."""
import argparse
import json
import math
import os
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk < 1024 or args.chunk % 256:
        raise ValueError("chunk must be at least 1024 and divisible by 256")
    if args.report.exists():
        raise ValueError("use a fresh report path")
    prompt = json.loads(args.prompt.read_text())
    ids = prompt["token_ids"]
    if len(ids) != prompt["input_tokens"] or args.chunk > len(ids):
        raise ValueError("prompt length or chunk size mismatch")
    count = len(ids)
    chunks = math.ceil(count / args.chunk)
    processed = chunks * args.chunk
    padded = ids + [0] * (processed - count)
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    events = []
    start = time.perf_counter()

    def event(name, **values):
        item = dict(
            event=name,
            utc=datetime.now(timezone.utc).isoformat(),
            epoch=time.time(),
            elapsed_s=time.perf_counter() - start,
            **values,
        )
        events.append(item)
        print("PREFILL_PERF_EVENT " + json.dumps(item), flush=True)
        with args.report.with_suffix(".events.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")

    event("process_started", input_tokens=count, processed_tokens=processed, chunk=args.chunk)
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    import ttnn

    if ttnn.get_num_devices() != 32:
        raise RuntimeError("exactly 32 visible devices required")
    event("imports_finished")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
    try:
        event("mesh_opened")
        model_args = ModelArgs(mesh_device=mesh)
        hf_config = model_args.hf_config
        if hf_config.num_hidden_layers != 36:
            raise ValueError("full 36-layer GPT-OSS required")
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        event("weights_started", cache=str(cache), weights_from_cache=os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE"))
        state = (
            {} if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1" else ModelArgs.load_state_dict(model_args.weights_path)
        )
        event("weights_loaded")
        cfg = TtPrefillRuntimeConfig(
            num_layers=36,
            max_seq_len=processed,
            mesh_shape=(4, 8),
            default_chunk_size=args.chunk,
            num_users=1,
            expert_weight_dtype=ttnn.bfloat4_b,
            cache_dtype=ttnn.bfloat8_b,
            weight_cache_path=cache,
            owns_kv_cache=True,
            topology=ttnn.Topology.Linear,
            bounded_sliding_kv_cache=False,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state, cfg)
        del state
        event("model_built")
        runtime.compile()
        ttnn.synchronize_device(mesh)
        event("compile_finished")
        runs = []
        for index in range(4):
            event("iteration_started", index=index, measured=index > 0)
            t0 = time.perf_counter()
            for chunk_index in range(chunks):
                a = chunk_index * args.chunk
                inp = runtime.make_chunk_input(padded[a : a + args.chunk])
                runtime.prefill_chunk(inp, slot_id=0, actual_start=a, actual_end=min(a + args.chunk, count))
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - t0
            run = dict(
                index=index,
                measured=index > 0,
                prefill_s=elapsed,
                total_s=elapsed,
                input_tokens=count,
                processed_tokens=processed,
                prefill_tps=count / elapsed,
                processed_tps=processed / elapsed,
            )
            runs.append(run)
            event("iteration_finished", **run)
        measured = [r["prefill_s"] for r in runs if r["measured"]]
        median = statistics.median(measured)
        report = dict(
            status="DONE",
            numerical_validation="not run, as requested",
            layers=36,
            slots=1,
            active_slots=1,
            input_tokens=count,
            processed_tokens=processed,
            chunk=args.chunk,
            prompt_sha256=prompt["sha256_token_ids"],
            warmup_s=runs[0]["prefill_s"],
            runs=runs,
            mean_s=statistics.mean(measured),
            variance_s2=statistics.variance(measured),
            median_s=median,
            median_tps=count / median,
            median_processed_tps=processed / median,
            events=events,
            cache=str(cache),
            fabric="FABRIC_1D",
            topology="linear",
        )
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.report)
        print("PREFILL_PERF_RESULT " + json.dumps(report), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    event("mesh_closed")


if __name__ == "__main__":
    main()
