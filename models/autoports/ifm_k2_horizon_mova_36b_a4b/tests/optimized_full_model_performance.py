# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-workload optimized-stage timing, including queued split token-out replay."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]


def memory(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        key: getattr(view, key)
        for key in ("num_banks", "total_bytes_per_bank", "total_bytes_allocated_per_bank", "total_bytes_free_per_bank")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--mode", choices=("perf", "sampler", "profile", "capacity"), default="perf")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "profile" and args.layers != 2:
        ap.error("Only the reduced real-layer variant may be profiled")
    if args.mode == "capacity" and args.context != 524288:
        ap.error("Capacity gate preserves the advertised 524288 context")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    result = dict(
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")},
        mode=args.mode,
        layers=args.layers,
        context=args.context,
        trace_alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        program_cache_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0",
    )
    try:
        gen = build_generator(ROOT, mesh, override_num_layers=args.layers, max_seq_len=args.context)
        result["constructed_memory"] = memory(mesh)
        print("CONSTRUCTED", result["constructed_memory"], flush=True)
        if args.mode == "profile":
            # Drain warmup markers between physical variants; only the final
            # signposted replay is measured. Never read inside a trace capture.
            original_prefill = gen.model.prefill_chunk
            original_sample = gen.sampler._run_sampling

            def drain_after(function):
                def wrapped(*a, **kw):
                    value = function(*a, **kw)
                    ttnn.ReadDeviceProfiler(mesh)
                    return value

                return wrapped

            ttnn.ReadDeviceProfiler(mesh)
            gen.model.prefill_chunk = drain_after(original_prefill)
            gen.sampler._run_sampling = drain_after(original_sample)
            gen.prepare()
            gen.model.prefill_chunk = original_prefill
            gen.sampler._run_sampling = original_sample
        gen.capture()
        if args.mode == "profile":
            ttnn.ReadDeviceProfiler(mesh)
        result["captured_memory"] = memory(mesh)
        print("CAPTURED", result["captured_memory"], flush=True)
        prompt = json.loads((ROOT / "doc/full_model/reference_metadata.json").read_text())["prompt_tokens"][:128]
        if args.mode == "capacity":
            length = 524275
            prompt = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
            original = gen._replay_prefill
            progress = [0]

            def observed(*a, **kw):
                value = original(*a, **kw)
                progress[0] += 1
                if progress[0] % 8 == 0:
                    ttnn.synchronize_device(mesh)
                    print("PREFILL_CHUNKS", progress[0], flush=True)
                return value

            gen._replay_prefill = observed
            start = time.perf_counter()
            out = gen.generate(prompt, 13, stop_on_eos=False)
            result["nonaligned_full_prefill"] = dict(
                length=length, generation=13, tokens=out, seconds=time.perf_counter() - start
            )
            tail = prompt + [100] * 13
            final = gen.prefill_forward(
                torch.tensor([tail]),
                page_table=gen.page_table_host,
                kv_cache=gen.kv_cache,
                start_pos=[length],
                prompt_lens=[524288],
                sampling_mode="device",
            )
            result["full_extent_continuation"] = dict(start=length, end=524288, token=int(final[0, 0]))
        else:
            gen.generate(prompt, 2 if args.mode == "profile" else 4, stop_on_eos=False)
            if args.mode == "perf":
                timestamps = []
                before = gen.counters.copy()
                start = time.perf_counter()
                out = gen.generate(
                    prompt, 128, stop_on_eos=False, on_token=lambda step, token: timestamps.append(time.perf_counter())
                )
                result["token_out"] = dict(
                    prompt_tokens=128,
                    generated_tokens=len(out),
                    ttft_ms=(timestamps[0] - start) * 1000,
                    decode_ms=(timestamps[-1] - timestamps[0]) * 1000 / 127,
                    decode_tps=127 / (timestamps[-1] - timestamps[0]),
                    counters={k: gen.counters[k] - before[k] for k in gen.counters},
                )
                gen.generate(prompt, 1, stop_on_eos=False)
                before = gen.counters.copy()
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(127):
                    gen.replay()
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - start
                result["queued_token_out"] = dict(
                    prompt_tokens=128,
                    generated_tokens=128,
                    decode_ms=elapsed * 1000 / 127,
                    decode_tps=127 / elapsed,
                    counters={k: gen.counters[k] - before[k] for k in gen.counters},
                    boundary_synchronizations=2,
                    per_token_synchronizations=0,
                )
                final = int(gen.read_tokens()[0])
                assert final == out[-1], (final, out[-1])
                result["queued_token_out"]["last_token_matches_readback_loop"] = True
                result["prefill_lengths"] = []
                for length in (127, 128, 129, 156, 2049):
                    ids = (prompt * ((length + 127) // 128))[:length]
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        gen.generate(ids, 1, stop_on_eos=False)
                        times.append((time.perf_counter() - start) * 1000)
                    result["prefill_lengths"].append(dict(logical_length=length, ttft_ms=times))
                # Explicitly separate the no-sampling, no-readback trace comparison.
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(128):
                    gen.replay(sample=False)
                ttnn.synchronize_device(mesh)
                result["logits_only_trace_ms"] = (time.perf_counter() - start) * 1000 / 128
            elif args.mode == "sampler":
                result["greedy_strategies"] = []
                expected = gen.read_logits(gen.logits)[0, 0, 0].argmax().item()
                for force_argmax in (False, True, False):
                    gen.sampler.tt_sampling._allow_force_argmax_sampling = force_argmax
                    gen.sampler.reset_sampling_params(gen.formatted_sampling_params)
                    tid, _ = gen._capture_sampler()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    token = int(gen.read_tokens()[0])
                    assert token == expected, (force_argmax, token, expected)
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(100):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    result["greedy_strategies"].append(
                        dict(
                            force_argmax=force_argmax,
                            token=token,
                            latency_ms=(time.perf_counter() - start) * 10,
                            k=1,
                            p=0,
                            inverse_temperature=1,
                        )
                    )
            else:
                from tracy import signpost

                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
                signpost("PERF_DECODE")
                start = time.perf_counter()
                gen.replay()
                gen.read_tokens()
                ttnn.synchronize_device(mesh)
                result["profile_decode_host_ms"] = (time.perf_counter() - start) * 1000
                signpost("PERF_DECODE_END")
                ttnn.ReadDeviceProfiler(mesh)
                signpost("PERF_PREFILL")
                start = time.perf_counter()
                gen.prefill_forward(
                    torch.tensor([prompt]),
                    page_table=gen.page_table_host,
                    kv_cache=gen.kv_cache,
                    prompt_lens=[128],
                    sampling_mode="device",
                )
                ttnn.synchronize_device(mesh)
                result["profile_prefill_host_ms"] = (time.perf_counter() - start) * 1000
                signpost("PERF_PREFILL_END")
        result["final_memory"] = memory(mesh)
        result["counters"] = dict(gen.counters)
        result["model_trace"] = str(gen.model_trace)
        result["prefill_traces"] = {str(k): str(v) for k, v in gen.prefill_traces.items()}
        result["sampling_traces"] = {str(k): str(v[0]) for k, v in gen.sampling_traces.items()}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
