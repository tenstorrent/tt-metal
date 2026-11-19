# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed full-model token-out performance, optional model-only timing/profile.

Default: all 32 layers, B1, 128 input/128 generated tokens, three repetitions.
--profile-reduced: layers 0/3 and exactly one warmed model+sampling replay
inside PERF_DECODE/PERF_DECODE_END. Do not use reduced timings as full-model
throughput. External orchestration owns device serialization and timeouts.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--profile-batch", type=int, default=1)
    parser.add_argument("--model-only-steps", type=int, default=128)
    parser.add_argument("--profile-reduced", action="store_true")
    parser.add_argument("--decode-head", choices=("dram", "interleaved"), default="dram")
    parser.add_argument("--prefill-lengths", help="Additional warmed TTFT logical lengths, comma-separated")
    parser.add_argument("--build-kwargs-file", type=Path)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.repetitions, args.model_only_steps) <= 0 or args.generated_tokens < 2:
        parser.error("Positive prompt/repetition/step counts and at least two generated tokens are required")
    if args.prompt_tokens + max(args.generated_tokens, args.model_only_steps) > args.context:
        parser.error("Context must cover prefill plus the requested decode window")
    if args.profile_reduced and args.model_only:
        parser.error("Run the reduced one-step profile separately from model-only benchmarking")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "0")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
    from models.common.sampling.generator import SamplingParams

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
    gen = None
    prefill_execute = None
    result = {
        "status": "running",
        "regime": "serial",
        "batch": 1,
        "context": args.context,
        "profile_reduced": args.profile_reduced,
        "trace_allocation_tracking": trace_allocation_tracker.TRACE_ALLOC_TRACKING,
        "requires_exclusive_hardware_and_quiet_host": True,
        "token_out_runs": [],
        "teacher_forcing": None,
        "teacher_forcing_note": "Not measured here. Optional model-only uses a fixed device-resident token, not teacher forcing.",
    }
    try:
        if args.profile_reduced:
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_runtime import PrefillRuntime

            prefill_execute = PrefillRuntime._execute
            result["warmup_profiler_drains"] = 0

            def drain_prefill_profile(runtime, *positional, **keywords):
                output = prefill_execute(runtime, *positional, **keywords)
                # The complete preparation catalogue can overflow profiler DRAM.
                # Drain between eager profiles, outside every trace and timing window.
                ttnn.ReadDeviceProfiler(mesh)
                result["warmup_profiler_drains"] += 1
                return output

            PrefillRuntime._execute = drain_prefill_profile
        result["phase"] = "load_prepare_warm"
        build_kwargs = json.loads(args.build_kwargs_file.read_text()) if args.build_kwargs_file else {}
        gen = build_generator(
            None,
            mesh,
            layer_indices=[0, 3] if args.profile_reduced else None,
            max_batch_size=args.profile_batch if args.profile_reduced else 1,
            max_seq_len=args.context,
            decode_head=args.decode_head,
            **build_kwargs,
        )
        result["runtime_policy"] = gen.model.precision_summary()
        result["build_kwargs"] = build_kwargs
        result["layers"] = gen.model.layer_indices
        result["memory_loaded"] = gen.model.memory_view()
        example = (
            "Explain in simple terms how a computer learns patterns from examples and uses them to make predictions."
        )
        example_ids = gen.tokenizer.encode(example, add_special_tokens=False)
        assert example_ids, "Tokenizer returned no tokens for the timing stimulus"
        prompt = (example_ids * ((args.prompt_tokens + len(example_ids) - 1) // len(example_ids)))[: args.prompt_tokens]
        assert len(prompt) == args.prompt_tokens and all(0 <= token < gen.model.vocab_size for token in prompt)
        result["prompt"] = {
            "source": example,
            "construction": "Repeat tokenizer-produced IDs and truncate to exact requested length; timing stimulus, no quality claim",
            "token_ids": prompt,
        }
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=0)
        warm_count = 4 if args.profile_reduced else args.generated_tokens
        warm_output = gen.generate(prompt, warm_count, sampling_params=params)
        result["warmup_metrics"] = copy.deepcopy(gen.last_generation_metrics)
        if prefill_execute is not None:
            PrefillRuntime._execute = prefill_execute
        state = gen.states[(id(gen.cache), tuple(range(gen.max_batch_size)))]
        assert state.model_traces and state.sample_traces
        result["active_rows"] = list(state.batch_state.active)
        ttnn.synchronize_device(mesh)
        programs = mesh.num_program_cache_entries()
        mesh.set_program_cache_misses_allowed(False)

        if args.profile_reduced:
            from tracy import signpost

            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_PREFILL")
            gen.generate(prompt, 1, sampling_params=params)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            ttnn.ReadDeviceProfiler(mesh)
            gen.generate(prompt, 4, sampling_params=params)
            result["phase"] = "profile_one_replay"
            counters = dict(gen.counters)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            begin = time.perf_counter()
            gen._execute(state, host_sampling=False)
            ttnn.synchronize_device(mesh)
            elapsed_ms = (time.perf_counter() - begin) * 1000
            signpost("PERF_DECODE_END")
            ttnn.ReadDeviceProfiler(mesh)
            delta = {key: value - counters[key] for key, value in gen.counters.items()}
            assert delta["model_replays"] == delta["sampling_replays"] == 1
            assert all(
                delta[key] == 0
                for key in ("readbacks", "token_refreshes", "position_refreshes", "page_table_refreshes", "captures")
            )
            result["profile"] = {
                "start_signpost": "PERF_DECODE",
                "end_signpost": "PERF_DECODE_END",
                "model_replays": 1,
                "sampling_replays": 1,
                "counter_delta": delta,
                "synchronized_host_ms": elapsed_ms,
                "scope": "Reduced layers 0/3 plus real LM head/common sampler; no token readback in window",
            }
        else:
            result["phase"] = "warmed_token_out"
            for repetition in range(args.repetitions):
                output = gen.generate(prompt, args.generated_tokens, sampling_params=params)
                ttnn.synchronize_device(mesh)
                metrics = copy.deepcopy(gen.last_generation_metrics)
                delta = metrics["counter_delta"]
                assert output == warm_output, "Warmed greedy output changed between identical requests"
                assert metrics["trace_verified"] and not metrics["host_sampling"] and not metrics["teacher_forcing"]
                assert delta["model_replays"] == delta["sampling_replays"] == args.generated_tokens - 1
                assert delta["token_refreshes"] == delta["page_table_refreshes"] == delta["captures"] == 0
                assert delta["position_refreshes"] == 1 and delta["readbacks"] == args.generated_tokens
                assert mesh.num_program_cache_entries() == programs
                metrics.update(
                    repetition=repetition, output_token_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest()
                )
                result["token_out_runs"].append(metrics)
                print("PERF_TOKEN_OUT", repetition, json.dumps(metrics), flush=True)
            result["token_out_summary"] = {
                "median_ttft_ms": statistics.median(item["ttft_ms"] for item in result["token_out_runs"]),
                "median_decode_tokens_per_second_per_user": statistics.median(
                    item["decode_tokens_per_second_per_user"] for item in result["token_out_runs"]
                ),
                "median_decode_ms_per_token": statistics.median(
                    item["decode_ms_per_token"] for item in result["token_out_runs"]
                ),
                "prompt_tokens": args.prompt_tokens,
                "generated_tokens": args.generated_tokens,
                "decode_steps_per_run": args.generated_tokens - 1,
                "repetitions": args.repetitions,
                "includes_per_token_readback": True,
                "sampling": "device traced greedy",
            }
            forced = gen.generate(
                prompt,
                args.generated_tokens,
                sampling_params=params,
                next_input=lambda step, predicted: warm_output[step],
            )
            assert forced == warm_output
            result["teacher_forcing"] = copy.deepcopy(gen.last_generation_metrics)
            result["teacher_forcing_note"] = (
                "Same 128/128 workload and sampled predictions as token-out; teacher forces the warmed greedy sequence. "
                "Includes common traced sampling and host token uploads, matching readiness generator boundary."
            )

            result["resident_token_out_runs"] = []
            for repetition in range(args.repetitions):
                # The first two output tokens initialize the same public state
                # as streaming generation, outside the steady-state window.
                prefix = gen.generate(prompt, 2, sampling_params=params)
                assert prefix == warm_output[:2]
                steps = args.generated_tokens - 2
                if steps < 1:
                    break
                before = dict(gen.counters)
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                with ExitStack() as stack:
                    for name in ("to_torch", "from_torch", "copy_host_to_device_tensor", "synchronize_device"):
                        stack.enter_context(
                            patch.object(ttnn, name, side_effect=AssertionError(f"Host boundary: {name}"))
                        )
                    resident = gen.decode_resident(steps)
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - start
                delta = {key: value - before[key] for key, value in gen.counters.items()}
                assert delta["model_replays"] == delta["sampling_replays"] == steps
                assert all(
                    value == 0 for key, value in delta.items() if key not in ("model_replays", "sampling_replays")
                )
                last = int(ttnn.to_torch(ttnn.get_device_tensors(resident)[0]).reshape(-1)[0])
                position = int(ttnn.to_torch(ttnn.get_device_tensors(state.positions[0])[0]).item())
                assert last == warm_output[-1]
                assert position == args.prompt_tokens + args.generated_tokens - 1
                result["resident_token_out_runs"].append(
                    {
                        "repetition": repetition,
                        "steps": steps,
                        "ms_per_token": elapsed * 1000 / steps,
                        "tokens_per_second_per_user": steps / elapsed,
                        "counter_delta": delta,
                        "last_token_matches_streaming": True,
                        "final_position": position,
                        "scope": "nonblocking split model+sampling, no per-token host transfers; one final synchronization",
                    }
                )
            if result["resident_token_out_runs"]:
                result["resident_token_out_summary"] = {
                    "median_ms_per_token": statistics.median(
                        row["ms_per_token"] for row in result["resident_token_out_runs"]
                    ),
                    "median_tokens_per_second_per_user": statistics.median(
                        row["tokens_per_second_per_user"] for row in result["resident_token_out_runs"]
                    ),
                    "prompt_tokens": args.prompt_tokens,
                    "generated_tokens": args.generated_tokens,
                    "timed_steps": args.generated_tokens - 2,
                }

        if args.model_only:
            result["phase"] = "model_only_fixed_device_input"
            # Reset and fill once outside timing; no host input/logit work occurs
            # between trace replays. This is deliberately not teacher forcing.
            gen.reset()
            gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=gen.page_table,
                kv_cache=gen.cache,
                prompt_lens=[len(prompt)],
                return_all_logits=False,
            )
            fixed_token = 777
            gen._copy(torch.full((1, 1, 1, 32), fixed_token, dtype=torch.int32), state.tokens, "token_refreshes")
            gen._copy(
                torch.tensor([len(prompt)], dtype=torch.int32), state.positions[0], "position_refreshes", signed=True
            )
            counters = dict(gen.counters)
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            for _ in range(args.model_only_steps):
                ttnn.execute_trace(mesh, state.model_traces[False], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - begin
            assert gen.counters == counters, "Model-only window changed generator upload/sampling/readback counters"
            position = int(ttnn.to_torch(ttnn.get_device_tensors(state.positions[0])[0]).item())
            resident = ttnn.to_torch(ttnn.get_device_tensors(state.tokens)[0]).reshape(-1)
            assert position == len(prompt) + args.model_only_steps and int(resident[0]) == fixed_token
            result["model_only"] = {
                "scope": "Fixed device token; model trace advances position/cache; excludes sampling/readback; not teacher forcing",
                "replays": args.model_only_steps,
                "elapsed_seconds": elapsed,
                "tokens_per_second_per_user": args.model_only_steps / elapsed,
                "ms_per_token": elapsed * 1000 / args.model_only_steps,
                "final_position": position,
                "token_unchanged": True,
                "per_token_host_inputs_or_reads": 0,
            }
        assert mesh.num_program_cache_entries() == programs
        if args.prefill_lengths:
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.prefill_plan import plan_prefill

            result["prefill_runs"] = []
            for length in map(int, args.prefill_lengths.split(",")):
                assert 0 < length < args.context
                ids = (example_ids * math.ceil(length / len(example_ids)))[:length]
                expected = gen.generate(ids, 2, sampling_params=params)
                runs = []
                for _ in range(args.repetitions):
                    assert gen.generate(ids, 2, sampling_params=params) == expected
                    runs.append(copy.deepcopy(gen.last_generation_metrics))
                result["prefill_runs"].append(
                    {
                        "logical_length": length,
                        "runs": runs,
                        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in runs),
                        "physical_buckets": {
                            kind: [
                                segment.physical_bucket
                                for segment in plan_prefill(
                                    total_length=length, start=0, end=length, kind=kind
                                ).segments
                            ]
                            for kind in ("linear_attention", "full_attention")
                        },
                    }
                )
            assert mesh.num_program_cache_entries() == programs
        result.update(
            status="pass",
            phase="complete",
            program_count=programs,
            counters=dict(gen.counters),
            memory_final=gen.model.memory_view(),
        )
    except Exception as error:
        result.update(status="fail", error=repr(error))
        raise
    finally:
        if prefill_execute is not None:
            PrefillRuntime._execute = prefill_execute
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        mesh.set_program_cache_misses_allowed(True)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
