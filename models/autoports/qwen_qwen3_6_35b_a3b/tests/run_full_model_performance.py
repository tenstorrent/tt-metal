# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Warm batch-one full-model performance, with persistent traced token feedback.

After lane-env.sh and python_env setup, run serialized with an external timeout:
 timeout 3600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance

Default: all 40 real layers, context 1024, 128-token AIME chat-template prefix,
128 generated tokens (first token from prefill plus 127 measured decode steps).
The unprofiled run reports both queued device token
production and caller-observed token delivery (one token readback per step).
It also measures 99 traced HF teacher-forcing steps after the full 161-token
reference prompt, including forced-input uploads but excluding sampling/readback.
--reduced selects real layers 0/3 for diagnostics.
--profile uses 16 steps/component, Tracy signposts and device profiler dumps;
its fenced timings include instrumentation and are not headline performance.
Example narrow capture (fresh directory, no watcher, devices assigned/idle):
 env -u TT_METAL_WATCHER -u TT_METAL_TRACE_ALLOC_TRACKING \\
   -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE TT_METAL_DEVICE_PROFILER=1 \\
   TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 TTNN_OP_PROFILER=1 \\
   TT_METAL_PROFILER_TRACE_TRACKING=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=5000 \\
   TT_METAL_PROFILER_MID_RUN_DUMP=1 \\
   timeout 3600 python -m tracy -p -v --no-web-server -o ARTIFACT/raw \\
   -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance \\
   --profile --reduced --output ARTIFACT/performance.json

Use native Tracy postprocessing and tt-perf-report with PERF_TOKEN_OUT,
PERF_MODEL_ONLY and PERF_SAMPLER_ONLY start/end signposts. The existing
postprocess_multichip_profile.py expects a different correctness schema and
must not be applied to this performance JSON unchanged.
"""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_LAYER_BUDGET_MS = 24.99350229
LINEAR_LAYER_MS = 0.719504474
FULL_ATTENTION_LAYER_MS = 0.340836807


def process_memory():
    """Linux host-process observations; no device access or profiler operation."""
    status = Path("/proc/self/status").read_text().splitlines()
    return {
        "pid": os.getpid(),
        "vm_rss_kib": next(int(line.split()[1]) for line in status if line.startswith("VmRSS:")),
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def install_profiler_drains(stack, *, ttnn, mesh, generator, result):
    """Bound eager profiler intervals without inserting reads into a trace.

    The supplied ExitStack owns every temporary patch, including exceptions.
    Captured model/sampling graphs remain unchanged; only host boundaries drain.
    """
    from unittest.mock import patch

    started = time.perf_counter()
    counts = {}

    def drain(label):
        # Query every queue before either host synchronization or profiler read.
        if ttnn.is_trace_capture_active(mesh):
            skipped = result.setdefault("profiler_drains_skipped_during_capture", {})
            skipped[label] = skipped.get(label, 0) + 1
            return
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        counts[label] = counts.get(label, 0) + 1
        result["profiler_drain_counts"] = dict(counts)
        result["preparation_profiler_drains"] = sum(counts.values())
        observation = {
            "event": "profiler_boundary_drain",
            "boundary": label,
            "profiler_drain_count": sum(counts.values()),
            "elapsed_seconds": time.perf_counter() - started,
            **process_memory(),
        }
        result["preparation_memory_latest"] = observation
        print(json.dumps(observation, allow_nan=False), flush=True)

    def after(original, label):
        def invoke(*values, **kwargs):
            output = original(*values, **kwargs)
            drain(label)
            return output

        return invoke

    executor = generator.prefill_executor
    for owner, name, label in (
        (executor, "_attention", "eager_prefill_attention"),
        (executor, "_moe", "eager_prefill_moe"),
        (generator, "_decode_into_sampler", "eager_model_forward"),
        (generator, "_sampling_forward", "eager_sampling_forward"),
        (generator.sampler, "precompile", "sampler_precompile"),
    ):
        stack.enter_context(patch.object(owner, name, after(getattr(owner, name), label)))

    begin_capture = ttnn.begin_trace_capture
    end_capture = ttnn.end_trace_capture

    def begin(device, *values, **kwargs):
        if device is mesh:
            drain("before_capture")
        return begin_capture(device, *values, **kwargs)

    def end(device, *values, **kwargs):
        output = end_capture(device, *values, **kwargs)
        if device is mesh:
            drain("after_capture")
        return output

    # Prepare contains both eager work and capture. Drain immediately before
    # begin and after end, never by patching an operation inside the graph.
    stack.enter_context(patch.object(ttnn, "begin_trace_capture", begin))
    stack.enter_context(patch.object(ttnn, "end_trace_capture", end))
    drain("after_construction")
    return drain


def counter_delta(after, before):
    return {key: after[key] - value for key, value in before.items()}


def fixture():
    path = ROOT / "doc/full_model/hf_reference/aime24_prompt_tokens.json"
    rendered = ROOT / "doc/full_model/hf_reference/rendered_aime24_prompt.txt"
    ids = json.loads(path.read_text())
    assert len(ids) == 161 and all(isinstance(token, int) and token >= 0 for token in ids)
    return ids[:128], {
        "kind": "performance-only prefix of pinned AIME24 HF-tokenizer chat-template input",
        "source_tokens": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rendered_source": str(rendered.relative_to(ROOT)),
        "rendered_sha256": hashlib.sha256(rendered.read_bytes()).hexdigest(),
        "original_length": len(ids),
        "prefix_length": 128,
        "token_ids": ids[:128],
        "qualitative_use": False,
        "note": "The truncated prefix is an activation fixture; its completions are not judged for coherence.",
    }


def measure_teacher_forcing(
    generator,
    mesh,
    forced_inputs,
    forced_token_ids,
    *,
    ttnn,
    device_only,
    position_value,
    prepared_entries,
    trace_ids,
):
    """Replay the existing model trace with retained, distinct host HF inputs.

    The caller prefills the reference prompt and initializes position once.
    Forced host uploads are timed; they are reference inputs, not sampled-token
    feedback. Only the model replay is inside the no-host-computation guard.
    """
    count = len(forced_inputs)
    assert count == len(forced_token_ids) == 99
    assert generator.prepared and generator.max_batch_size == 1 and generator.active_slots == (0,)
    assert not generator.host_sampling
    start_position = position_value()
    assert start_position == 161
    before = dict(generator.counters)
    ttnn.synchronize_device(mesh)
    started = time.perf_counter()
    for host_input in forced_inputs:
        # Each host tensor remains alive through the final synchronization.
        # Same-CQ ordering places this write after the preceding model replay.
        ttnn.copy_host_to_device_tensor(host_input, generator.tokens, cq_id=0)
        generator.counters["token_refreshes"] += 1
        with device_only():
            ttnn.execute_trace(mesh, generator.traces["model"], cq_id=0, blocking=False)
        generator.counters["model_replays"] += 1
    ttnn.synchronize_device(mesh)
    elapsed = time.perf_counter() - started
    counters = counter_delta(generator.counters, before)
    assert counters["token_refreshes"] == counters["model_replays"] == count
    for key in (
        "sampling_replays",
        "position_refreshes",
        "page_table_refreshes",
        "activity_refreshes",
        "readbacks",
        "captures",
        "releases",
    ):
        assert counters[key] == 0, ("teacher_forcing_logits_only", counters)
    # These observations are outside the measured interval and counter delta.
    end_position = position_value()
    final_input = generator._read_tokens([0])[0]
    assert end_position == 260
    assert final_input == forced_token_ids[98], "Final persistent input is not the last forced HF token"
    entries = int(mesh.num_program_cache_entries())
    assert entries == prepared_entries, ("Teacher-forcing program cache grew", entries, prepared_entries)
    assert {key: int(value) for key, value in generator.traces.items()} == trace_ids
    return {
        "steps": count,
        "elapsed_seconds": elapsed,
        "milliseconds_per_step": elapsed * 1000 / count,
        "tokens_per_second_per_user": count / elapsed,
        "loop_counter_delta": counters,
        "direct_model_trace_replays": count,
        "initial_position": start_position,
        "final_position": end_position,
        "forced_token_ids": forced_token_ids,
        "final_persistent_input": final_input,
        "final_input_matches_reference_index": 98,
        "program_cache_entries": entries,
        "runtime_device_only_guard": "pass for each model replay; required HF input uploads outside guard",
        "timing_scope": "queued HF input uploads and model traces, with synchronization at boundaries",
        "readback_location": "final persistent input and position verified after timing only",
        "includes": [
            "99 forced HF token uploads to the persistent UINT32 input on CQ0",
            "embedding, all selected decoder layers, final norm and LM head",
            "persistent device logits staging and device position advance",
            "host enqueue, counter and audit-guard overhead",
            "final synchronization",
        ],
        "excludes": [
            "reset and 161-token prefill",
            "initial position upload",
            "host reference tensor preparation",
            "sampling and sampled-token feedback",
            "logit/token/position readback",
            "compilation and capture",
        ],
        "semantics": "Traced HF teacher forcing to resident logits; not sampled generation, fixed-token replay or an accuracy score",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced", action="store_true", help="Real layers 0/3; diagnostic, not all-layer performance")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "doc/full_model/performance/result.json")
    args = parser.parse_args()
    steps = 16 if args.profile else 127
    prompt, prompt_metadata = fixture()
    assert args.context >= len(prompt) + steps
    if not args.profile:
        assert args.context >= 260, "HF teacher forcing requires context >= 260 (161 prompt + 99 decode steps)"
    # Configure before importing TTNN. Correctness/tracker and profiler runs
    # remain separate, matching the existing narrow multichip profile wrapper.
    if args.profile:
        assert not os.environ.get("TT_METAL_WATCHER"), "Watcher and profiling must run separately"
        assert os.environ.get("TT_METAL_DEVICE_PROFILER") == "1", "Launch through the documented profiler environment"
        assert os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") == "1", "Profiler reads must release host markers"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1", "Run trace allocation evidence separately"
    else:
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1", "Use --profile for instrumented runs"
        assert not os.environ.get("TT_METAL_WATCHER"), "Headline performance excludes watcher instrumentation"
        os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "0")
        os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
        assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] in ("0", "1")
        assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

    import torch

    teacher_entry = teacher_reference = None
    if not args.profile:
        from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_batched_teacher import load_fixture

        teacher_entry, teacher_reference = load_fixture(ROOT / "readiness_aime24_chat.refpt")

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import HF_MODEL, HF_REVISION, QwenModel

    if args.profile:
        from tracy import signpost
    else:

        def signpost(_name):
            pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), ROOT / "tests/runtime_audit.py", ROOT / "tests/multichip_functional.py"] + [
        ROOT / "tt" / name
        for name in (
            "model.py",
            "precision.py",
            "lm_head.py",
            "generator.py",
            "prefill.py",
            "prefill_transport.py",
            "slot_state.py",
            "multichip_decoder.py",
            "mesh_direct_recurrent_sum.py",
            "mesh_product_recurrent_sum.py",
            "mesh_state_slab_recurrence.py",
            "mesh_narrow_scan.py",
            "mesh_cached_scan.py",
        )
    ]
    if not args.profile:
        source_files.append(ROOT / "tests/run_full_model_batched_teacher.py")
    result = {
        "status": "RUNNING",
        "passed": False,
        "command": sys.argv,
        "time_unix": time.time(),
        "model": HF_MODEL,
        "revision": HF_REVISION,
        "layers": [0, 3] if args.reduced else list(range(40)),
        "context": args.context,
        "batch_size": 1,
        "steps_per_component": steps,
        "generated_tokens_including_prefill": steps + 1,
        "workload": "128 input / 128 output" if not args.profile else "128 input / 17 output diagnostic",
        "profile": args.profile,
        "headline_eligible": not args.profile
        and not args.reduced
        and os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "0",
        "fixture": prompt_metadata,
        "measurements": {},
        "full_40_layer_reference_budget_ms": FULL_LAYER_BUDGET_MS,
        "reference_budget_source": "doc/full_model/work_log.md: 30*0.719504474 + 10*0.340836807",
        "reference_budget_note": "Prior isolated decoder lower bound; excludes terminal, sampler and orchestration. Different context/workload is not an exact decomposition.",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
                "TTNN_OP_PROFILER",
                "TT_METAL_PROFILER_TRACE_TRACKING",
                "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY",
                "TT_METAL_PROFILER_MID_RUN_DUMP",
                "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT",
                "TT_METAL_PROFILER_CPP_POST_PROCESS",
                "TT_METAL_PROFILER_NO_CACHE_OP_INFO",
            )
        },
    }

    def save():
        if args.profile:
            result["host_process_memory"] = process_memory()
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = generator = model = None
    profiler_patches = ExitStack()
    profile_drain = None
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        model = QwenModel(mesh, layer_indices=[0, 3] if args.reduced else None, max_seq_len=args.context)
        result["execution_geometry"] = {
            "lm_head_mode": model.lm_head_mode,
            "lm_head": model.dram_head.geometry if model.dram_head is not None else None,
            "layers": [layer.execution_geometry for layer in model.layers],
        }
        generator = QwenGenerator(model, max_batch_size=1, host_sampling=False)
        result["runtime_precision"] = model.precision_summary()
        result["precision_construction"] = "QwenModel default resolution; no precision override"
        selected_path = ROOT / "doc/datatype_sweep/selected_precision_config.json"
        result["selected_precision_artifact"] = (
            {"path": str(selected_path), "sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest()}
            if selected_path.is_file()
            else None
        )
        generator.configure_sampling(temperature=0.0, top_k=1, top_p=1.0, seed=0)
        tokens = torch.tensor([prompt], dtype=torch.int64)
        position = torch.tensor([len(prompt)], dtype=torch.int32)
        if args.profile:
            profile_drain = install_profiler_drains(
                profiler_patches, ttnn=ttnn, mesh=mesh, generator=generator, result=result
            )

        def seed_request(*, captured=True):
            # Same device terminal-to-sampler handoff as QwenGenerator.generate.
            # Reset/configuration belongs to the caller, outside TTFT timing.
            logits = generator.prefill_forward(tokens, prompt_lens=[len(prompt)], device_output=True)[0]
            logits = ttnn.reshape(logits, (1, 1, 1, logits.shape[-1]))
            ttnn.copy(ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], 0), generator.sampler_logits)
            ttnn.deallocate(logits)
            generator._copy_host(position, generator.positions[0])
            generator.counters["position_refreshes"] += 1
            if captured:
                generator._sample()
            else:
                generator._sampling_forward()
            first = generator._read_tokens([0])[0]
            if args.profile:
                profile_drain("seed_terminal_handoff")
            return first

        def position_value():
            return int(ttnn.to_torch(ttnn.get_device_tensors(generator.positions[0])[0]).flatten()[0])

        # Warm the complete first-token handoff and all measurement readbacks
        # before capture, so they cannot introduce late program-cache entries.
        generator.reset()
        seed_request(captured=False)
        if args.profile:
            save()
        teacher_inputs = []
        teacher_token_ids = []
        if not args.profile:
            assert generator.tokens.dtype == ttnn.uint32
            teacher_token_ids = teacher_entry.generated_tokens[0, :99].tolist()
            for token in teacher_token_ids:
                ids = torch.zeros(tuple(generator.tokens.shape), dtype=torch.int32)
                ids[..., 0] = token
                teacher_inputs.append(
                    ttnn.from_torch(
                        ids,
                        dtype=generator.tokens.dtype,
                        layout=generator.tokens.layout,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    )
                )
            # Warm this host-copy shape and the full reference prompt before
            # capture. No persistent device tensor or trace is added.
            ttnn.copy_host_to_device_tensor(teacher_inputs[0], generator.tokens, cq_id=0)
            generator.counters["token_refreshes"] += 1
            generator.reset()
            for logits in generator.prefill_forward(
                teacher_entry.prompt_tokens.int(), prompt_lens=[161], device_output=True
            ):
                ttnn.deallocate(logits)
        position_value()
        generator.prepare(prefill_cases=((len(prompt), 0),))
        if args.profile:
            profile_drain("after_prepare_reset")
        prepared_entries = int(mesh.num_program_cache_entries())
        trace_ids = {key: int(value) for key, value in generator.traces.items()}
        captures = generator.counters["captures"]
        assert set(trace_ids) == {"model", "sampling"} and captures == 2
        result["trace_ids"] = trace_ids
        result["program_cache_entries_after_prepare"] = prepared_entries
        # Warm traced request execution once, then restore every mutable state.
        generator.reset()
        seed_request()
        if args.profile:
            for _ in range(4):
                with device_only():
                    generator.decode_forward(None, [0], refresh_inputs=False, read_tokens=False)
                profile_drain("warm_token_out_replay")
        else:
            with device_only():
                for _ in range(4):
                    generator.decode_forward(None, [0], refresh_inputs=False, read_tokens=False)
                ttnn.synchronize_device(mesh)
        generator.reset()
        ttnn.synchronize_device(mesh)
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)

        signpost("PERF_TTFT")
        started = time.perf_counter()
        first_token = seed_request()
        ttnn.synchronize_device(mesh)
        ttft_ms = (time.perf_counter() - started) * 1000
        signpost("PERF_TTFT_END")
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)
        result["ttft"] = {
            "milliseconds": ttft_ms,
            "first_token": first_token,
            "includes": [
                "public prefill",
                "device terminal-to-sampler handoff",
                "initial position upload",
                "traced sampling",
                "first token readback",
            ],
            "excludes": ["weights/loading", "prepare/compile/capture", "warmup", "reset", "tokenization"],
        }
        if args.profile:
            result["ttft"]["includes"].append("eager-component and terminal-handoff profiler drains/synchronizations")
            result["ttft"]["headline_eligible"] = False
            result["ttft"]["timing_scope"] = "Fenced profiler diagnostic including drain IO; not unprofiled TTFT"
        assert position_value() == len(prompt)
        save()

        def measure(name, invoke, *, expected_model, expected_sampling, expected_position_advance):
            before = dict(generator.counters)
            before_position = position_value()
            ttnn.synchronize_device(mesh)
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)
            label = "PERF_" + name.upper()
            elapsed = 0.0
            signpost(label)
            if args.profile:
                # Drain after each trace pair to prevent profiler ring overflow.
                # Dumps are outside each timed sample and outside device_only;
                # fenced/instrumented samples remain diagnostic-only.
                samples = []
                for _ in range(steps):
                    with device_only():
                        begin = time.perf_counter()
                        invoke()
                        ttnn.synchronize_device(mesh)
                        duration = time.perf_counter() - begin
                    samples.append(duration * 1000)
                    elapsed += duration
                    ttnn.ReadDeviceProfiler(mesh)
            else:
                samples = None
                with device_only():
                    begin = time.perf_counter()
                    for _ in range(steps):
                        invoke()
                    ttnn.synchronize_device(mesh)
                    elapsed = time.perf_counter() - begin
            signpost(label + "_END")
            counters = counter_delta(generator.counters, before)
            for key in (
                "token_refreshes",
                "position_refreshes",
                "page_table_refreshes",
                "activity_refreshes",
                "readbacks",
                "captures",
            ):
                assert counters[key] == 0, (name, counters)
            assert counters["model_replays"] == expected_model, (name, counters)
            assert counters["sampling_replays"] == expected_sampling, (name, counters)
            # First token/position read following the entire measured loop.
            final_token = generator._read_tokens([0])[0]
            after_position = position_value()
            assert after_position == before_position + expected_position_advance
            entries = int(mesh.num_program_cache_entries())
            assert entries == prepared_entries, ("Program cache grew after preparation", entries, prepared_entries)
            assert {key: int(value) for key, value in generator.traces.items()} == trace_ids
            assert generator.counters["captures"] == captures
            measurement = {
                "steps": steps,
                "elapsed_seconds": elapsed,
                "milliseconds_per_step": elapsed * 1000 / steps,
                "steps_per_second": steps / elapsed,
                "loop_counter_delta": counters,
                "initial_position": before_position,
                "final_position": after_position,
                "final_token": final_token,
                "readback_location": "after timed loop only",
                "runtime_device_only_guard": "pass",
                "program_cache_entries": entries,
                "samples_ms": samples,
                "timing_scope": (
                    "per-step fenced profiler diagnostic excluding dumps"
                    if args.profile
                    else "queued trace loop with synchronization at boundaries"
                ),
            }
            result["measurements"][name] = measurement
            print("FULL_MODEL_PERF", name, json.dumps(measurement), flush=True)
            save()
            return measurement

        token_out = measure(
            "token_out",
            lambda: generator.decode_forward(None, [0], refresh_inputs=False, read_tokens=False),
            expected_model=steps,
            expected_sampling=steps,
            expected_position_advance=steps,
        )
        assert token_out["initial_position"] == 128 and token_out["final_position"] == 128 + steps
        token_out["tokens_per_second_per_user"] = token_out["steps_per_second"]
        token_out[
            "semantics"
        ] = "Queued greedy token production with model/sampling traces and device-resident feedback; intermediate tokens are not delivered to the caller"
        token_out["headline_decode_metric"] = False

        if not args.profile:
            # The caller-visible metric includes every output read, while all
            # next-token feedback stays in the captured device buffers. Reads
            # use the exact layout warmed by seed_request before capture.
            generator.reset()
            observed_first_token = seed_request()
            assert observed_first_token == first_token, "Reset changed deterministic first token"
            observed_start = position_value()
            before = dict(generator.counters)
            delivered = []
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            for _ in range(steps):
                with device_only():
                    generator.decode_forward(None, [0], refresh_inputs=False, read_tokens=False)
                # This is output delivery only, never a host argmax or an
                # input to the next replay. Guard entry/exit overhead remains
                # included in caller-visible timing rather than subtracted.
                delivered.append(generator._read_tokens([0])[0])
            ttnn.synchronize_device(mesh)
            observed_elapsed = time.perf_counter() - begin
            observed_counters = counter_delta(generator.counters, before)
            observed_end = position_value()
            for key in (
                "token_refreshes",
                "position_refreshes",
                "page_table_refreshes",
                "activity_refreshes",
                "captures",
            ):
                assert observed_counters[key] == 0, ("token_out_observed", observed_counters)
            assert observed_counters["readbacks"] == steps
            assert observed_counters["model_replays"] == observed_counters["sampling_replays"] == steps
            assert observed_start == 128 and observed_end == 128 + steps
            assert delivered[-1] == token_out["final_token"], "Observed and queued feedback paths diverged"
            assert int(mesh.num_program_cache_entries()) == prepared_entries
            assert {key: int(value) for key, value in generator.traces.items()} == trace_ids
            assert generator.counters["captures"] == captures
            result["measurements"]["token_out_observed"] = {
                "steps": steps,
                "elapsed_seconds": observed_elapsed,
                "milliseconds_per_step": observed_elapsed * 1000 / steps,
                "tokens_per_second_per_user": steps / observed_elapsed,
                "headline_decode_metric": not args.reduced,
                "loop_counter_delta": observed_counters,
                "initial_position": observed_start,
                "final_position": observed_end,
                "first_token_before_loop": observed_first_token,
                "delivered_token_ids": delivered,
                "final_token": delivered[-1],
                "queued_final_token_and_position_equal": True,
                "runtime_device_only_guard": "pass for each model/sampler forward; output readback outside guard",
                "readback_location": "one token readback inside timing per step; no full-logit readback",
                "program_cache_entries": prepared_entries,
                "timing_scope": "caller-observed output delivery including readbacks, Python loop and audit guard overhead",
                "semantics": "Greedy resident-feedback decode; host observes tokens but never feeds them back",
            }
            print(
                "FULL_MODEL_PERF",
                "token_out_observed",
                json.dumps(result["measurements"]["token_out_observed"]),
                flush=True,
            )
            save()

        generator.reset()
        seed_request()
        model_only = measure(
            "model_only",
            lambda: ttnn.execute_trace(mesh, generator.traces["model"], cq_id=0, blocking=False),
            expected_model=0,
            expected_sampling=0,
            expected_position_advance=steps,
        )
        model_only["direct_model_trace_replays"] = steps
        model_only[
            "semantics"
        ] = "Synthetic fixed-token model-only replay with advancing positions and mutable cache; excludes sampling; NOT teacher forcing or generated-token throughput"

        generator.reset()
        seed_request()
        # Produce real decode logits once outside the sampler component timer.
        with device_only():
            ttnn.execute_trace(mesh, generator.traces["model"], cq_id=0, blocking=True)
        sampler_only = measure(
            "sampler_only",
            generator._sample,
            expected_model=0,
            expected_sampling=steps,
            expected_position_advance=0,
        )
        sampler_only[
            "semantics"
        ] = "Sampling trace over fixed real decode logits, including token merge and seed advance; excludes model compute"
        selected_budget = (LINEAR_LAYER_MS + FULL_ATTENTION_LAYER_MS) if args.reduced else FULL_LAYER_BUDGET_MS
        result["overhead_diagnostics"] = {
            "selected_stack_reference_budget_ms": selected_budget,
            "token_out_minus_reference_budget_ms": token_out["milliseconds_per_step"] - selected_budget,
            "model_only_minus_reference_budget_ms": model_only["milliseconds_per_step"] - selected_budget,
            "token_out_minus_model_only_ms": token_out["milliseconds_per_step"] - model_only["milliseconds_per_step"],
            "sampler_only_fraction_of_token_out": sampler_only["milliseconds_per_step"]
            / token_out["milliseconds_per_step"],
            "sampler_dominates_estimate": sampler_only["milliseconds_per_step"]
            > 0.5 * token_out["milliseconds_per_step"],
            "interpretation": "Separate replay diagnostics are not an additive attribution; inspect signposted op reports before assigning bottlenecks.",
        }
        if not args.profile:
            generator.reset()
            for logits in generator.prefill_forward(
                teacher_entry.prompt_tokens.int(), prompt_lens=[161], device_output=True
            ):
                ttnn.deallocate(logits)
            generator._copy_host(torch.tensor([161], dtype=torch.int32), generator.positions[0])
            generator.counters["position_refreshes"] += 1
            teacher_measurement = measure_teacher_forcing(
                generator,
                mesh,
                teacher_inputs,
                teacher_token_ids,
                ttnn=ttnn,
                device_only=device_only,
                position_value=position_value,
                prepared_entries=prepared_entries,
                trace_ids=trace_ids,
            )
            result["measurements"]["teacher_forcing_logits_only"] = teacher_measurement
            result["teacher_forcing"] = {
                "measured": True,
                "reference": teacher_reference,
                "measurement": "measurements.teacher_forcing_logits_only",
                "layers": result["layers"],
                "headline_eligible": not args.reduced,
                "note": "99 actual HF inputs; timed uploads included, sampling/readback excluded. Model-only remains a separate fixed-token diagnostic.",
            }
            print("FULL_MODEL_PERF", "teacher_forcing_logits_only", json.dumps(teacher_measurement), flush=True)
            save()
        else:
            result["teacher_forcing"] = {
                "measured": False,
                "note": "Supplemental HF teacher-forcing timing runs only without profiling; model-only is not teacher forcing.",
            }
        model.census("performance_after_measurement")
        result["memory_views"] = model.memory_views
        result["counters_before_release"] = dict(generator.counters)
        if not args.profile and not args.reduced:
            from models.autoports.qwen_qwen3_6_35b_a3b.tests.multichip_functional import byte_model

            positions = range(token_out["initial_position"], token_out["final_position"])
            decoder_bytes = sum(
                30 * byte_model(0, p)["logical_bytes_per_rank"] + 10 * byte_model(3, p)["logical_bytes_per_rank"]
                for p in positions
            ) / len(positions)
            terminal_bytes = {"lm_head": 2048 * 62080 * 17 / 16, "embedding_row": 512 * 2, "norm_weight": 2048 * 4}
            useful_bytes = decoder_bytes + sum(terminal_bytes.values())
            ideal_ms = useful_bytes / 512e6
            latency_ms = token_out["milliseconds_per_step"]
            result["functional_floor"] = {
                "mean_decoder_bytes_per_rank": decoder_bytes,
                "terminal_bytes_per_rank": terminal_bytes,
                "mean_useful_bytes_per_rank": useful_bytes,
                "bandwidth_gb_s_per_rank": 512,
                "ideal_ms": ideal_ms,
                "required_fraction": 0.10,
                "maximum_ms_per_token": ideal_ms / 0.10,
                "measured_ms_per_token": latency_ms,
                "achieved_fraction": ideal_ms / latency_ms,
                "passed": latency_ms <= ideal_ms / 0.10,
                "scope": "Separate stage gate; benchmark execution PASS alone does not imply this gate passed. No padding, rereads, intermediates or CCL in useful bytes.",
            }
        result["runtime_precision_after"] = model.precision_summary()
        assert result["runtime_precision_after"]["selected_policy"] == result["runtime_precision"]["selected_policy"]
        result["status"] = "PASS"
        result["passed"] = True
        print("FULL_MODEL_PERFORMANCE_OK", flush=True)
    except BaseException:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        # Restore host wrappers before releasing traces/closing the mesh, even
        # after any preparation, capture, execution or profiler-read failure.
        profiler_patches.close()
        if model is not None:
            result["memory_views"] = model.memory_views
        if generator is not None:
            generator.close()
            result["counters_after_release"] = dict(generator.counters)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        result["elapsed_seconds"] = time.time() - result["time_unix"]
        save()


if __name__ == "__main__":
    main()
