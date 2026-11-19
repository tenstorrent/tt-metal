# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the real terminal path, or time warmed full-model generation.

Reduced profile (one real layer of each kind, terminal norm/head and sampler):
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_TRACE_ALLOC_TRACKING=0 \\
    python -m tracy -r -p --no-web-server -o <tracy-dir> -m \\
    models.autoports.qwen_qwen3_6_27b.tests.profile_full_terminal --output <json>
Full-stack, uninstrumented timing (128 prompt / 128 generated tokens):
  TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=0 \\
    python -m models.autoports.qwen_qwen3_6_27b.tests.profile_full_terminal \\
    --performance --all-layers --output <json>

Render the copied Tracy ops CSV with advice enabled, for each PERF_PREFILL,
PERF_DECODE and optional PERF_LOGITS_ONLY/PERF_SAMPLING window:
  tt-perf-report <ops.csv> --start-signpost PERF_DECODE \\
    --end-signpost PERF_DECODE_END --csv <report.csv> > <report.console.log>
  tt-perf-report <ops.csv> --start-signpost PERF_DECODE \\
    --end-signpost PERF_DECODE_END --no-summary > <report.txt>

The reduced profiler prepares only length128 to bound profiler output. Full
performance uses the complete public preparation contract. Neither replaces
the separate trace-allocation acceptance probe. No full-stack profiler run,
hardware reset, full-logits readback or host feedback.
Test-local setup hooks drain at eager helper boundaries and after completed
captures; they are removed before timing. Window drains occur after end signposts.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_model_evidence import source_identity
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import MODEL_ID, REVISION, Model

PROMPT_LENGTH = 128
GENERATED_TOKENS = 128
CONTEXT = 2048


def enabled(name):
    return os.environ.get(name, "0").lower() not in ("", "0", "false", "off")


def delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0) for key in sorted(before.keys() | after.keys())}


def drain_profiler(mesh, label, report):
    """Flush bounded profiler work outside capture and measurement windows."""
    assert not ttnn.is_trace_capture_active(mesh), "Profiler drain cannot run inside a captured graph"
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    report.setdefault("profiler_drains", []).append(label)
    print(f"PROFILER_DRAIN {label}", flush=True)


@contextmanager
def setup_profiler_drains(gen, report):
    """Test-local instrumentation, removed before any measured signpost.

    Preparation executes many eager programs before its first trace. Draining
    only once after warmup can already be too late for the profiler DRAM ring.
    Bound each layer/head/terminal and generator phase instead. Captured graph
    calls skip the hook; _capture drains after end_trace_capture has completed.
    """
    hooks = [
        (gen.prefill_single, "_layer_chunk"),
        (gen.prefill_single, "_attention_head_step"),
        (gen.model, "terminal"),
        *[(gen, name) for name in ("_decode_graph", "_sample_graph", "_reset_graph", "_capture", "_execute")],
    ]
    originals = []
    try:
        for owner, name in hooks:
            original = getattr(owner, name)
            originals.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))
            label = f"setup:{type(owner).__name__}.{name}"

            @wraps(original)
            def wrapped(*args, _original=original, _label=label, **kwargs):
                result = _original(*args, **kwargs)
                if not ttnn.is_trace_capture_active(gen.mesh):
                    drain_profiler(gen.mesh, _label, report)
                return result

            setattr(owner, name, wrapped)
        yield
    finally:
        for owner, name, had_local, original in reversed(originals):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def snapshot(gen):
    return {
        "trace_ids": {key: str(value) for key, value in gen.traces.items()},
        "captures": gen.counters["captures"],
        "program_cache_entries": int(gen.mesh.num_program_cache_entries()),
        "addresses": {
            key: int(getattr(gen, key).buffer_address()) for key in ("tokens", "positions", "rope_positions", "logits")
        },
    }


def check_counters(counters, *, decodes, samples, reads, setup_positions=0):
    for key in ("full_logits_readbacks", "token_refreshes", "host_sampling_refreshes", "teacher_forcing_refreshes"):
        assert counters.get(key, 0) == 0, (key, counters)
    assert counters.get("decode_replays", 0) == decodes, counters
    assert counters.get("sample_replays", 0) == samples, counters
    assert counters.get("token_readbacks", 0) == reads, counters
    assert counters.get("position_refreshes", 0) == setup_positions, counters
    assert counters.get("rope_refreshes", 0) == setup_positions, counters
    assert counters.get("page_table_refreshes", 0) == 0, counters


def begin_request(gen, prompt):
    """Same first-token operations and order as QwenGenerator.generate."""
    gen.reset()
    gen.configure_sampling(temperature=0.0, top_k=1, top_p=0.0, seed=1700)
    gen._prefill_last_into_logits(prompt)
    gen._reset_rng(torch.tensor([len(prompt) - 1]))
    gen._execute("sample")
    return gen._read_tokens()[0]


def set_decode_position(gen):
    # One request-boundary update, never a per-token feedback operation.
    value = torch.tensor([PROMPT_LENGTH], dtype=torch.int32)
    gen._write(value, gen.positions, "position_refreshes")
    gen._write(value, gen.rope_positions, "rope_refreshes")


def timed_logits_only(gen, prompt, steps, *, signpost=None, drain=None):
    begin_request(gen, prompt)
    set_decode_position(gen)
    ttnn.synchronize_device(gen.mesh)
    if drain:
        drain("before_logits_only")
    before = dict(gen.counters)
    if signpost:
        signpost("PERF_LOGITS_ONLY")
    start = time.perf_counter()
    for _ in range(steps):
        gen._execute("decode")
    ttnn.synchronize_device(gen.mesh)
    seconds = time.perf_counter() - start
    if signpost:
        signpost("PERF_LOGITS_ONLY_END")
    if drain:
        drain("after_logits_only")
    counters = delta(before, dict(gen.counters))
    check_counters(counters, decodes=steps, samples=0, reads=0)
    return {
        "seconds": seconds,
        "steps": steps,
        "tokens_per_second_per_user": steps / seconds,
        "counters": counters,
        "workload": "Timing-only: fixed first generated input token, advancing device positions; no sampling or readback",
        "teacher_forcing": False,
        "synchronization": "One end-of-window device synchronization",
    }


def profile(gen, prompt, args, report):
    from tracy import signpost

    # Drain preparation/warmup markers before the bounded profile windows.
    drain_profiler(gen.mesh, "before_profile", report)
    print("PERF_PREFILL_BEGIN", flush=True)
    before = dict(gen.counters)
    signpost("PERF_PREFILL")
    start = time.perf_counter()
    first_token = begin_request(gen, prompt)
    seconds = time.perf_counter() - start
    signpost("PERF_PREFILL_END")
    drain_profiler(gen.mesh, "after_prefill", report)
    report["prefill"] = {
        "ttft_seconds": seconds,
        "first_token": first_token,
        "includes": "Request reset/configuration, prefill, final norm/LM head, traced greedy sampling and token readback",
        "counters": delta(before, dict(gen.counters)),
    }
    set_decode_position(gen)
    ttnn.synchronize_device(gen.mesh)
    before = dict(gen.counters)
    print(f"PERF_DECODE_BEGIN steps={args.profile_steps}", flush=True)
    signpost("PERF_DECODE")
    start = time.perf_counter()
    tokens = []
    for _ in range(args.profile_steps):
        if args.resident:
            gen.decode_forward(
                None, None, page_table=gen.page_host, kv_cache=gen.cache, reload_inputs=False, read_from_device=False
            )
        else:
            gen._execute("decode")
            gen._execute("sample")
            tokens.append(gen._read_tokens()[0])
    if args.resident:
        ttnn.synchronize_device(gen.mesh)
    seconds = time.perf_counter() - start
    signpost("PERF_DECODE_END")
    drain_profiler(gen.mesh, "after_decode", report)
    counters = delta(before, dict(gen.counters))
    check_counters(
        counters,
        decodes=args.profile_steps,
        samples=args.profile_steps,
        reads=0 if args.resident else args.profile_steps,
    )
    report["token_out"] = {
        "seconds": seconds,
        "steps": args.profile_steps,
        "tokens_per_second_per_user": args.profile_steps / seconds,
        "tokens": tokens,
        "counters": counters,
        "includes": "Traced model including final norm/LM head and traced greedy sampling",
        "per_token_readback": not args.resident,
    }
    if args.separate_phases:
        print("PERF_LOGITS_ONLY_BEGIN", flush=True)
        report["logits_only"] = timed_logits_only(
            gen,
            prompt,
            args.profile_steps,
            signpost=signpost,
            drain=lambda label: drain_profiler(gen.mesh, label, report),
        )
        before = dict(gen.counters)
        print("PERF_SAMPLING_BEGIN", flush=True)
        signpost("PERF_SAMPLING")
        start = time.perf_counter()
        for _ in range(args.profile_steps):
            gen._execute("sample")
            if not args.resident:
                gen._read_tokens()
        if args.resident:
            ttnn.synchronize_device(gen.mesh)
        seconds = time.perf_counter() - start
        signpost("PERF_SAMPLING_END")
        drain_profiler(gen.mesh, "after_sampling", report)
        counters = delta(before, dict(gen.counters))
        check_counters(
            counters, decodes=0, samples=args.profile_steps, reads=0 if args.resident else args.profile_steps
        )
        report["sampling_only"] = {
            "seconds": seconds,
            "steps": args.profile_steps,
            "counters": counters,
            "workload": "Same resident logits sampled repeatedly",
            "per_token_readback": not args.resident,
        }
    drain_profiler(gen.mesh, "profile_complete", report)


def performance(gen, prompt, args, report, save):
    report["runs"] = []
    for repeat in range(args.repeats):
        print(f"PERFORMANCE_GENERATE repeat={repeat} prompt=128 generated=128", flush=True)
        ttnn.synchronize_device(gen.mesh)
        before = dict(gen.counters)
        start = time.perf_counter()
        tokens = gen.generate(prompt, GENERATED_TOKENS, temperature=0.0, seed=1700, ignore_eos=True)
        elapsed = time.perf_counter() - start
        counters = delta(before, dict(gen.counters))
        check_counters(counters, decodes=127, samples=128, reads=128, setup_positions=1)
        assert len(tokens) == GENERATED_TOKENS
        run = {
            **gen.last_timings,
            "total_seconds": elapsed,
            "decode_steps": GENERATED_TOKENS - 1,
            "token_out_tokens_per_second_per_user": (GENERATED_TOKENS - 1) / gen.last_timings["decode_seconds"],
            "tokens": tokens,
            "counters": counters,
            "decode_window": "Generator timing includes one initial position/RoPE setup, then127 traced model/sample/token-read steps",
        }
        print(f"PERFORMANCE_LOGITS_ONLY repeat={repeat} steps=127", flush=True)
        run["logits_only"] = timed_logits_only(gen, prompt, GENERATED_TOKENS - 1)
        report["runs"].append(run)
        save()
    report["median"] = {
        key: statistics.median(run[key] for run in report["runs"])
        for key in ("ttft_seconds", "decode_seconds", "token_out_tokens_per_second_per_user")
    }
    report["median"]["logits_only_tokens_per_second_per_user"] = statistics.median(
        run["logits_only"]["tokens_per_second_per_user"] for run in report["runs"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--performance", action="store_true", help="Uninstrumented prompt128/gen128 performance")
    parser.add_argument("--all-layers", action="store_true", help="All64 real layers, only with --performance")
    parser.add_argument("--profile-steps", type=int, choices=(2, 3), default=3)
    parser.add_argument("--separate-phases", action="store_true", help="Add bounded logits-only and sampling windows")
    parser.add_argument("--resident", action="store_true", help="Profile public token-out without per-step reads")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.all_layers and not args.performance:
        parser.error("Full-stack profiler collection is prohibited; use --performance --all-layers")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if enabled("TT_METAL_WATCHER") or enabled("TT_METAL_TRACE_ALLOC_TRACKING"):
        parser.error("Disable Watcher and trace allocation tracking for timing; correctness acceptance is separate")
    if args.performance and enabled("TT_METAL_DEVICE_PROFILER"):
        parser.error("--performance requires TT_METAL_DEVICE_PROFILER=0 and a plain Python invocation, outside Tracy")
    tracy_state = sys.modules.get("tracy.tracy_state")
    if args.performance and (
        sys.getprofile() is not None or sys.gettrace() is not None or getattr(tracy_state, "doPartial", False)
    ):
        parser.error("--performance must run outside Tracy and Python tracing/profiling")
    if not args.performance and not enabled("TT_METAL_DEVICE_PROFILER"):
        parser.error("Profile mode requires TT_METAL_DEVICE_PROFILER=1 and the documented Tracy invocation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_sha256": source_identity(),
        "passed": False,
        "mode": "uninstrumented_performance" if args.performance else "reduced_instrumented_profile",
        "provenance": {
            "model": MODEL_ID,
            "revision": REVISION,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "environment": {key: value for key, value in os.environ.items() if key.startswith("TT_METAL_")},
        },
        "batch": 1,
        "length": PROMPT_LENGTH,
        # Primary performance uses advertised capacity, including its reset and
        # attention-window costs. Reduced terminal profiling keeps its bounded
        # diagnostic allocation; its timing is not the published full-model rate.
        "context_allocation": 262144 if args.performance else CONTEXT,
        "diagnostic_preparation_lengths": None if args.performance else [PROMPT_LENGTH],
        "acceptance_ready": False,
        "limitations": [
            "Separate general-length trace/allocation acceptance is required",
            "Synthetic deterministic prompt; no qualitative or accuracy claim",
            "Logits-only timing is not teacher-forcing accuracy or autonomous token-out throughput",
        ],
    }
    if not args.performance:
        report["limitations"].append("Profiler-instrumented timings are diagnostic, not full-model performance")
        report["limitations"].append("Reduced profiling prepares length128 only to bound profiler output")

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    start = time.perf_counter()
    try:
        model = Model(mesh, checkpoint=args.model_dir, layer_indices=None if args.all_layers else [0, 3])
        if not args.performance:
            drain_profiler(mesh, "model_constructed", report)
        report["layers"] = model.layer_indices
        report["memory_weights"] = model.loaded_memory
        gen = QwenGenerator(model, batch_size=1, max_seq_len=report["context_allocation"])
        if not args.performance:
            drain_profiler(mesh, "generator_constructed", report)
        report["memory_generator_allocated"] = model.memory_view()
        prompt = ((torch.arange(PROMPT_LENGTH) * 37 + 101) % model.config.vocab_size).tolist()
        print(f"TERMINAL_PREPARE full={args.performance}", flush=True)
        prepare_start = time.perf_counter()
        with nullcontext() if args.performance else setup_profiler_drains(gen, report):
            if not args.performance:
                gen.prefill_single.prepare(_diagnostic_lengths=[PROMPT_LENGTH])
            gen.prepare(prefill_lengths=(PROMPT_LENGTH,))
            ttnn.synchronize_device(mesh)
            report["preparation_seconds"] = time.perf_counter() - prepare_start
            print("TERMINAL_WARMUP prompt=128 generated=4", flush=True)
            gen.generate(prompt, 4, temperature=0.0, seed=1700, ignore_eos=True)
            ttnn.synchronize_device(mesh)
        if not args.performance:
            drain_profiler(mesh, "setup_hooks_removed", report)
        report["memory_prepared"] = model.memory_view()
        report["trace_before"] = snapshot(gen)
        save()
        if args.performance:
            performance(gen, prompt, args, report, save)
        else:
            profile(gen, prompt, args, report)
        ttnn.synchronize_device(mesh)
        report["trace_after"] = snapshot(gen)
        assert (
            report["trace_before"] == report["trace_after"]
        ), "Trace identities, buffers or program-cache count changed"
        report["memory_after"] = model.memory_view()
        report["counters"] = dict(gen.counters)
        report["passed"] = True
    except BaseException as error:
        report["exception"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - start
        save()
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
    print(f"TERMINAL_COMPLETE passed={report['passed']} mode={report['mode']}", flush=True)


if __name__ == "__main__":
    main()
