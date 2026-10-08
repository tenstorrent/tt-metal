# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Loaded prefill comparisons with four populated 256K KV slots.

GEMMA4_LOAD_MODE selects ragged, regular2048, regular4096, regular8192, or
stable8192 (the independent FP32 reduction control). Each scenario warms once,
then measures five replays of the same boundary. This is a service-time study,
not a request arrival simulator or a numerical correctness test.
"""

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import ttnn
from models.common.weight_cache import build_cached_state_dict
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model, weight_cache_identity
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs, resolve_cache_dir_from_tt_cache_path
from models.demos.gemma4_d_p.tt.precision import Gemma4Precision
from models.demos.gemma4_d_p.tt.ragged_prefill import PrefillRequest, RaggedPrefillPlan
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


@pytest.fixture
def run_batch(mesh_device, tmp_path, request):
    mode = os.getenv("GEMMA4_LOAD_MODE", "ragged")
    chunk_sizes = {"ragged": 8192, "regular2048": 2048, "regular4096": 4096, "regular8192": 8192, "stable8192": 8192}
    if mode not in chunk_sizes:
        raise ValueError(f"GEMMA4_LOAD_MODE must be one of {tuple(chunk_sizes)}")
    repeats = int(os.getenv("GEMMA4_LOAD_REPEATS", "5"))
    if repeats < 1:
        raise ValueError("GEMMA4_LOAD_REPEATS must be positive")
    model_id = os.getenv("HF_MODEL", "google/gemma-4-31B-it")
    root = os.environ["TT_CACHE_PATH"]
    mesh = MeshConfig(mesh_device)
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(model_id))
    num_layers = int(os.getenv("GEMMA4_RAGGED_TEST_LAYERS", "60"))
    if not 1 <= num_layers <= args.num_hidden_layers:
        raise ValueError("GEMMA4_RAGGED_TEST_LAYERS is outside the model's layer count")
    selected = set(filter(None, os.getenv("GEMMA4_LOAD_CASES", "").split(",")))
    config = SimpleNamespace(
        num_users=4, chunk_size=chunk_sizes[mode], max_seq_len=262144, mesh_shape=(8, 4), num_layers=num_layers
    )
    cache_dir = resolve_cache_dir_from_tt_cache_path(root, dtype=ttnn.bfloat16, mesh_shape=config.mesh_shape)
    identity = weight_cache_identity(
        model_id, args.num_hidden_layers, config.mesh_shape, Gemma4Precision.load(model_id)
    )
    state = build_cached_state_dict(cache_dir, args=args, build_variant=identity["build_variant"])
    caches = allocate_ring_kv_caches(
        mesh,
        args,
        num_users=config.num_users,
        max_seq_len=config.max_seq_len,
        prefill_chunk_size=config.chunk_size,
        num_layers=num_layers,
    )
    _, model, _, _ = create_tt_model(
        mesh,
        config.chunk_size,
        max_batch_size=config.num_users,
        max_seq_len=config.max_seq_len,
        num_layers=num_layers,
        hf_model_id=model_id,
        state_dict=state,
        ring_kv_caches=caches,
        tt_cache_path=root,
    )
    model.stable_prefill_reductions = mode == "stable8192"
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id=model_id, tt_cache_path=root, config=config)
    runtime.model = model
    tokens = tuple(_get_prefill_tokens(model_id, config.max_seq_len, args.vocab_size, "text")[0].tolist())
    runtime.input_tokens = runtime.make_chunk_input(tokens[: config.chunk_size])
    runtime.positions = runtime.make_chunk_input(range(config.chunk_size))
    runtime.metadata = None
    model.set_prefill_rope_positions(runtime.positions)
    model._prefill_metadata_external = True
    runtime._stage_positions(0, 0)
    warmup = runtime._forward()
    ttnn.synchronize_device(mesh_device)
    warmup.deallocate(True)
    runtime.capture_trace(caches)

    measurements = []
    full_stream = None
    populate_prefixes = getattr(request, "param", True)
    output = Path(os.getenv("GEMMA4_LOAD_OUTPUT", str(tmp_path / f"{mode}.json")))
    output.parent.mkdir(parents=True, exist_ok=True)

    def write_measurements():
        output.write_text(
            json.dumps(
                dict(
                    mode=mode,
                    model=model_id,
                    layers=num_layers,
                    mesh=[8, 4],
                    chunk_size=config.chunk_size,
                    cache_capacity=config.max_seq_len,
                    allocated_slots=config.num_users,
                    repeats=repeats,
                    prefix_tokens=253952 if populate_prefixes else 0,
                    scheduler="round-robin chunks" if mode != "ragged" else "packed batch",
                    activations_dram_only=os.getenv("GEMMA4_ACTIVATIONS_DRAM_ONLY", "0"),
                    trace_allocation_tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
                    tp_reduction="fixed_fp32" if model.stable_prefill_reductions else "reduce_scatter",
                    full_batch_dram_override=os.getenv("GEMMA4_LOAD_FULL_DRAM", "0"),
                    measurements=measurements,
                    full_stream=full_stream,
                ),
                indent=2,
            )
            + "\n"
        )

    def regular_chunk(slot, start, end):
        # Real continuation tokens fill the unused suffix of a partial chunk.
        # Valid outputs are causal; computing this suffix preserves the seeded
        # prefix for subsequent fixed-boundary performance samples.
        inp = runtime.make_chunk_input(tokens[start : start + config.chunk_size])
        runtime.prefill_chunk(inp, caches, slot_id=slot, actual_start=start, actual_end=end)

    try:
        # Populate actual KV, not just slot counters or zero-filled history.
        # All four slots use the same text, but have separate physical KV and
        # execute independently: there is no prefix sharing/cache reuse.
        if populate_prefixes:
            print(f"\nLOAD STUDY: {mode}, {num_layers} layers; populating four 248K prefixes", flush=True)
            for slot in range(config.num_users):
                for start in range(0, 253952, config.chunk_size):
                    regular_chunk(slot, start, start + config.chunk_size)
                print(f"  prefix slot={slot} ready", flush=True)

        def run_full_context():
            """Four simultaneous 256K prompts, progressing through real contiguous chunks."""
            nonlocal full_stream

            def step(start):
                t0 = time.perf_counter()
                completion = []
                if mode == "ragged":
                    requests = tuple(
                        PrefillRequest(100 + slot, slot, start, tokens[start : start + config.chunk_size])
                        for slot in range(config.num_users)
                    )
                    result = runtime.prefill_batch(requests, caches)
                    elapsed = time.perf_counter() - t0
                    result.deallocate()
                    completion = [elapsed] * config.num_users
                else:
                    for slot in range(config.num_users):
                        regular_chunk(slot, start, start + config.chunk_size)
                        completion.append(time.perf_counter() - t0)
                    elapsed = time.perf_counter() - t0
                return elapsed, completion

            step(0)  # Warm/capture the four-request shape outside the measurement.
            rows = []
            t0 = time.perf_counter()
            for start in range(0, config.max_seq_len, config.chunk_size):
                round_start = time.perf_counter() - t0
                elapsed, completion = step(start)
                rows.append(
                    dict(
                        start=start,
                        end=start + config.chunk_size,
                        seconds=elapsed,
                        cumulative_wall_seconds=time.perf_counter() - t0,
                        request_completion_seconds=[round_start + value for value in completion],
                    )
                )
            wall = time.perf_counter() - t0
            full_stream = dict(
                requests=config.num_users,
                tokens_per_request=config.max_seq_len,
                useful_tokens=config.num_users * config.max_seq_len,
                wall_seconds=wall,
                useful_tokens_per_second=config.num_users * config.max_seq_len / wall,
                rounds=rows,
            )
            write_measurements()
            print(
                f"STREAM {mode}: {full_stream['useful_tokens']} tokens in {wall:.3f}s, {full_stream['useful_tokens_per_second']:.0f} tok/s",
                flush=True,
            )

        def run_batch(lengths, *, starts, label):
            """Warm once, then replay this fixed boundary; report complete-batch service time.

            All measured lengths are tile-aligned, so packed cache writes preserve
            every seeded row beyond the valid suffix. Host request bookkeeping is
            restored before each repetition; KV storage and computed prefixes remain.
            """
            if selected and label not in selected:
                return
            if len(lengths) != len(starts) or not 1 <= len(lengths) <= config.num_users:
                raise ValueError("Provide matching lengths and starts for 1 to 4 slots")
            if any(n <= 0 or n > 8192 or n % 32 for n in lengths):
                raise ValueError("Study lengths must be tile-aligned and at most 8192")
            requests = tuple(
                PrefillRequest(100 + slot, slot, start, tokens[start : start + length])
                for slot, (length, start) in enumerate(zip(lengths, starts))
            )
            if any(r.actual_end > config.max_seq_len or r.actual_start % 8192 for r in requests):
                raise ValueError("Study starts must be 8K aligned and ranges must fit in 256K")
            capture_needed = (
                mode == "ragged" and RaggedPrefillPlan.for_requests(requests) not in runtime.ragged_variants
            )

            def execute():
                for request in requests:
                    runtime.slot_ends[request.slot_id] = request.actual_start
                    runtime.slot_requests[request.slot_id] = request.request_id
                t0 = time.perf_counter()
                if mode == "ragged":
                    result = runtime.prefill_batch(requests, caches)
                    elapsed = time.perf_counter() - t0
                    result.deallocate()
                    completion = [elapsed] * len(requests)
                else:
                    completion = [None] * len(requests)
                    # Under load, native prefill visits each active slot once
                    # per chunk round instead of completing one prompt first.
                    for offset in range(0, max(lengths), config.chunk_size):
                        for i, request in enumerate(requests):
                            start = request.actual_start + offset
                            if start >= request.actual_end:
                                continue
                            end = min(start + config.chunk_size, request.actual_end)
                            regular_chunk(request.slot_id, start, end)
                            if end == request.actual_end:
                                completion[i] = time.perf_counter() - t0
                    elapsed = time.perf_counter() - t0
                return dict(seconds=elapsed, request_completion_seconds=completion)

            # The default L1 path cannot hold the four-full-chunk MLP activations.
            # Opt in explicitly to the existing DRAM option for that scenario;
            # record the configuration on every row rather than hiding the change.
            storage = os.getenv("GEMMA4_ACTIVATIONS_DRAM_ONLY", "0")
            if mode == "ragged" and lengths == [8192] * 4 and os.getenv("GEMMA4_LOAD_FULL_DRAM", "0") == "1":
                storage = "1"
            with mock.patch.dict(os.environ, {"GEMMA4_ACTIVATIONS_DRAM_ONLY": storage}):
                first = execute()
                samples = [execute() for _ in range(repeats)]
            seconds = statistics.median(sample["seconds"] for sample in samples)
            row = dict(
                label=label,
                lengths=lengths,
                starts=starts,
                ends=[r.actual_end for r in requests],
                useful_tokens=sum(lengths),
                activation_storage="DRAM" if storage == "1" else "default L1",
                capture_needed=capture_needed,
                first_call_seconds=first["seconds"],
                median_seconds=seconds,
                useful_tokens_per_second=sum(lengths) / seconds,
                median_request_completion_seconds=[
                    statistics.median(s["request_completion_seconds"][i] for s in samples) for i in range(len(requests))
                ],
                samples=samples,
            )
            measurements.append(row)
            write_measurements()
            print(
                f"LOAD {mode:12s} {label:24s} | useful={sum(lengths):5d} | {seconds * 1000:8.2f} ms"
                f" | {sum(lengths) / seconds:8.0f} tok/s | first={first['seconds']:.2f}s"
                f" capture={capture_needed} storage={row['activation_storage']}",
                flush=True,
            )
            for i, request in enumerate(requests):
                print(
                    f"  slot={i} [{request.actual_start:6d}, {request.actual_end:6d})"
                    f" completion={row['median_request_completion_seconds'][i] * 1000:.2f} ms",
                    flush=True,
                )

        run_batch.full_context = run_full_context
        yield run_batch
    finally:
        runtime.release_trace()
        write_measurements()


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_ragged_prefill_under_load(run_batch):
    # Same useful work and four allocated 256K slots in every execution mode.
    # Short chunks represent final tails; ongoing long requests use full 8K chunks.
    run_batch([2048, 2048, 2048, 2048], starts=[0, 0, 0, 0], label="tails4_early")
    run_batch([2048, 2048, 2048, 2048], starts=[253952] * 4, label="tails4_late")
    run_batch([2048, 2048, 2048, 2048], starts=[0, 8192, 131072, 253952], label="tails4_mixed")
    run_batch([2048, 2048, 2048, 2048], starts=[0, 0, 0, 253952], label="tails4_one_late")
    run_batch([2048, 2048, 2048, 2048], starts=[0, 0, 253952, 253952], label="tails4_two_late")
    run_batch([2048, 2048, 2048, 2048], starts=[0, 253952, 253952, 253952], label="tails4_three_late")
    run_batch([4096, 4096], starts=[0, 0], label="tails2_early")
    run_batch([4096, 4096], starts=[253952] * 2, label="tails2_late")
    run_batch([4096, 4096], starts=[0, 253952], label="tails2_mixed")
    run_batch([8192, 8192, 8192, 8192], starts=[0, 0, 0, 0], label="full4_early")
    run_batch([8192, 8192, 8192, 8192], starts=[253952] * 4, label="full4_late")
    run_batch([8192, 8192, 8192, 8192], starts=[0, 8192, 131072, 253952], label="full4_mixed")
    run_batch([1024, 32], starts=[0, 0], label="small2_early")
    run_batch([1024, 32], starts=[253952] * 2, label="small2_late")
    run_batch([1024, 32], starts=[0, 253952], label="small2_mixed")
    # Single-request controls isolate reduction cost from split/concat overhead.
    run_batch([8192], starts=[0], label="single_early")
    run_batch([8192], starts=[253952], label="single_late")
    # Revisit compiled shapes to quantify the one-resident-trace policy under churn.
    run_batch([2048, 2048, 2048, 2048], starts=[0, 0, 0, 0], label="churn_tails4")
    run_batch([4096, 4096], starts=[0, 0], label="churn_tails2")


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
@pytest.mark.parametrize("run_batch", [False], indirect=True, ids=["four_256k_prompts"])
@pytest.mark.skipif(
    os.getenv("GEMMA4_LOAD_STREAM", "0") != "1",
    reason="Set GEMMA4_LOAD_STREAM=1 for the full-context stream experiment",
)
def test_full_context_loaded_stream(run_batch):
    run_batch.full_context()
