# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model traced prefill: striped batches versus sequential legacy requests."""

import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.packed_prefill import _output
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE, _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime

CASES = {
    "single_8192": (8192,),
    "single_1600": (1600,),
    "four_1600": (1600,) * 4,
    "mixed_six": (256, 512, 768, 1024, 1536, 2048),
}


def _distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "samples": values}


def _device_span(mesh, trace_id):
    chips, dropped, invalid = {}, [0], [0]
    last_record = [time.monotonic()]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            start, end, frequency = int(record.start_timestamp), int(record.end_timestamp), float(record.frequency)
            if end <= start or frequency <= 0:
                invalid[0] += 1
                continue
            chip = int(record.chip_id)
            if chip not in chips:
                chips[chip] = dict(start=start, end=end, frequency_ghz=frequency, programs=0, kernel_ns=0)
            row = chips[chip]
            row["start"] = min(row["start"], start)
            row["end"] = max(row["end"], end)
            row["programs"] += 1
            row["kernel_ns"] += (end - start) / frequency
        last_record[0] = time.monotonic()

    ttnn.synchronize_device(mesh)
    time.sleep(0.2)
    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        begin = time.perf_counter()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        host_us = (time.perf_counter() - begin) * 1e6
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(chips) == 32 and time.monotonic() - last_record[0] >= 0.3:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    assert not dropped[0] and not invalid[0], f"Incomplete profiler stream: dropped={dropped}, invalid={invalid}"
    assert len(chips) == 32, f"Profiler returned {len(chips)}/32 chips"
    counts = {row["programs"] for row in chips.values()}
    assert len(counts) == 1, f"Incomplete per-chip program counts: {counts}"
    for row in chips.values():
        row["span_us"] = (row["end"] - row["start"]) / row["frequency_ghz"] / 1000
    return dict(device_us=max(row["span_us"] for row in chips.values()), profiled_replay_host_us=host_us, chips=chips)


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_packed_perf(mesh_device, reset_seeds, monkeypatch):
    monkeypatch.setenv("GEMMA4_PREFILL_MAX_TRACES", str(len(CASES)))
    root = Path(os.getenv("GEMMA4_PERF_OUTPUT", "investigations/gemma4_varlen_perf"))
    root.mkdir(parents=True, exist_ok=True)
    repeats = int(os.getenv("GEMMA4_PERF_REPEATS", "7"))
    device_repeats = int(os.getenv("GEMMA4_PERF_DEVICE_REPEATS", "3"))
    prefixes = tuple(int(value) for value in os.getenv("GEMMA4_PERF_PREFIXES", "0,253952").split(","))
    assert repeats > 0 and device_repeats > 0
    assert prefixes == tuple(sorted(set(prefixes))) and all(0 <= p <= 253952 and p % 8192 == 0 for p in prefixes)
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "This benchmark requires the real-time device profiler"
    mesh = MeshConfig(mesh_device)
    hf_model = os.environ["HF_MODEL"]
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(hf_model))
    tokens = _get_prefill_tokens(hf_model, 262144 + 5 * 8192, args.vocab_size)[0]
    config = SimpleNamespace(
        mesh_shape=(8, 4), sp_axis=0, chunk_size=8192, max_seq_len=262144, num_users=6, num_layers=60, use_trace=True
    )
    caches = allocate_ring_kv_caches(mesh, args, num_users=6, max_seq_len=262144)
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id=hf_model, tt_cache_path=None, config=config)
    report = dict(
        complete=False,
        started_at=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        model=hf_model,
        layers=60,
        mesh=[8, 4],
        chunk_size=8192,
        prefixes=prefixes,
        device_metric="max per-chip first-program-start to last-program-end span; includes device op gaps",
        wall_metric="host token packing/upload, metadata/RoPE staging, trace replay and synchronize; excludes sockets",
        baseline="one legacy 8192-padded forward per request; packed uses one shared 8192-row forward",
        repeats=repeats,
        device_repeats=device_repeats,
        results=[],
    )

    def save():
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "results.json")

    def legacy_stage(slot, prefix, length):
        values = tokens[slot * 8192 + prefix : slot * 8192 + prefix + length].tolist() + [0] * (8192 - length)
        ttnn.copy_host_to_device_tensor(runtime._host_rows(values), runtime.input_tokens)
        ttnn.copy_host_to_device_tensor(runtime._host_rows(list(range(prefix, prefix + 8192))), runtime.positions)
        runtime.model.prefill_metadata.update(slot_idx=slot, kv_actual_global=prefix)

    def legacy_forward():
        embeddings = runtime.model.transform_and_embed_prefill_inputs_device(runtime.input_tokens)
        output = runtime.model(hidden_states=embeddings)
        ttnn.copy(output, runtime.output)
        output.deallocate(True)

    def packed_run(lengths, prefix, slots=None):
        slots = range(len(lengths)) if slots is None else slots
        segments = tuple(PrefillSegment(slot, prefix, prefix + length) for slot, length in zip(slots, lengths))
        values = [
            tokens[s.slot_id * 8192 + s.actual_start : s.slot_id * 8192 + s.actual_end].tolist() for s in segments
        ]
        for segment in segments:
            runtime.slot_ends[segment.slot_id] = prefix
        inputs = runtime.make_batch_input(values, segments)
        runtime.prefill_batch(inputs, caches, segments=segments)

    legacy_trace = None
    try:
        save()
        runtime.compile(caches)
        runtime.capture_trace(caches)
        legacy_stage(0, 0, 8192)
        legacy_forward()
        ttnn.synchronize_device(mesh_device)
        legacy_trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            legacy_forward()
        finally:
            ttnn.end_trace_capture(mesh_device, legacy_trace, cq_id=0)

        for path in ("legacy", "packed"):
            populated = 0
            for prefix in prefixes:
                if prefix:
                    logger.info("[prefill_perf] populating {} histories to {}", path, prefix)
                    for slot in range(6):
                        for position in range(populated, prefix, 8192):
                            if path == "legacy":
                                legacy_stage(slot, position, 8192)
                                ttnn.execute_trace(mesh_device, legacy_trace, cq_id=0, blocking=True)
                            else:
                                packed_run((8192,), position, (slot,))
                        logger.info("[prefill_perf] {} slot {} prefix ready", path, slot)
                    populated = prefix
                for name, lengths in CASES.items():
                    logger.info("[prefill_perf] measuring {} {} prefix={}", path, name, prefix)

                    def run():
                        if path == "packed":
                            packed_run(lengths, prefix)
                        else:
                            for slot, length in enumerate(lengths):
                                legacy_stage(slot, prefix, length)
                                ttnn.execute_trace(mesh_device, legacy_trace, cq_id=0, blocking=True)

                    for _ in range(2):
                        run()
                    assert torch.isfinite(_output(runtime, striped=path == "packed")).all()
                    wall_us = []
                    for _ in range(repeats):
                        begin = time.perf_counter()
                        run()
                        wall_us.append((time.perf_counter() - begin) * 1e6)
                    replay_us, captures = [], []
                    count = 1 if path == "packed" else len(lengths)
                    for slot in range(count):
                        if path == "legacy":
                            legacy_stage(slot, prefix, lengths[slot])
                        trace_id = runtime.trace_id if path == "packed" else legacy_trace
                        for sample in range(device_repeats):
                            capture = _device_span(mesh_device, trace_id)
                            captures.append(dict(slot=slot if path == "legacy" else None, sample=sample, **capture))
                        samples = []
                        for _ in range(repeats):
                            begin = time.perf_counter()
                            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
                            samples.append((time.perf_counter() - begin) * 1e6)
                        replay_us.append(samples)
                    device_us = [
                        sum(capture["device_us"] for capture in captures if capture["sample"] == sample)
                        for sample in range(device_repeats)
                    ]
                    replay_sum = [sum(samples[i] for samples in replay_us) for i in range(repeats)]
                    real_tokens = sum(lengths)
                    row = dict(
                        path=path,
                        case=name,
                        request_lengths=lengths,
                        prefix=prefix,
                        real_tokens=real_tokens,
                        projection_rows=8192 * count,
                        device_us=_distribution(device_us),
                        replay_host_us=_distribution(replay_sum),
                        wall_us=_distribution(wall_us),
                        device_us_per_token=statistics.median(device_us) / real_tokens,
                        wall_us_per_token=statistics.median(wall_us) / real_tokens,
                        captures=captures,
                    )
                    report["results"].append(row)
                    save()
                    logger.info(
                        "[prefill_perf] {} {} prefix={} device={:.3f}ms wall={:.3f}ms device/token={:.3f}us",
                        path,
                        name,
                        prefix,
                        row["device_us"]["median"] / 1000,
                        row["wall_us"]["median"] / 1000,
                        row["device_us_per_token"],
                    )
            if path == "legacy":
                ttnn.release_trace(mesh_device, legacy_trace)
                legacy_trace = None
        report["complete"] = True
    finally:
        save()
        if legacy_trace is not None:
            ttnn.release_trace(mesh_device, legacy_trace)
        runtime.release_trace()
