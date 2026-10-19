# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model traced prefill: striped batches versus sequential legacy requests."""

import gc
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
from models.demos.gemma4_d_p.demo.text_demo_prefill import (
    TRACE_REGION_SIZE,
    _get_prefill_tokens,
    _shared_device_weights,
)
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


def _device_samples(mesh, trace_id, *, repeats, num_slots=1, stage=None, profile=False):
    records, dropped, sources = {}, [0], {}
    last_record = [time.monotonic()]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            source_id = None
            if profile:
                signature = tuple(str(source) for source in record.kernel_sources)
                source_id = sources.setdefault(signature, len(sources))
            records.setdefault(int(record.chip_id), []).append(
                (
                    int(record.runtime_id),
                    int(record.start_timestamp),
                    int(record.end_timestamp),
                    float(record.frequency),
                    source_id,
                )
            )
        last_record[0] = time.monotonic()

    ttnn.synchronize_device(mesh)
    time.sleep(0.2)
    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    host_us = []
    try:
        for slot in range(num_slots):
            if stage is not None:
                stage(slot)
            for _ in range(repeats + 1):
                begin = time.perf_counter()
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                host_us.append((time.perf_counter() - begin) * 1e6)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if len(records) == mesh.get_num_devices() and time.monotonic() - last_record[0] >= 0.1:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    assert not dropped[0], f"Incomplete profiler stream: dropped={dropped[0]}"
    return _replay_spans(records, host_us, repeats, num_slots, mesh.get_num_devices(), sources if profile else None)


def _replay_spans(records, host_us, repeats, num_slots, num_devices, sources=None):
    assert len(records) == num_devices, f"Profiler returned {len(records)}/{num_devices} chips"
    num_replays = num_slots * (repeats + 1)
    counts = {len(rows) for rows in records.values()}
    assert len(counts) == 1, f"Incomplete per-chip program counts: {counts}"
    count = counts.pop()
    assert count > 0 and count % num_replays == 0, f"Cannot split {count} programs into {num_replays} replays"
    programs_per_replay = count // num_replays
    captures = [dict(chips={}) for _ in range(num_replays)]
    for chip, rows in records.items():
        rows.sort(key=lambda row: row[1])
        signature = [row[0] for row in rows[:programs_per_replay]]
        for replay, capture in enumerate(captures):
            programs = rows[replay * programs_per_replay : (replay + 1) * programs_per_replay]
            assert [
                row[0] for row in programs
            ] == signature, f"Incomplete program sequence on chip {chip}, replay {replay}"
            assert all(end > start and frequency > 0 for _, start, end, frequency, _ in programs)
            frequencies = {row[3] for row in programs}
            assert len(frequencies) == 1, f"Inconsistent timestamp frequency on chip {chip}"
            frequency = frequencies.pop()
            start, end = min(row[1] for row in programs), max(row[2] for row in programs)
            capture["chips"][chip] = dict(
                start=start,
                end=end,
                frequency_ghz=frequency,
                programs=programs_per_replay,
                span_us=(end - start) / frequency / 1000,
            )
    result = []
    for replay, capture in enumerate(captures):
        slot, sample = divmod(replay, repeats + 1)
        if sample == 0:
            continue
        chip = max(capture["chips"], key=lambda chip: capture["chips"][chip]["span_us"])
        capture.update(
            slot=slot,
            sample=sample - 1,
            device_us=capture["chips"][chip]["span_us"],
            profiled_replay_host_us=host_us[replay],
        )
        if sources is not None:
            programs = records[chip][replay * programs_per_replay : (replay + 1) * programs_per_replay]
            capture["profile"] = dict(
                chip_id=chip,
                kernel_sources={source_id: signature for signature, source_id in sources.items()},
                programs=[(start, end, source_id) for _, start, end, _, source_id in programs],
            )
        result.append(capture)
    return result


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_packed_perf(mesh_device, reset_seeds, monkeypatch):
    _run_comparison(mesh_device, monkeypatch, cases=CASES, chunk_size=8192, output_name="gemma4_varlen_perf")


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_single_request_comparison(mesh_device, reset_seeds, monkeypatch):
    chunk_size = int(os.getenv("GEMMA4_PERF_CHUNK_SIZE", "8192"))
    assert 0 < chunk_size <= 262144 and chunk_size % 1024 == 0, "Chunk size must be a multiple of 1024 up to 262144"
    _run_comparison(
        mesh_device,
        monkeypatch,
        cases={f"single_{chunk_size}": (chunk_size,)},
        chunk_size=chunk_size,
        output_name=f"gemma4_single_prefill_{chunk_size}",
    )


@torch.no_grad()
@pytest.mark.timeout(14400)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_comprehensive_comparison(mesh_device, reset_seeds, monkeypatch):
    mixed_requests = {
        4096: (256, 512, 768, 1024, 1536),
        6144: (256, 512, 768, 1024, 1536, 2048),
        8192: (512, 512, 1024, 1536, 2048, 2560),
    }
    root = Path(os.getenv("GEMMA4_PERF_OUTPUT", "investigations/gemma4_comprehensive_prefill"))
    root.mkdir(parents=True, exist_ok=True)
    summary = dict(complete=False, chunk_sizes=list(mixed_requests), runs=[])

    def save():
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(root / "results.json")

    try:
        save()
        with _shared_device_weights():
            for chunk_size, lengths in mixed_requests.items():
                assert sum(lengths) == chunk_size
                logger.info("[prefill_perf] comparing chunk={} single vs mixed={}", chunk_size, lengths)
                report = _run_comparison(
                    mesh_device,
                    monkeypatch,
                    cases={f"single_{chunk_size}": (chunk_size,), "mixed": lengths},
                    chunk_size=chunk_size,
                    output_name="gemma4_comprehensive_prefill",
                    output_dir=root / f"chunk_{chunk_size}",
                )
                summary["runs"].append(
                    {
                        **report,
                        "results": [
                            {key: value for key, value in row.items() if key not in ("captures", "program_profile")}
                            for row in report["results"]
                        ],
                    }
                )
                save()
                gc.collect()
                mesh_device.clear_program_cache()
        summary["complete"] = True
    finally:
        save()


def _run_comparison(mesh_device, monkeypatch, *, cases, chunk_size, output_name, output_dir=None):
    monkeypatch.setenv("GEMMA4_PREFILL_MAX_TRACES", str(len(cases)))
    num_users = max(len(lengths) for lengths in cases.values())
    max_seq_len = -(-262144 // chunk_size) * chunk_size
    last_prefix = (262144 - chunk_size) // chunk_size * chunk_size
    root = Path(
        output_dir if output_dir is not None else os.getenv("GEMMA4_PERF_OUTPUT", f"investigations/{output_name}")
    )
    root.mkdir(parents=True, exist_ok=True)
    device_repeats = int(os.getenv("GEMMA4_PERF_DEVICE_REPEATS", "3"))
    prefixes = tuple(
        int(value) for value in os.getenv("GEMMA4_PERF_PREFIXES", f"0,{last_prefix}" if last_prefix else "0").split(",")
    )
    assert device_repeats > 0
    assert prefixes == tuple(sorted(set(prefixes))) and all(
        0 <= p <= last_prefix and p % chunk_size == 0 for p in prefixes
    )
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "This benchmark requires the real-time device profiler"
    mesh = MeshConfig(mesh_device)
    hf_model = os.environ["HF_MODEL"]
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(hf_model))
    tokens = _get_prefill_tokens(hf_model, max_seq_len + (num_users - 1) * chunk_size, args.vocab_size)[0]
    config = SimpleNamespace(
        mesh_shape=(8, 4),
        sp_axis=0,
        chunk_size=chunk_size,
        max_seq_len=max_seq_len,
        num_users=num_users,
        num_layers=60,
        use_trace=True,
    )
    caches = allocate_ring_kv_caches(
        mesh, args, num_users=num_users, max_seq_len=max_seq_len, prefill_chunk_size=chunk_size
    )
    runtime = Gemma4PrefillRuntime(mesh_device=mesh_device, hf_model_id=hf_model, tt_cache_path=None, config=config)
    report = dict(
        complete=False,
        started_at=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        model=hf_model,
        layers=60,
        mesh=[8, 4],
        chunk_size=chunk_size,
        cache_capacity=caches.max_seq_len,
        num_users=num_users,
        prefixes=prefixes,
        device_metric="max per-chip first-program-start to last-program-end span; includes device op gaps",
        cache_history="zero-initialized synthetic KV; measured forwards overwrite their current chunks",
        warmup_replays=1,
        baseline=f"one legacy {chunk_size}-padded forward per request; packed uses one shared {chunk_size}-row forward",
        device_repeats=device_repeats,
        results=[],
    )

    profile_programs = os.getenv("GEMMA4_PERF_PROFILE_PROGRAMS", "0") == "1"

    def save():
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "results.json")

    def legacy_stage(slot, prefix, length):
        values = tokens[slot * chunk_size + prefix : slot * chunk_size + prefix + length].tolist() + [0] * (
            chunk_size - length
        )
        ttnn.copy_host_to_device_tensor(runtime._host_rows(values), runtime.input_tokens)
        ttnn.copy_host_to_device_tensor(runtime._host_rows(list(range(prefix, prefix + chunk_size))), runtime.positions)
        runtime.model.prefill_metadata.update(slot_idx=slot, kv_actual_global=prefix)

    def legacy_forward():
        embeddings = runtime.model.transform_and_embed_prefill_inputs_device(runtime.input_tokens)
        output = runtime.model(hidden_states=embeddings)
        ttnn.copy(output, runtime.output)
        output.deallocate(True)

    def packed_run(lengths, prefix):
        segments = tuple(PrefillSegment(slot, prefix, prefix + length) for slot, length in enumerate(lengths))
        values = [
            tokens[s.slot_id * chunk_size + s.actual_start : s.slot_id * chunk_size + s.actual_end].tolist()
            for s in segments
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
        legacy_stage(0, 0, chunk_size)
        legacy_forward()
        ttnn.synchronize_device(mesh_device)
        legacy_trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            legacy_forward()
        finally:
            ttnn.end_trace_capture(mesh_device, legacy_trace, cq_id=0)

        for path in ("legacy", "packed"):
            for prefix in prefixes:
                for name, lengths in cases.items():
                    logger.info("[prefill_perf] measuring {} {} prefix={}", path, name, prefix)
                    count = 1 if path == "packed" else len(lengths)
                    if path == "packed":
                        packed_run(lengths, prefix)
                    trace_id = runtime.trace_id if path == "packed" else legacy_trace
                    captures = _device_samples(
                        mesh_device,
                        trace_id,
                        repeats=device_repeats,
                        num_slots=count,
                        stage=(lambda slot: legacy_stage(slot, prefix, lengths[slot])) if path == "legacy" else None,
                        profile=profile_programs and len(lengths) == 1,
                    )
                    device_us = [
                        sum(capture["device_us"] for capture in captures if capture["sample"] == sample)
                        for sample in range(device_repeats)
                    ]
                    real_tokens = sum(lengths)
                    row = dict(
                        path=path,
                        case=name,
                        request_lengths=lengths,
                        prefix=prefix,
                        real_tokens=real_tokens,
                        projection_rows=chunk_size * count,
                        device_us=_distribution(device_us),
                        device_us_per_token=statistics.median(device_us) / real_tokens,
                        captures=captures,
                    )
                    if path == "packed":
                        legacy = next(
                            result
                            for result in report["results"]
                            if result["path"] == "legacy" and result["case"] == name and result["prefix"] == prefix
                        )
                        row["device_speedup_vs_legacy"] = legacy["device_us"]["median"] / row["device_us"]["median"]
                        logger.info(
                            "[prefill_perf] {} prefix={}: legacy={:.3f}ms packed={:.3f}ms device speedup={:.3f}x",
                            name,
                            prefix,
                            legacy["device_us"]["median"] / 1000,
                            row["device_us"]["median"] / 1000,
                            row["device_speedup_vs_legacy"],
                        )
                    if profile_programs and len(lengths) == 1:
                        row["program_profile"] = captures[0]
                    report["results"].append(row)
                    save()
                    logger.info(
                        "[prefill_perf] {} {} prefix={} device={:.3f}ms device/token={:.3f}us",
                        path,
                        name,
                        prefix,
                        row["device_us"]["median"] / 1000,
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
    return report
