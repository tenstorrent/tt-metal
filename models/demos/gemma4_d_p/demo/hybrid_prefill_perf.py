# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warmed full-model legacy, fully traced packed, and hybrid prefill comparison."""

import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from loguru import logger
from tabulate import tabulate

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.prefill_perf import CASES, _device_samples, _distribution
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE, _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.hybrid_prefill import Gemma4HybridPrefill
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment, pack_positions, pack_tokens


def _host_samples(device, run, repeats, count, stage):
    totals = [0.0] * repeats
    for slot in range(count):
        if stage is not None:
            stage(slot)
        run()
        ttnn.synchronize_device(device)
        for sample in range(repeats):
            start = time.perf_counter()
            run()
            ttnn.synchronize_device(device)
            totals[sample] += (time.perf_counter() - start) * 1e6
    return totals


def _summary(report, metric):
    rows = []
    for prefix in report["prefixes"]:
        for name, lengths in report["cases"].items():
            measurements = {
                row["path"]: row[metric]["median"] / 1000
                for row in report["results"]
                if row["case"] == name and row["prefix"] == prefix
            }
            if len(measurements) != 3:
                continue
            requests = f"{len(lengths)} x {lengths[0]}" if len(set(lengths)) == 1 else " + ".join(map(str, lengths))
            rows.append(
                [
                    requests,
                    f"{prefix:,}",
                    *(f"{measurements[path]:.2f}" for path in ("legacy", "packed", "hybrid")),
                    f"{measurements['hybrid'] / measurements['packed']:.2f}x",
                ]
            )
    return tabulate(
        rows,
        headers=["Requests", "Prefix", "Legacy ms", "Packed ms", "Hybrid ms", "Hybrid / packed"],
        tablefmt="github",
        disable_numparse=True,
        colalign=("left", "right", "right", "right", "right", "right"),
    )


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_hybrid_comparison(mesh_device, reset_seeds):
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "Requires the real-time device profiler"
    chunk_size = int(os.getenv("GEMMA4_PERF_CHUNK_SIZE", "8192"))
    repeats = int(os.getenv("GEMMA4_PERF_DEVICE_REPEATS", "5"))
    assert 0 < chunk_size <= 262144 and chunk_size % 1024 == 0 and repeats > 0
    cases = {
        "single_full": (chunk_size,),
        **{name: lengths for name, lengths in CASES.items() if name != "single_8192"},
    }
    cases = {name: lengths for name, lengths in cases.items() if sum(-(-n // 256) * 256 for n in lengths) <= chunk_size}
    last_prefix = (262144 - chunk_size) // chunk_size * chunk_size
    prefixes = tuple(
        int(p) for p in os.getenv("GEMMA4_PERF_PREFIXES", f"0,{last_prefix}" if last_prefix else "0").split(",")
    )
    assert prefixes == tuple(sorted(set(prefixes))) and all(
        0 <= p <= last_prefix and p % chunk_size == 0 for p in prefixes
    )
    num_users = max(map(len, cases.values()))
    mesh = MeshConfig(mesh_device)
    hf_model = os.environ["HF_MODEL"]
    args = Gemma4ModelArgs.from_hf_config(Gemma4ModelArgs.load_hf_config(hf_model))
    capacity = -(-262144 // chunk_size) * chunk_size
    caches = allocate_ring_kv_caches(
        mesh, args, num_users=num_users, max_seq_len=capacity, prefill_chunk_size=chunk_size
    )
    _, model, _, _ = create_tt_model(
        mesh_config=mesh,
        hf_model_id=hf_model,
        prefill_chunk_size=chunk_size,
        max_seq_len=capacity,
        ring_kv_caches=caches,
        stripe_size=32,
        max_prefill_requests=num_users,
    )
    model._prefill_metadata_external = True
    text = _get_prefill_tokens(hf_model, chunk_size * num_users, args.vocab_size)[0].tolist()

    def host(rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int64).reshape(1, chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (8, 4), dims=(1, None)),
        )

    tokens = ttnn.to_device(host([0] * chunk_size), mesh_device)
    positions = ttnn.to_device(host([0] * chunk_size), mesh_device)
    model.set_prefill_rope_positions(positions)
    hybrid = Gemma4HybridPrefill(model, tokens)

    def packed_stage(lengths, prefix):
        segments = tuple(PrefillSegment(slot, prefix, prefix + length) for slot, length in enumerate(lengths))
        model.packed_prefill_metadata.update(
            (s.slot_id, s.actual_start, s.actual_end - s.actual_start, s.padded_length()) for s in segments
        )
        values = [text[slot * chunk_size : slot * chunk_size + length] for slot, length in enumerate(lengths)]
        ttnn.copy_host_to_device_tensor(host(pack_tokens(values, segments, chunk_size=chunk_size)), tokens)
        ttnn.copy_host_to_device_tensor(host(pack_positions(segments, chunk_size=chunk_size)), positions)

    def legacy_stage(slot, prefix, length):
        values = text[slot * chunk_size : slot * chunk_size + length] + [0] * (chunk_size - length)
        ttnn.copy_host_to_device_tensor(host(values), tokens)
        ttnn.copy_host_to_device_tensor(host(list(range(prefix, prefix + chunk_size))), positions)
        model.prefill_metadata.update(slot_idx=slot, kv_actual_global=prefix)

    def forward(packed):
        embedding = model.transform_and_embed_prefill_inputs_device(tokens)
        result = model(
            hidden_states=embedding, prefill_lengths=model.packed_prefill_metadata.layout if packed else None
        )
        ttnn.copy(result, hybrid.output)
        result.deallocate(True)

    def capture(packed):
        forward(packed)
        ttnn.synchronize_device(mesh_device)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            forward(packed)
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        return trace

    root = Path(os.getenv("GEMMA4_PERF_OUTPUT", "investigations/gemma4_hybrid_prefill"))
    root.mkdir(parents=True, exist_ok=True)
    report = dict(
        complete=False,
        started_at=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        model=hf_model,
        layers=len(model.layers),
        mesh=[8, 4],
        chunk_size=chunk_size,
        cases=cases,
        prefixes=prefixes,
        repeats=repeats,
        cache_history="Synthetic KV; current chunks are overwritten on every replay",
        device_metric="Maximum per-chip first-program-start to last-program-end span, including gaps",
        host_metric="Replay dispatch through final synchronization, without a Python profiler callback",
        excluded="Loading, compilation, capture, input/metadata staging, and warmup replays",
        baseline="Legacy pads each request to the full chunk and runs sequentially; both packed paths run one batch",
        results=[],
    )

    def save():
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "results.json")

    trace = None
    try:
        save()
        legacy_stage(0, 0, chunk_size)
        forward(False)
        ttnn.synchronize_device(mesh_device)
        packed_stage((chunk_size,), 0)
        hybrid.capture()
        hybrid_trace_ids = tuple(hybrid.traces)
        for path in ("legacy", "packed", "hybrid"):
            if path == "legacy":
                legacy_stage(0, 0, chunk_size)
                trace = capture(False)
            for name, lengths in cases.items():
                if path == "packed":
                    packed_stage(lengths, 0)
                    trace = capture(True)
                run = (
                    hybrid.run
                    if path == "hybrid"
                    else lambda: ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
                )
                for prefix in prefixes:
                    logger.info("[hybrid_perf] {} {} prefix={} warmup", path, name, prefix)
                    stage = (lambda slot: legacy_stage(slot, prefix, lengths[slot])) if path == "legacy" else None
                    if path != "legacy":
                        packed_stage(lengths, prefix)
                    count = len(lengths) if path == "legacy" else 1
                    host_us = _host_samples(mesh_device, run, repeats, count, stage)
                    captures = _device_samples(
                        mesh_device,
                        trace if path != "hybrid" else None,
                        repeats=repeats,
                        num_slots=count,
                        stage=stage,
                        run=run if path == "hybrid" else None,
                    )
                    device_us = [sum(c["device_us"] for c in captures if c["sample"] == i) for i in range(repeats)]
                    profiled_host_us = [
                        sum(c["profiled_replay_host_us"] for c in captures if c["sample"] == i) for i in range(repeats)
                    ]
                    row = dict(
                        path=path,
                        case=name,
                        request_lengths=lengths,
                        prefix=prefix,
                        real_tokens=sum(lengths),
                        device_us=_distribution(device_us),
                        device_us_per_token=statistics.median(device_us) / sum(lengths),
                        host_us=_distribution(host_us),
                        profiled_host_us=_distribution(profiled_host_us),
                        captures=captures,
                    )
                    report["results"].append(row)
                    save()
                    logger.info(
                        "[hybrid_perf] {} {} prefix={}: device={:.2f}ms host={:.2f}ms",
                        path,
                        name,
                        prefix,
                        statistics.median(device_us) / 1000,
                        statistics.median(host_us) / 1000,
                    )
                    assert tuple(hybrid.traces) == hybrid_trace_ids
                if path == "packed":
                    ttnn.release_trace(mesh_device, trace)
                    trace = None
            if trace is not None:
                ttnn.release_trace(mesh_device, trace)
                trace = None
        report["complete"] = True
    finally:
        save()
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        hybrid.close()
        tokens.deallocate(True)
        positions.deallocate(True)
    summary = "Device span (median):\n\n" + _summary(report, "device_us")
    summary += "\n\nHost replay latency without Python profiler callbacks (median):\n\n" + _summary(report, "host_us")
    summary += "\n\nEach value covers the entire request group. Hybrid / packed >1 means hybrid is slower.\n"
    summary += report["excluded"] + " are excluded. KV history is synthetic.\n"
    (root / "summary.md").write_text(summary)
    print("\n" + summary, flush=True)
