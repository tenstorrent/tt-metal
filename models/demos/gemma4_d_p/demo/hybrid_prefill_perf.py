# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One full legacy request versus one full hybrid batch of mixed requests."""

import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from loguru import logger
from tabulate import tabulate

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.prefill_perf import _device_samples, _distribution
from models.demos.gemma4_d_p.demo.text_demo_prefill import TRACE_REGION_SIZE, _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.hybrid_prefill import Gemma4HybridPrefill
from models.demos.gemma4_d_p.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4_d_p.tt.runners.kv_caches import allocate_ring_kv_caches
from models.demos.gemma4_d_p.tt.runners.packing import PrefillSegment, pack_positions, pack_tokens


def _summary(report):
    rows = []
    for prefix in report["prefixes"]:
        measurements = {
            row["path"]: row["device_us"]["median"] / 1000 for row in report["results"] if row["prefix"] == prefix
        }
        rows.append([f"{prefix:,}", *(f"{measurements[path]:.2f}" for path in ("legacy", "hybrid"))])
    return tabulate(
        rows,
        headers=["Prefix per request", "Legacy ms", "Hybrid ms"],
        tablefmt="github",
        disable_numparse=True,
        colalign=("right", "right", "right"),
    )


@torch.no_grad()
@pytest.mark.timeout(0)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_prefill_hybrid_comparison(mesh_device, reset_seeds):
    assert ttnn.device.IsProgramRealtimeProfilerActive(), "Requires the real-time device profiler"
    chunk_size = int(os.getenv("GEMMA4_PERF_CHUNK_SIZE", "8192"))
    repeats = int(os.getenv("GEMMA4_PERF_DEVICE_REPEATS", "5"))
    assert 0 < chunk_size <= 262144 and chunk_size % 1024 == 0 and repeats > 0
    mixed_lengths = tuple(
        int(n) for n in os.getenv("GEMMA4_PERF_REQUEST_LENGTHS", "512,512,1024,1536,2048,2560").split(",")
    )
    assert all(n > 0 and n % 256 == 0 for n in mixed_lengths) and sum(mixed_lengths) == chunk_size
    last_prefix = (262144 - chunk_size) // chunk_size * chunk_size
    prefixes = (0, last_prefix) if last_prefix else (0,)
    requests = {"legacy": (chunk_size,), "hybrid": mixed_lengths}
    num_users = len(mixed_lengths)
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
    text = _get_prefill_tokens(hf_model, chunk_size, args.vocab_size)[0].tolist()
    mixed_tokens, offset = [], 0
    for length in mixed_lengths:
        mixed_tokens.append(text[offset : offset + length])
        offset += length

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

    def hybrid_stage(prefix):
        segments = tuple(PrefillSegment(slot, prefix, prefix + length) for slot, length in enumerate(mixed_lengths))
        model.packed_prefill_metadata.update(
            (s.slot_id, s.actual_start, s.actual_end - s.actual_start, s.padded_length()) for s in segments
        )
        ttnn.copy_host_to_device_tensor(host(pack_tokens(mixed_tokens, segments, chunk_size=chunk_size)), tokens)
        ttnn.copy_host_to_device_tensor(host(pack_positions(segments, chunk_size=chunk_size)), positions)

    def legacy_stage(prefix):
        ttnn.copy_host_to_device_tensor(host(text), tokens)
        ttnn.copy_host_to_device_tensor(host(list(range(prefix, prefix + chunk_size))), positions)
        model.prefill_metadata.update(slot_idx=0, kv_actual_global=prefix)

    def legacy_forward():
        embedding = model.transform_and_embed_prefill_inputs_device(tokens)
        result = model(hidden_states=embedding)
        ttnn.copy(result, hybrid.output)
        result.deallocate(True)

    root = Path(os.getenv("GEMMA4_PERF_OUTPUT", "investigations/gemma4_hybrid_8k_comparison"))
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
        requests=requests,
        prefixes=prefixes,
        repeats=repeats,
        cache_history="Synthetic KV; current chunks are overwritten on every replay",
        device_metric="Maximum per-chip first-program-start to last-program-end span, including gaps",
        excluded="Loading, compilation, capture, input/metadata staging, and warmup replays",
        baseline="One full legacy request versus one hybrid batch; both process the same total number of tokens",
        results=[],
    )

    def save():
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(root / "results.json")

    trace = None
    try:
        save()
        legacy_stage(0)
        legacy_forward()
        ttnn.synchronize_device(mesh_device)
        hybrid_stage(0)
        hybrid.capture()
        hybrid_trace_ids = tuple(hybrid.traces)
        legacy_stage(0)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            legacy_forward()
        finally:
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        for path, lengths in requests.items():
            for prefix in prefixes:
                logger.info("[hybrid_perf] {} requests={} prefix={} warmup", path, lengths, prefix)
                (legacy_stage if path == "legacy" else hybrid_stage)(prefix)
                captures = _device_samples(
                    mesh_device,
                    trace if path == "legacy" else None,
                    repeats=repeats,
                    run=hybrid.run if path == "hybrid" else None,
                )
                device_us = [c["device_us"] for c in captures]
                report["results"].append(
                    dict(
                        path=path,
                        request_lengths=lengths,
                        prefix=prefix,
                        real_tokens=sum(lengths),
                        device_us=_distribution(device_us),
                        device_us_per_token=statistics.median(device_us) / sum(lengths),
                        captures=captures,
                    )
                )
                save()
                logger.info(
                    "[hybrid_perf] {} prefix={}: device={:.2f}ms", path, prefix, statistics.median(device_us) / 1000
                )
                assert tuple(hybrid.traces) == hybrid_trace_ids
        report["complete"] = True
    finally:
        save()
        if trace is not None:
            ttnn.release_trace(mesh_device, trace)
        hybrid.close()
        tokens.deallocate(True)
        positions.deallocate(True)
    summary = f"Legacy: one {chunk_size}-token request. Hybrid: {' + '.join(map(str, mixed_lengths))} = {chunk_size} tokens.\n\n"
    summary += f"Median device time over {repeats} warmed replays, including gaps between programs:\n\n" + _summary(
        report
    )
    summary += (
        "\n\nEach request uses the indicated prefix. "
        + report["excluded"]
        + " are excluded. KV history is synthetic.\n"
    )
    (root / "summary.md").write_text(summary)
    print("\n" + summary, flush=True)
