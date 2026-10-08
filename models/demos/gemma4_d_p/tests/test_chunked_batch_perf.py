# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed 4x1K batching and the canonical 1x4K path at equal prefix lengths.

GEMMA4_BATCH_PERF_MODE=canonical or chunked4 (default), separate processes.
All histories are populated by real model calls; no random or synthesized KV.
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gemma4_d_p.demo.text_demo_prefill import _get_prefill_tokens
from models.demos.gemma4_d_p.tests.test_chunked_batch import build_model
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.chunked_batch import ChunkedRequest
from models.demos.gemma4_d_p.tt.runners.chunked_batch_runtime import ChunkedBatchRuntime


class CanonicalRuntime:
    def __init__(self, model):
        self.model = model
        self.device = model.mesh_device
        self.chunk_size = model.prefill_chunk_size
        self.input = ttnn.to_device(self.host([0] * self.chunk_size), self.device)
        self.positions = ttnn.to_device(self.host(range(self.chunk_size)), self.device)
        model.set_prefill_rope_positions(self.positions)
        model._prefill_metadata_external = True
        model.prefill_metadata.update(slot_idx=0, kv_actual_global=0)
        warmup = self.forward()
        ttnn.synchronize_device(self.device)
        warmup.deallocate(True)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output = self.forward()
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def host(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32).reshape(1, self.chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, (8, 4), dims=(1, None)),
        )

    def forward(self):
        return self.model(self.model.transform_and_embed_prefill_inputs_device(self.input))

    def stage(self, requests):
        (req,) = requests
        ttnn.copy_host_to_device_tensor(self.host(req.token_ids), self.input)
        ttnn.copy_host_to_device_tensor(self.host(range(req.actual_start, req.actual_end)), self.positions)
        self.model.prefill_metadata.update(slot_idx=0, kv_actual_global=req.actual_start)

    def execute(self):
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)

    def close(self):
        ttnn.release_trace(self.device, self.trace_id)


@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": 256_000_000})
def test_chunked_batch_perf(mesh_device):
    mode = os.environ.get("GEMMA4_BATCH_PERF_MODE", "chunked4")
    assert mode in ("canonical", "chunked4")
    lanes, chunk = (1, 4096) if mode == "canonical" else (4, 1024)
    context = int(os.environ.get("GEMMA4_BATCH_PERF_CONTEXT", "262144"))
    layers = int(os.environ.get("GEMMA4_BATCH_TEST_LAYERS", "60"))
    repeats = int(os.environ.get("GEMMA4_BATCH_PERF_REPEATS", "5"))
    selected = {n * 1024 for n in (0, 8, 32, 64, 128, 192, 248)}
    selected.add(context - 8192)
    model, caches = build_model(mesh_device, chunk_size=chunk, num_slots=lanes, context_len=context, num_layers=layers)
    source = _get_prefill_tokens(os.environ["HF_MODEL"], context, model.vocab_size)[0]
    prompts = [torch.roll(source, shifts=lane * 701).tolist() for lane in range(lanes)]
    runtime = CanonicalRuntime(model) if mode == "canonical" else ChunkedBatchRuntime(model, num_slots=lanes)
    if mode == "chunked4":
        runtime.capture()
    records = []

    def run_batch(lengths, *, starts, request_base=0, label="full"):
        requests = tuple(
            ChunkedRequest(request_base + lane, lane, start, tuple(prompts[lane][start : start + length]))
            for lane, (start, length) in enumerate(zip(starts, lengths))
        )
        stage_start = time.perf_counter()
        runtime.stage(requests)
        staging_ms = (time.perf_counter() - stage_start) * 1000
        samples = []
        for _ in range(repeats if starts[0] in selected else 1):
            start_time = time.perf_counter()
            runtime.execute()
            samples.append((time.perf_counter() - start_time) * 1000)
        median = statistics.median(samples)
        row = dict(
            label=label,
            starts=starts,
            ends=[r.actual_end for r in requests],
            lengths=lengths,
            useful_tokens=sum(lengths),
            padded_tokens=lanes * chunk,
            samples_ms=samples,
            median_ms=median,
            staging_ms=staging_ms,
            first_call_wall_ms=staging_ms + samples[0],
            useful_tokens_per_second=sum(lengths) * 1000 / median,
        )
        records.append(row)
        print(
            f"BATCH {mode} {label} starts={starts} lengths={lengths} device={median:.2f}ms "
            f"stage={staging_ms:.2f}ms useful={row['useful_tokens_per_second']:.0f}tok/s repeats={len(samples)}",
            flush=True,
        )

    try:
        for start in range(0, context, chunk):
            run_batch([chunk] * lanes, starts=[start] * lanes)
        if mode == "chunked4":
            run_batch([512] * 4, starts=[0] * 4, request_base=100, label="half_full")
            run_batch([1024, 512, 128, 32], starts=[0] * 4, request_base=200, label="uneven_final")
        output = Path(os.environ.get("GEMMA4_BATCH_PERF_OUTPUT", f"/tmp/gemma4-chunked-batching/{mode}.json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                dict(
                    mode=mode,
                    layers=layers,
                    batch_size=lanes,
                    chunk_size=chunk,
                    allocated_slots=lanes,
                    context=context,
                    repeats=repeats,
                    history="real model calls",
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )
        print(f"Measurements: {output}", flush=True)
    finally:
        runtime.close()
