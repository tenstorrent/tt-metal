#!/usr/bin/env python3
"""Batch-32 traced-decode harness for the Phi-3.5 advisor challenger."""

from __future__ import annotations

import json
import os
import time

import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_optimized_decoder import (
    _hf_config,
    _synthetic_state_dict,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


BATCH = 32
CONTEXT = 32
REPEATS = int(os.getenv("CHALLENGER_REPEATS", "3"))
ITERS = int(os.getenv("CHALLENGER_ITERS", "50"))


def main() -> None:
    cfg = _hf_config()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64_000_000)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            _synthetic_state_dict(seed=23),
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh,
            max_position_embeddings=64,
            batch=BATCH,
        )
        kv_cache = OptimizedDecoder.allocate_paged_kv_cache(
            hf_config=cfg,
            mesh_device=mesh,
            max_batch_size=BATCH,
            max_seq_len=64,
            block_size=32,
        )
        page_table_host = torch.arange(BATCH * 2, dtype=torch.int32).reshape(BATCH, 2)
        page_table = ttnn.Tensor(page_table_host, ttnn.int32).to(mesh)
        current_pos_host = torch.full((BATCH,), CONTEXT, dtype=torch.int32)
        current_pos = ttnn.Tensor(current_pos_host, ttnn.int32).to(mesh)
        position_ids = ttnn.Tensor(current_pos_host.to(torch.uint32), ttnn.uint32).to(mesh)
        torch.manual_seed(29)
        hidden_host = torch.randn(1, 1, BATCH, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
        hidden = ttnn.Tensor(hidden_host, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(mesh)

        decoder.decode_forward(
            hidden,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=CONTEXT + 1,
        )
        ttnn.synchronize_device(mesh)

        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = decoder.decode_forward(
            hidden,
            current_pos=current_pos,
            position_ids=position_ids,
            page_table=page_table,
            kv_cache=kv_cache,
            rope_sequence_length=CONTEXT + 1,
        )
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)

        repeats_ms = []
        for repeat in range(REPEATS):
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(ITERS):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / ITERS
            repeats_ms.append(elapsed_ms)
            print(f"CHALLENGER_REPEAT_{repeat}_MS={elapsed_ms:.9f}", flush=True)

        signpost("PERF_DECODE")
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        if os.getenv("CHALLENGER_READ_DEVICE_PROFILER") == "1":
            ttnn.ReadDeviceProfiler(mesh)

        got = ttnn.to_torch(output)
        assert got.shape == (1, 1, BATCH, cfg.hidden_size)
        assert torch.isfinite(got).all()
        ttnn.release_trace(mesh, trace_id)
        print(json.dumps({"decode_batch": BATCH, "iters": ITERS, "repeats_ms": repeats_ms}))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
