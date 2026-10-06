# SPDX-License-Identifier: Apache-2.0
"""Profile exactly one real layer of each kind plus the real terminal path."""

import json
import os
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn

from ..tt.generator import build_generator
from .full_provenance import provenance


def main():
    assert not os.environ.get("TT_METAL_WATCHER")
    torch.set_num_threads(8)
    (
        Path(os.environ.get("FULL_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "doc/full_model"))
        / "profile_source.json"
    ).write_text(json.dumps(provenance(), indent=2) + "\n")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    timings = {}
    try:
        gen = build_generator(Path(__file__).resolve().parents[1], mesh, layer_indices=[0, 4])
        assert len(gen.model.layers) == 2
        original = gen._prefill_bucket

        def drained_prefill(*args, **kwargs):
            result = original(*args, **kwargs)
            if not ttnn.is_trace_capture_active(mesh):
                ttnn.ReadDeviceProfiler(mesh)
            return result

        # Drain between eager warm variants (never during capture) so setup
        # cannot overflow profiler buffers. Measured regions retain real calls.
        gen._prefill_bucket = drained_prefill
        gen.prepare()
        gen._prefill_bucket = original
        ttnn.ReadDeviceProfiler(mesh)
        gen.reset()
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_PREFILL")
        tick = time.perf_counter()
        gen.prefill_forward(
            torch.full((1, 128), 42, dtype=torch.long),
            page_table=gen.state.host_page_tables,
            kv_cache=gen.state,
            prompt_lens=[128],
        )
        ttnn.synchronize_device(mesh)
        timings["prefill_host_ms"] = (time.perf_counter() - tick) * 1000
        signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        gen.bind([42], [128])
        gen.replay()
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE")
        tick = time.perf_counter()
        gen.replay()
        gen.read_tokens()
        timings["decode_host_ms"] = (time.perf_counter() - tick) * 1000
        signpost("PERF_DECODE_END")
        ttnn.ReadDeviceProfiler(mesh)
        timings.update(
            layers=[0, 4],
            batch=1,
            prompt_len=128,
            decode_replays=1,
            capacity=gen.logical_capacity,
            trace_ids={k: str(v) for k, v in gen.traces.items()},
        )
        (
            Path(os.environ.get("FULL_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "doc/full_model"))
            / "profile_host_timing.json"
        ).write_text(json.dumps(timings, indent=2) + "\n")
        print("REDUCED_FULL_PROFILE_PASS", flush=True)
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
