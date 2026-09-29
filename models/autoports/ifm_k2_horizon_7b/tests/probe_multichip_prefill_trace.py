"""Bounded fixed-shape prefill tracing advice probe; reports host time only."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def run(args):
    import torch

    import ttnn

    from ..tt.multichip_decoder import MultichipDecoder
    from .run_functional import load_reference, pcc
    from .run_multichip import read, refresh, upload
    from .sweep_optimized import real_activations

    torch.set_grad_enabled(False)
    torch.set_num_threads(16)
    config, state, _, rope_fn = load_reference()
    x = real_activations(4096)[None]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    trace = None
    try:
        layer = MultichipDecoder.from_state_dict(state, hf_config=config, layer_idx=0, mesh_device=mesh)
        tx = upload(x.unsqueeze(0), mesh, shard=-1)
        rope = tuple(upload(r.unsqueeze(1), mesh) for r in rope_fn(x, torch.arange(4096)[None]))
        table = upload(torch.arange(132, dtype=torch.int32)[None], mesh, integer=True)
        cache = tuple(
            ttnn.zeros((132, 2, 32, 128), device=mesh, dtype=layer.kv_dtype, layout=ttnn.TILE_LAYOUT) for _ in range(2)
        )
        plan = layer.prepare_prefill(seq_len=4096)

        def prefill():
            return layer.prefill_forward(tx, rope=rope, kv_cache=cache, page_table=table, plan=plan)

        eager = prefill()
        ttnn.synchronize_device(mesh)
        reference = read(eager, mesh, -1)
        eager_us = []
        for _ in range(10):
            start = time.perf_counter()
            eager = prefill()
            ttnn.synchronize_device(mesh)
            eager_us.append((time.perf_counter() - start) * 1e6)
        # Complete eager references before capture: no new device allocations
        # may overlap the live trace's retained buffers.
        refresh(tx, x.roll(1, dims=1).unsqueeze(0), mesh, shard=-1)
        eager_changed = read(prefill(), mesh, -1)
        refresh(tx, x.unsqueeze(0), mesh, shard=-1)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = prefill()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        same = read(traced, mesh, -1)
        trace_us = []
        for _ in range(10):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            trace_us.append((time.perf_counter() - start) * 1e6)
        repeat = read(traced, mesh, -1)
        refresh(tx, x.roll(1, dims=1).unsqueeze(0), mesh, shard=-1)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        changed = read(traced, mesh, -1)
        result = {
            "scope": "fixed-shape TP4 prefill diagnostic; host latency only, no decode workload measured",
            "input_tokens": 4096,
            "batch": 1,
            "samples": 10,
            "eager_host_median_us": statistics.median(eager_us),
            "trace_host_median_us": statistics.median(trace_us),
            "same_input_pcc": pcc(same, reference),
            "changed_input_pcc": pcc(changed, eager_changed),
            "changed_input_outputs_differ": not torch.equal(same, changed),
            "repeat_bitwise": torch.equal(same, repeat),
            "implementation_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py").read_bytes()
            ).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        assert min(result["same_input_pcc"], result["changed_input_pcc"]) >= 0.995
        assert result["changed_input_outputs_differ"] and result["repeat_bitwise"]
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
