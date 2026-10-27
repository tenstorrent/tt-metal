# SPDX-License-Identifier: Apache-2.0
"""Compare the grouped recurrence candidate with the current split TP4 path."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.grouped_recurrence_candidate import fp32_recurrence as candidate
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import fp32_recurrence as baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads-per-core", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch <= 32:
        parser.error("batch must be within 1..32")
    torch.set_num_threads(8)
    torch.manual_seed(982)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024**2)
    traces = []
    result = dict(
        scope="synthetic real-shape FP32 recurrence on four chips; not a model timing",
        batch=args.batch,
        heads_per_core=args.heads_per_core,
        iterations=args.iterations,
        inputs="replicated normalized Q/K, per-row FP32 state, fixed gates; advancing state",
        allocation_tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
        program_cache_skip=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
        status="running",
    )
    try:

        def put(x):
            return ttnn.from_torch(
                x,
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        b = args.batch
        q, k = (put(torch.nn.functional.normalize(torch.randn(b, 1, 4, 128), dim=-1)) for _ in range(2))
        v = put(torch.randn(b, 1, 12, 128) * 0.1)
        g = put(-torch.rand(b, 1, 12) * 0.1)
        beta = put(torch.rand(b, 1, 12))
        initial = put(torch.randn(b, 12, 128, 128) * 0.01)
        states = [ttnn.clone(initial), ttnn.clone(initial)]
        outputs = [put(torch.zeros(b, 1, 12, 128)), put(torch.zeros(b, 1, 12, 128))]
        grid = mesh.compute_with_storage_grid_size()
        chunk = grid.x * grid.y // 12

        def current(state, output_buffer):
            outputs, next_states = [], []
            for start in range(0, b, chunk):
                end = min(start + chunk, b)
                output, next_state = baseline(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    g[start:end],
                    beta[start:end],
                    initial_state=state[start:end],
                )
                outputs.append(output)
                next_states.append(next_state)
            output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=0)
            next_state = next_states[0] if len(next_states) == 1 else ttnn.concat(next_states, dim=0)
            ttnn.copy(next_state, state)
            ttnn.copy(output, output_buffer)
            return output_buffer

        def grouped(state, output_buffer):
            output, next_state = candidate(q, k, v, g, beta, initial_state=state, heads_per_core=args.heads_per_core)
            ttnn.copy(next_state, state)
            ttnn.copy(output, output_buffer)
            return output_buffer

        runs = [("split", current), ("grouped", grouped)]
        for (_, run), state, output_buffer in zip(runs, states, outputs):
            for _ in range(2):
                run(state, output_buffer)
        ttnn.synchronize_device(mesh)
        for state in states:
            ttnn.copy(initial, state)
        for (_, run), state, output_buffer in zip(runs, states, outputs):
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traces.append(trace)
            try:
                run(state, output_buffer)
            finally:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for state in states:
            ttnn.copy(initial, state)
        semantics = []
        for step in range(3):
            for trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for name, tensors in [("output", outputs), ("state", states)]:
                for device, pair in enumerate(zip(*(ttnn.get_device_tensors(t) for t in tensors))):
                    left, right = [ttnn.to_torch(t) for t in pair]
                    diff = (left - right).abs().max().item()
                    exact = torch.equal(left, right)
                    assert exact, (step, name, device, diff)
                    semantics.append(dict(step=step + 1, tensor=name, device=device, exact=exact, max_abs_diff=diff))
        result["semantics"] = semantics
        timings = {}
        for (name, _), trace in zip(runs, traces):
            windows = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                windows.append((time.perf_counter() - start) * 1e6 / args.iterations)
            timings[name] = dict(replay_us=windows, median_replay_us=statistics.median(windows))
        result["timing"] = timings
        for name, tensors in [("output", outputs), ("state", states)]:
            for device, pair in enumerate(zip(*(ttnn.get_device_tensors(t) for t in tensors))):
                left, right = [ttnn.to_torch(t) for t in pair]
                assert torch.equal(left, right), ("after_timing", name, device, (left - right).abs().max().item())
        result["after_timing_exact"] = True
        result["speedup"] = timings["split"]["median_replay_us"] / timings["grouped"]["median_replay_us"]
        result["status"] = "pass"
    except BaseException as exc:
        result.update(status="fail", error=repr(exc))
        raise
    finally:
        root = Path(__file__).resolve().parents[1]
        sources = [
            Path(__file__),
            root / "tests/grouped_recurrence_candidate.py",
            root / "tt/multichip_decoder.py",
            *root.glob("tt/kernels/*grouped_recurrence.cpp"),
        ]
        result["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
    print(json.dumps({k: v for k, v in result.items() if k not in ("semantics", "source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
