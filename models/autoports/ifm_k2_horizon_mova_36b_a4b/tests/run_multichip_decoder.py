# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched optimized baseline / TP4 decoder measurements and numerical checks."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .fused_reference import ROOT, pcc, reference
from .multichip_activations import recorded_inputs
from .multichip_test_utils import device_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=33)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--input-variant", type=int, default=0)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--prefill-iterations", type=int, default=5)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--trace-prefill", action="store_true")
    ap.add_argument("--options", type=json.loads, default={})
    ap.add_argument("--tag", default="initial")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--artifact-dir", type=Path)
    args = ap.parse_args()
    if args.profile and args.iterations != 1:
        ap.error("Profiler windows require --iterations 1; use nonprofile runs for repeated wall timing")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    baseline_root = ROOT / "doc/multichip_decoder"
    root = args.artifact_dir or baseline_root
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    (root / "local").mkdir(parents=True, exist_ok=True)
    c, hf, rope, sd = reference(args.layer)
    b, s = args.batch, args.seq
    x = recorded_inputs(args.layer, b, s + 1, args.input_variant)
    cs = rope(x, torch.arange(s).unsqueeze(0).expand(b, -1))
    dc = rope(x, torch.full((b, 1), s))
    blocks = (args.context + args.page_size - 1) // args.page_size
    pages = torch.randperm(b * blocks).reshape(b, blocks).int()
    baseline_path = baseline_root / f"local/baseline_l{args.layer}_b{b}_s{s}_c{args.context}_p{args.page_size}.pt"
    if args.input_variant:
        baseline_path = root / "local" / (baseline_path.stem + f"_v{args.input_variant}.pt")
    klass = OptimizedDecoder if args.baseline else MultichipDecoder
    if args.source:
        from importlib.machinery import SourceFileLoader

        klass = SourceFileLoader("multichip_candidate", str(args.source)).load_module().MultichipDecoder
    if not args.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[0] if args.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    tid = None
    try:
        ttnn.CONFIG.throw_exception_on_fallback = True

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
            mapper = ttnn.ReplicateTensorToMesh(mesh) if shard is None else ttnn.ShardTensorToMesh(mesh, dim=shard)
            return ttnn.from_torch(v.contiguous(), device=mesh, mesh_mapper=mapper, dtype=dtype, layout=layout)

        def cpu(v, shard=None):
            full = ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=shard if shard is not None else 0))
            return full if shard is not None or args.baseline else full.chunk(4, dim=0)[0]

        sharded = not args.baseline and args.options.get("residual_layout") == "sharded"
        model = klass.from_state_dict(
            sd,
            hf_config=c,
            layer_idx=args.layer,
            mesh_device=mesh,
            options=args.options,
            **({"chunk_size": args.chunk_size} if args.chunk_size is not None else {}),
        )
        table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kv = [
            tt(
                torch.zeros(b * blocks, 8, args.page_size, 128),
                model.kv_cache_dtype,
                shard=None if args.baseline else 1,
            )
            for _ in range(2)
        ]
        tx = tt(x[:, :s].unsqueeze(0), shard=3 if sharded else None)
        cos, sin = [tt(v.unsqueeze(1)) for v in cs]
        dx = tt(x[:, s:].reshape(1, 1, b, 2560), shard=3 if sharded else None)
        dcos, dsin = [tt(v.reshape(1, b, 1, 128)) for v in dc]
        pos = tt(torch.full((b,), s, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pk = dict(cos=cos, sin=sin, page_table=table, kv_cache=kv)
        dk = dict(cos=dcos, sin=dsin, page_table=table, current_pos=pos, kv_cache=kv)
        memory = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        memory_snapshot = {
            name: getattr(memory, name)
            for name in (
                "num_banks",
                "total_bytes_per_bank",
                "total_bytes_allocated_per_bank",
                "total_bytes_free_per_bank",
            )
        }
        print("MEMORY", memory_snapshot, flush=True)
        print("READY", flush=True)

        def drain():
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)

        drain()
        with device_only():
            y = model.prefill_forward(tx, **pk)
        yp = cpu(y, 3 if sharded else None).clone()
        if args.trace_prefill:
            yp_replicas = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).clone()
        y.deallocate(True)
        drain()
        with device_only():
            y = model.decode_forward(dx, **dk)
        yd = cpu(y, 3 if sharded else None).clone()
        y.deallocate(True)
        drain()
        caches = [cpu(v, None if args.baseline else 1).clone() for v in kv]
        result = dict(
            layer=args.layer,
            input_variant=args.input_variant,
            batch=b,
            seq=s,
            context=args.context,
            chunk_size=model.chunk_size,
            page_size=args.page_size,
            baseline=args.baseline,
            options=args.options,
            iterations=args.iterations,
            prefill_iterations=min(args.iterations, args.prefill_iterations),
            tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        )
        result["effective_options"] = model.options
        result["decode_weight_specs"] = {
            name: dict(shape=list(w.shape), dtype=str(w.dtype), memory=str(w.memory_config()))
            for name, w in model.decode_weights.items()
        }
        if args.baseline:
            torch.save(dict(prefill=yp, decode=yd, caches=caches), baseline_path)
        else:
            baseline = torch.load(baseline_path, weights_only=True)
            result["pcc"] = {k: pcc(v, baseline[k]) for k, v in [("prefill", yp), ("decode", yd)]}
            result["cache_pcc"] = [pcc(v, w) for v, w in zip(caches, baseline["caches"])]
            print("PCC", result["pcc"], "CACHE", result["cache_pcc"], flush=True)
            assert min(result["pcc"].values()) >= 0.995, result
            assert min(result["cache_pcc"]) >= 0.99, result
            result["weights"] = model.weight_manifest
        # Warm both CCL semaphore sets before capture, compile complete forward.
        for _ in range(3):
            y = model.decode_forward(dx, **dk)
            y.deallocate(True)
            drain()
        ttnn.synchronize_device(mesh)
        if args.trace:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                with device_only():
                    y = model.decode_forward(dx, **dk)
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traced = []
            for i in range(5):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                actual = cpu(y, 3 if sharded else None)
                traced.append(pcc(actual, yd))
                assert torch.equal(actual, yd), "Trace differs from eager"
                drain()
            result["trace_pcc"] = traced
        if args.profile:
            from tracy import signpost
        else:

            def signpost(*a):
                pass

        signpost("PERF_DECODE")
        begin = time.perf_counter()
        for _ in range(args.iterations):
            if tid is not None:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            else:
                out = model.decode_forward(dx, **dk)
                out.deallocate(True)
        ttnn.synchronize_device(mesh)
        result["decode_ms"] = (time.perf_counter() - begin) * 1000 / args.iterations
        signpost("PERF_DECODE_END")
        drain()
        if tid is not None:
            ttnn.release_trace(mesh, tid)
            tid = None
            y.deallocate(True)
        y = model.prefill_forward(tx, **pk)
        y.deallocate(True)
        ttnn.synchronize_device(mesh)
        drain()
        signpost("PERF_PREFILL")
        begin = time.perf_counter()
        for _ in range(min(args.iterations, args.prefill_iterations)):
            out = model.prefill_forward(tx, **pk)
            out.deallocate(True)
        ttnn.synchronize_device(mesh)
        result["prefill_ms"] = (time.perf_counter() - begin) * 1000 / min(args.iterations, args.prefill_iterations)
        signpost("PERF_PREFILL_END")
        drain()
        if args.trace_prefill:
            if args.profile:
                raise ValueError("Prefill trace advice trials are separate from profiler collection")
            warm_outputs = []
            for _ in range(10):
                warm = model.prefill_forward(tx, **pk)
                warm_full = ttnn.to_torch(warm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                assert torch.equal(warm_full, yp_replicas), "Repeated eager prefill differs on one or more ranks"
                warm_cpu = cpu(warm, 3 if sharded else None).clone()
                warm_outputs.append(warm_cpu)
                print(
                    "PREFILL_WARM_COMPARE",
                    dict(
                        pcc=pcc(warm_cpu, yp),
                        equal=torch.equal(warm_cpu, yp),
                        max_abs=float((warm_cpu.float() - yp.float()).abs().max()),
                        different_rows=torch.where((warm_cpu != yp).any(dim=-1).reshape(-1))[0].tolist(),
                    ),
                    flush=True,
                )
                assert torch.equal(warm_cpu, yp), "Repeated eager prefill differs from initial"
                warm.deallocate(True)
            ttnn.synchronize_device(mesh)
            capture_begin = time.perf_counter()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                with device_only():
                    y = model.prefill_forward(tx, **pk)
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            result["prefill_trace_capture_ms"] = (time.perf_counter() - capture_begin) * 1000
            for _ in range(100):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                replay_full = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                assert torch.equal(replay_full, yp_replicas), "Prefill trace differs on one or more ranks"
                replay_cpu = cpu(y, 3 if sharded else None)
                print(
                    "PREFILL_TRACE_COMPARE",
                    dict(
                        initial_pcc=pcc(replay_cpu, yp),
                        warmed_pcc=pcc(replay_cpu, warm_outputs[-1]),
                        warmed_equal=torch.equal(replay_cpu, warm_outputs[-1]),
                        max_abs=float((replay_cpu.float() - warm_outputs[-1].float()).abs().max()),
                    ),
                    flush=True,
                )
                assert torch.equal(replay_cpu, yp), "Prefill trace differs from eager"
            # Frozen semaphore handles must also survive queued replay boundaries.
            for burst in range(5):
                for _ in range(20):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                queued = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                assert torch.equal(queued, yp_replicas), "Queued prefill trace differs on one or more ranks"
                print("PREFILL_QUEUED_COMPARE", dict(burst=burst, replays=20, all_ranks_equal=True), flush=True)
            count = min(args.iterations, args.prefill_iterations)
            begin = time.perf_counter()
            for _ in range(count):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            result["traced_prefill_ms"] = (time.perf_counter() - begin) * 1000 / count
            timed_output = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            assert torch.equal(timed_output, yp_replicas), "Timed queued prefill output differs"
            result[
                "prefill_trace_determinism"
            ] = "all four ranks bitwise; ten eager repeats; 100 blocking replays; five queued bursts of20 and timed burst checked"
            ttnn.release_trace(mesh, tid)
            tid = None
            y.deallocate(True)
        result["memory"] = memory_snapshot
        result["profile_samples"] = args.iterations if args.profile else 0
        result["implementation_sources"] = {}
        for idx, base in enumerate(klass.__mro__):
            if base is object:
                continue
            source_path = Path(__import__("inspect").getfile(base))
            if "autoports" not in str(source_path):
                continue
            content = source_path.read_bytes()
            result["implementation_sources"][str(source_path)] = hashlib.sha256(content).hexdigest()
            (root / f"candidates/{args.tag}_dependency{idx}.py.txt").write_bytes(content)
        result["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["source_sha256"] = hashlib.sha256(Path(__import__("inspect").getfile(klass)).read_bytes()).hexdigest()
        target = root / f"{args.tag}_l{args.layer}_b{b}_s{s}.json"
        target.write_text(json.dumps(result, indent=2) + "\n")
        print("PASS", target, result["prefill_ms"], result["decode_ms"], flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
