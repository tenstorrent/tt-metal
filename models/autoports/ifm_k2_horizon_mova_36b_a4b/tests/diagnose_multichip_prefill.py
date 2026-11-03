# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched optimized baseline / TP4 decoder measurements and numerical checks."""

import argparse
import json
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
        y.deallocate(True)

        def snapshots():
            return {
                name: ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).clone()
                for name, value in model.debug_tensors.items()
            }

        expected = snapshots()

        def compare(label):
            actual = snapshots()
            differences = []
            for name, value in actual.items():
                other = expected[name]
                same = torch.equal(value, other)
                if not same:
                    differences.append(
                        dict(
                            name=name,
                            pcc=pcc(value, other),
                            max_abs=float((value.float() - other.float()).abs().max()),
                            rows=torch.where((value != other).any(dim=-1).reshape(-1))[0].tolist(),
                        )
                    )
            print("INTERMEDIATE_COMPARE", label, differences, flush=True)
            return not differences

        for i in range(30):
            y = model.prefill_forward(tx, **pk)
            okay = compare("eager_" + str(i))
            y.deallocate(True)
            if not okay:
                break
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        y = model.prefill_forward(tx, **pk)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for i in range(30):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            if not compare("trace_" + str(i)):
                break
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
