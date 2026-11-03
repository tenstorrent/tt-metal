# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense->MoVA decoder stack contract without per-layer host/layout conversions."""

import argparse
import hashlib
import inspect
import json
import os
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
    ap.add_argument("--source")
    ap.add_argument("--dense-layer", type=int, default=2)
    ap.add_argument("--tag", default="stack")
    ap.add_argument("--options", type=json.loads, default={})
    ap.add_argument("--artifact-dir", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(145)
    layers = (args.dense_layer, 3)
    refs = [reference(layer) for layer in layers]
    c, _, rope, _ = refs[0]
    x = recorded_inputs(args.dense_layer, 2, 34)
    cs = rope(x, torch.arange(33)[None].expand(2, -1))
    ds = rope(x, torch.tensor([[33], [29]]))
    pages = torch.randperm(16).reshape(2, 8).int()
    baseline_root = ROOT / "doc/multichip_decoder"
    root = args.artifact_dir or baseline_root
    root.mkdir(parents=True, exist_ok=True)
    path = baseline_root / f"local/stack_baseline_l{args.dense_layer}.pt"
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
        sharded = not args.baseline and args.options.get("residual_layout") == "sharded"

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, shard=None):
            return ttnn.from_torch(
                v.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                mesh_mapper=(
                    ttnn.ReplicateTensorToMesh(mesh) if shard is None else ttnn.ShardTensorToMesh(mesh, dim=shard)
                ),
            )

        def cpu(v):
            result = ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3 if sharded else 0))
            return result if args.baseline or sharded else result[:1]

        klass = OptimizedDecoder if args.baseline else MultichipDecoder
        if args.source:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("stack_candidate", args.source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            klass = module.MultichipDecoder
        models = [
            klass.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh, options=args.options)
            for layer, (cfg, _, _, sd) in zip(layers, refs)
        ]
        kv = [
            [tt(torch.zeros(16, 8, 32, 128), ttnn.bfloat8_b, shard=None if args.baseline else 1) for _ in range(2)]
            for model in models
        ]
        table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        tx = tt(x[:, :33].unsqueeze(0), shard=3 if sharded else None)
        co, si = [tt(v.unsqueeze(1)) for v in cs]
        dx = tt(x[:, 33:].reshape(1, 1, 2, 2560), shard=3 if sharded else None)
        dc, di = [tt(v.reshape(1, 2, 1, 128)) for v in ds]
        pos = tt(torch.tensor([33, 29], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def forward(inp, decode):
            with device_only():
                for model, cache in zip(models, kv):
                    if decode:
                        inp = model.decode_forward(
                            inp, cos=dc, sin=di, page_table=table, current_pos=pos, kv_cache=cache
                        )
                    else:
                        inp = model.prefill_forward(inp, cos=co, sin=si, page_table=table, kv_cache=cache)
            return inp

        out = forward(tx, False)
        pre = cpu(out).clone()
        out.deallocate(True)
        for _ in range(4):
            out = forward(dx, True)
            dec = cpu(out).clone()
            out.deallocate(True)
        if args.baseline:
            torch.save({"prefill": pre, "decode": dec}, path)
        expected = torch.load(path, weights_only=True)
        scores = {k: pcc(v, expected[k]) for k, v in [("prefill", pre), ("decode", dec)]}
        assert min(scores.values()) >= 0.995, scores
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            traced = forward(dx, True)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(10):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            assert torch.equal(cpu(traced), dec)
        result = dict(
            baseline=args.baseline,
            options=args.options,
            pcc=scores,
            layers=list(layers),
            batch=2,
            seq=33,
            decode_positions=[33, 29],
            trace_replays=10,
            layer_boundary_host_or_layout_conversions=0,
            trace_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
            source_sha256=hashlib.sha256(Path(inspect.getfile(klass)).read_bytes()).hexdigest(),
        )
        (root / f'{args.tag}_{"baseline" if args.baseline else "tp4"}.json').write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print("PASS", result, flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
