# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dense->MoVA decoder stack contract without per-layer host/layout conversions."""

import argparse
import json

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
    ap.add_argument("--teacher-boundary", action="store_true")
    ap.add_argument("--independent-ccl", action="store_true")
    ap.add_argument("--dense-layer", type=int, default=2)
    ap.add_argument("--tag", default="stack")
    ap.add_argument("--options", type=json.loads, default={})
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
    root = ROOT / "doc/multichip_decoder"
    path = root / f"local/stack_debug_baseline_l{args.dense_layer}.pt"
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
        from models.common.tt_ccl import TT_CCL

        models = [
            klass.from_state_dict(
                sd,
                hf_config=cfg,
                layer_idx=layer,
                mesh_device=mesh,
                options=args.options,
                **({"ccl": TT_CCL(mesh)} if args.independent_ccl and not args.baseline else {}),
            )
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
        snapshots = {}
        teacher = torch.load(path, weights_only=True) if args.teacher_boundary else None
        route_tensors = {}
        original_route = models[1]._route

        def route(*args, **kwargs):
            out = original_route(*args, **kwargs)
            route_tensors[args[1]] = out
            route_tensors[args[1] + ".logits"] = args[0]
            return out

        models[1]._route = route

        def forward(inp, decode):
            for idx, (model, cache) in enumerate(zip(models, kv)):
                with device_only():
                    if decode:
                        inp = model.decode_forward(
                            inp, cos=dc, sin=di, page_table=table, current_pos=pos, kv_cache=cache
                        )
                    else:
                        inp = model.prefill_forward(inp, cos=co, sin=si, page_table=table, kv_cache=cache)
                snapshots[f"{decode}_{idx}"] = cpu(inp).clone()
                if idx == 1:
                    for name, tensor in route_tensors.items():
                        snapshots[f"{decode}_route_{name}"] = cpu(tensor).clone()
                elif teacher is not None:
                    inp = tt(teacher[f"{decode}_0"])
            return inp

        out = forward(tx, False)
        pre = cpu(out).clone()
        out.deallocate(True)
        for _ in range(4):
            out = forward(dx, True)
            dec = cpu(out).clone()
            out.deallocate(True)
        if args.baseline:
            torch.save(snapshots, path)
        expected = torch.load(path, weights_only=True)
        scores = {key: pcc(value, expected[key]) for key, value in snapshots.items() if key in expected}
        selections = {
            key: [torch.nonzero(row).flatten().tolist() for row in value.reshape(-1, value.shape[-1])[:2]]
            for key, value in snapshots.items()
            if "route" in key and not key.endswith(".logits")
        }
        result = dict(
            baseline=args.baseline,
            source=args.source,
            independent_ccl=args.independent_ccl,
            layers=layers,
            pcc=scores,
            selections=selections,
            teacher_boundary=args.teacher_boundary,
        )
        print("DIAGNOSTIC", result, flush=True)
        torch.save(snapshots, root / f"local/{args.tag}_snapshots.pt")
        (root / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
