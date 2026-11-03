# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only real checkpoint validation; host work is outside decoder calls."""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder

MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
REVISION = "e9ce717642b2074082875be233c02c729ba47960"
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots" / REVISION


from .fused_reference import reference


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack((a, b)))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--options-file", type=Path)
    ap.add_argument("--input-variant", type=int, default=0)
    ap.add_argument("--real-inputs", action="store_true")
    ap.add_argument("--options", type=json.loads, default={})
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args()
    if args.options_file:
        args.options = json.loads(args.options_file.read_text())
    if args.baseline:
        from ..tt.fused_decoder import FusedDecoder

        decoder_class = FusedDecoder
    else:
        decoder_class = OptimizedDecoder
    if args.source:
        from importlib.machinery import SourceFileLoader

        decoder_class = SourceFileLoader("optimization_candidate", str(args.source)).load_module().OptimizedDecoder
    source_path = Path(inspect.getfile(decoder_class))
    source = source_path.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    implementation_sources = {}
    for index, klass in enumerate(decoder_class.__mro__):
        if klass is object:
            continue
        path = Path(inspect.getfile(klass))
        if "autoports" not in str(path):
            continue
        content = path.read_bytes()
        implementation_sources[str(path)] = hashlib.sha256(content).hexdigest()
        (ROOT / f"doc/optimized_decoder/candidates/{args.tag}_dependency{index}.py.txt").write_bytes(content)
    (ROOT / f"doc/optimized_decoder/candidates/{args.tag}_source.py.txt").write_bytes(source)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    c, hf, rope, sd = reference(args.layer)
    b, s = args.batch, args.seq
    if args.real_inputs:
        from .optimized_activations import recorded_inputs

        x = recorded_inputs(args.layer, b, s + 1, args.input_variant)
    else:
        x = (torch.randn(b, s + 1, c.hidden_size) * 0.1).bfloat16()
    cache = DynamicCache(config=c)
    with torch.no_grad():
        cs = rope(x, torch.arange(s).unsqueeze(0).expand(b, -1))
        mask = torch.full((s, s), float("-inf")).triu(1)
        y = hf(x[:, :s], position_embeddings=cs, attention_mask=mask[None, None], past_key_values=cache)
        dcs = rope(x, torch.full((b, 1), s))
        yd = hf(x[:, s:], position_embeddings=dcs, past_key_values=cache)
    print("HF_READY", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(v, device=mesh, dtype=dtype, layout=layout)

        def cpu(v):
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        model = decoder_class.from_state_dict(
            sd,
            hf_config=c,
            layer_idx=args.layer,
            mesh_device=mesh,
            **(
                {}
                if args.baseline
                else {
                    "options": args.options,
                    **({"chunk_size": args.options["chunk_size"]} if "chunk_size" in args.options else {}),
                }
            ),
        )
        blocks = (args.context + 31) // 32
        pages = torch.randperm(b * blocks).reshape(b, blocks).int()
        table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kv = [
            tt(
                torch.zeros(b * blocks, 8, 32, 128, dtype=torch.bfloat16),
                getattr(model, "kv_cache_dtype", getattr(ttnn, args.options.get("cache_dtype", "bfloat16"))),
            )
            for _ in range(2)
        ]
        tx = tt(x[:, :s].unsqueeze(0))
        cos, sin = [tt(v.unsqueeze(1)) for v in cs]
        dx = tt(x[:, s:].reshape(1, 1, b, c.hidden_size))
        dc, ds = [tt(v.reshape(1, b, 1, 128)) for v in dcs]
        pos = tt(torch.full((b,), s, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(cos=dc, sin=ds, page_table=table, current_pos=pos, kv_cache=kv)
        print("TT_READY", flush=True)
        out = model.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=kv)
        pre = pcc(y, cpu(out).reshape(b, s, -1))
        print("PREFILL_PCC", pre, flush=True)
        assert pre >= 0.995
        outd = model.decode_forward(dx, **kwargs)
        dec = pcc(yd, cpu(outd).reshape(b, 1, -1))
        print("DECODE_PCC", dec, flush=True)
        assert dec >= 0.995
        if args.profile:
            from tracy import signpost

            warm = model.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=kv)
            warm.deallocate(True)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            t0 = time.perf_counter()
            for _ in range(args.iterations):
                measured = model.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=kv)
                measured.deallocate(True)
            ttnn.synchronize_device(mesh)
            prefill_ms = (time.perf_counter() - t0) * 1000 / args.iterations
            signpost("PERF_PREFILL_END")
        if args.trace:
            outd.deallocate(True)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                outd = model.decode_forward(dx, **kwargs)
            finally:
                # End even a failed capture before device teardown.
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for i in range(3):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                traced = pcc(yd, cpu(outd).reshape(b, 1, -1))
                print("TRACE_PCC", i, traced, flush=True)
                assert traced >= 0.995
            if args.profile:
                ttnn.synchronize_device(mesh)
                signpost("PERF_DECODE")
                t0 = time.perf_counter()
                for _ in range(args.iterations):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                decode_ms = (time.perf_counter() - t0) * 1000 / args.iterations
                signpost("PERF_DECODE_END")
                print("WARM_WALL_MS", prefill_ms, decode_ms, flush=True)
                (ROOT / f"doc/optimized_decoder/candidates/{args.tag}_layer{args.layer}.json").write_text(
                    json.dumps(
                        dict(
                            source_sha256=source_sha,
                            implementation_sources=implementation_sources,
                            options=args.options,
                            effective_options=getattr(model, "options", {}),
                            activation_source="checkpoint-HF" if args.real_inputs else "random-real-weights",
                            input_variant=args.input_variant,
                            layer=args.layer,
                            batch=b,
                            prefill_length=s,
                            decode_position=s,
                            warmed_prefill_wall_ms=prefill_ms,
                            traced_warmed_decode_wall_ms=decode_ms,
                            implementation=decoder_class.__module__,
                            weight_contract={
                                name: dict(shape=list(w.shape), dtype=str(w.dtype), memory=str(w.memory_config()))
                                for name, w in model.weights.items()
                            },
                            decode_weight_contract={
                                name: dict(shape=list(w.shape), dtype=str(w.dtype), memory=str(w.memory_config()))
                                for name, w in getattr(model, "decode_weights", {}).items()
                            },
                            attention_configuration=(
                                [str(v) for v in model._decode_attention_config(b, table, kv[0])]
                                if hasattr(model, "_decode_attention_config")
                                else None
                            ),
                            measured_prefill_calls=args.iterations,
                            measured_decode_replays=args.iterations,
                            blocking=False,
                            prefill_pcc=pre,
                            traced_decode_pcc=traced,
                            environment={
                                key: os.environ.get(key)
                                for key in (
                                    "TT_METAL_TRACE_ALLOC_TRACKING",
                                    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                                    "TT_METAL_WATCHER",
                                    "TTNN_CONFIG_OVERRIDES",
                                )
                            },
                            runner_argv=sys.argv,
                            decode_window_host_counters=dict(
                                trace_replays=args.iterations,
                                input_refreshes=0,
                                current_position_rope_refreshes=0,
                                page_table_refreshes=0,
                                synchronizations=1,
                                readbacks=0,
                                positions_advanced=False,
                            ),
                        ),
                        indent=2,
                    )
                )
            fused_prefill_cpu, fused_decode_cpu = cpu(out), cpu(outd)
            ttnn.release_trace(mesh, tid)
        if not args.baseline and args.trace and args.profile:
            from ..tt.fused_decoder import FusedDecoder

            control = FusedDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
            control_kv = (
                kv
                if kv[0].dtype == ttnn.bfloat16
                else [tt(torch.zeros(b * blocks, 8, 32, 128, dtype=torch.bfloat16)) for _ in range(2)]
            )
            cp = control.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=control_kv)
            cd = control.decode_forward(dx, **{**kwargs, "kv_cache": control_kv})
            equivalence = dict(prefill=pcc(fused_prefill_cpu, cpu(cp)), decode=pcc(fused_decode_cpu, cpu(cd)))
            print("FUNCTIONAL_EQUIVALENCE", equivalence, flush=True)
            assert min(equivalence.values()) >= 0.995
            path = ROOT / f"doc/optimized_decoder/candidates/{args.tag}_layer{args.layer}.json"
            report = json.loads(path.read_text())
            report["functional_equivalence_pcc"] = equivalence
            path.write_text(json.dumps(report, indent=2))
        print("PASS", vars(args), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
