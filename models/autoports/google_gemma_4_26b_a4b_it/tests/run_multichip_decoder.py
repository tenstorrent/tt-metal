# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paired optimized TP1 / TP4 checks with identical real weights and inputs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=65)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--tp", type=int, choices=[1, 4])
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--repeat-input", action="store_true")
    parser.add_argument("--prefill-timing-samples", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--sharded-residual", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime_hash = hashlib.sha256(
        Path(__file__).parents[1].joinpath("tt/multichip_decoder.py").read_bytes()
    ).hexdigest()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    root = Path(__file__).parents[1]
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(config, args.layer, True)
    fixture = torch.load(root / f"doc/optimized_decoder/actual_text_layer{args.layer}_4096_128.pt", weights_only=True)
    source_prefill = fixture["prefill"]
    if args.repeat_input:
        source_prefill = source_prefill.repeat(
            1, (args.length + source_prefill.shape[1] - 1) // source_prefill.shape[1], 1
        )
    x = source_prefill[:, : args.length]
    assert x.shape[1] == args.length, "Long inputs require --repeat-input or a matching fixture"
    assert args.length + args.steps <= config.max_position_embeddings
    decode = fixture["decode"][:, : args.steps]
    extent = (args.length + args.steps + 1023) // 1024 * 1024
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    block = 32
    pages = extent // block
    table = torch.randperm(pages, dtype=torch.int32)[None]
    outputs = {}
    caches = {}
    timings = {}
    for tp, cls in ((1, OptimizedDecoder), (4, MultichipDecoder)):
        if args.tp is not None and tp != args.tp:
            continue
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED if tp == 1 else ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, tp), trace_region_size=16777216)
        try:
            mapper = ttnn.ReplicateTensorToMesh(mesh)

            def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
                return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper)

            def input_upload(value):
                return ttnn.from_torch(
                    value,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if tp == 4 and args.sharded_residual else mapper,
                )

            def read(value):
                if tp == 4 and args.sharded_residual:
                    return ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
                parts = [ttnn.to_torch(v).float() for v in ttnn.get_device_tensors(value)]
                assert all(torch.equal(parts[0], v) for v in parts[1:]), "Replicas differ"
                return parts[0]

            decoder = cls.from_state_dict(
                hf.state_dict(),
                hf_config=config,
                layer_idx=args.layer,
                mesh_device=mesh,
                **(
                    {
                        "sharded_residual": args.sharded_residual,
                        **({"expert_parallel": True} if args.expert_parallel else {}),
                    }
                    if tp == 4
                    else {}
                ),
            )
            cfg = decoder.layer.self_attn.config
            cache = [
                upload(torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim), ttnn.bfloat8_b)
                for _ in range(2)
            ]
            page_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            rope = tuple(upload(t.unsqueeze(0)) for t in (cos, sin))
            dx = input_upload(x.unsqueeze(0))
            with device_only():
                y = decoder.prefill_forward(dx, rope_mats=rope, page_table=page_table, kv_cache=cache)
            prefill = read(y)
            prefill_times = []
            for sample in range(args.prefill_timing_samples):
                ttnn.synchronize_device(mesh)
                if args.profile and sample == args.prefill_timing_samples - 1:
                    from tracy import signpost

                    signpost("PERF_PREFILL")
                before = time.perf_counter()
                with device_only():
                    y = decoder.prefill_forward(dx, rope_mats=rope, page_table=page_table, kv_cache=cache)
                ttnn.synchronize_device(mesh)
                prefill_times.append((time.perf_counter() - before) * 1e6)
                if args.profile and sample == args.prefill_timing_samples - 1:
                    signpost("PERF_PREFILL_END")
            timings[tp] = dict(prefill_host_us=prefill_times)
            rope_decode = tuple(upload(t.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for t in (cos, sin))
            decoded = []
            token = input_upload(decode[:, :1].unsqueeze(0))
            pos = upload(torch.tensor([[args.length]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            cache_pos = upload(torch.tensor([args.length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

            def forward():
                with device_only():
                    return decoder.decode_forward(
                        token,
                        rope_mats=rope_decode,
                        current_pos=pos,
                        cache_pos=cache_pos,
                        page_table=page_table,
                        kv_cache=cache,
                    )

            trace_id = None
            if args.trace:
                for _ in range(2):
                    y = forward()
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                y = forward()
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            decode_times = []
            if args.profile:
                signpost("PERF_DECODE")
            for step in range(args.steps):
                for value, dst, dtype, layout in (
                    (decode[:, step : step + 1].unsqueeze(0), token, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    (torch.tensor([[args.length + step]], dtype=torch.int32), pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    (
                        torch.tensor([args.length + step], dtype=torch.int32),
                        cache_pos,
                        ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    ),
                ):
                    host = ttnn.from_torch(
                        value,
                        dtype=dtype,
                        layout=layout,
                        mesh_mapper=(
                            ttnn.ShardTensorToMesh(mesh, dim=-1)
                            if dst is token and tp == 4 and args.sharded_residual
                            else mapper
                        ),
                    )
                    ttnn.copy_host_to_device_tensor(host, dst)
                ttnn.synchronize_device(mesh)
                before = time.perf_counter()
                if args.trace:
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                else:
                    y = forward()
                    ttnn.synchronize_device(mesh)
                decode_times.append((time.perf_counter() - before) * 1e6)
                decoded.append(read(y))
                if args.trace and not args.profile:
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    assert torch.equal(decoded[-1], read(y)), "Replay is not deterministic"
            if args.profile:
                signpost("PERF_DECODE_END")
            timings[tp]["decode_host_us"] = decode_times
            if trace_id is not None:
                ttnn.release_trace(mesh, trace_id)
            outputs[tp] = [prefill, *decoded]
            if args.check_cache:
                caches[tp] = [[ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(c)] for c in cache]
            print("TP_DONE", tp, flush=True)
        finally:
            ttnn.close_mesh_device(mesh)
    values = []
    for a, b in zip(outputs.get(1, []), outputs.get(4, [])):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        values.append(torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1].item())
    cache_pcc = []
    if len(caches) == 2:
        logical_end = args.length + args.steps
        for ref_ranks, actual_ranks in zip(caches[1], caches[4]):
            source = ref_ranks[0]
            for rank, actual in enumerate(actual_ranks):
                local_heads = actual.shape[1]
                head_start = rank * local_heads if config.layer_types[args.layer] == "sliding_attention" else rank // 2
                wanted = source[:, head_start : head_start + local_heads]

                def ordered(c):
                    return c[table[0].long()].permute(1, 0, 2, 3).reshape(local_heads, extent, -1)[:, :logical_end]

                a, b = ordered(wanted), ordered(actual)
                cache_pcc.append(torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1].item())
        assert min(cache_pcc) >= 0.995, cache_pcc
    result = dict(
        runtime_sha256=runtime_hash,
        command=sys.argv,
        expert_parallel=args.expert_parallel,
        cache_pcc=cache_pcc,
        layer_type=config.layer_types[args.layer],
        length=args.length,
        steps=args.steps,
        pcc=values,
        passed=min(values) >= 0.995 if values else None,
        trace=args.trace,
        timings=timings,
        all_replicas_equal=not args.sharded_residual,
        sharded_residual=args.sharded_residual,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {k: v for k, v in result.items() if k not in ("timings", "pcc")},
        "min_pcc",
        min(values) if values else None,
        flush=True,
    )
    if values:
        assert result["passed"]


if __name__ == "__main__":
    main()
