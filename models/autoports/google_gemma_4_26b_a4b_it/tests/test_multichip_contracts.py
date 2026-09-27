# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight paged slot, prefix, and heterogeneous batch contract comparisons."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def pcc(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--base-length", type=int, default=32)
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--hybrid-experts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-shared", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared-geometry", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--grouped-moe-reduce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sliding-sharded-rope", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output-fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="LoFi")
    parser.add_argument("--qkv-fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="LoFi")
    parser.add_argument("--attention-ccl-dtype", choices=["float32", "bfloat16", "bfloat8_b"], default="bfloat16")
    parser.add_argument("--full-attention-ccl-dtype", choices=["float32", "bfloat16", "bfloat8_b"], default="bfloat8_b")
    parser.add_argument("--attention-precision", choices=["baseline", "qkv", "output", "both"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expert_parallel and args.hybrid_experts:
        parser.error("--expert-parallel requires --no-hybrid-experts")
    if args.shared_geometry and not args.optimized_shared:
        parser.error("--no-optimized-shared requires --shared-geometry 0")
    torch.set_num_threads(8)
    torch.manual_seed(99)
    root = Path(__file__).parents[1]
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(config, args.layer, True)
    fixture = torch.load(root / f"doc/optimized_decoder/actual_text_layer{args.layer}_4096_128.pt", weights_only=True)
    batch = args.batch
    assert 1 <= batch <= 32 and args.base_length >= 32
    lengths = [args.base_length + slot for slot in range(batch)]
    extent = (max(lengths) + 1 + 127) // 128 * 128
    block = 32
    stride = args.base_length + 64
    assert (batch - 1) * stride + max(lengths) < fixture["prefill"].shape[1]
    inputs = [fixture["prefill"][:, slot * stride : slot * stride + length] for slot, length in enumerate(lengths)]
    tokens = torch.cat(
        [
            fixture["prefill"][:, slot * stride + length : slot * stride + length + 1]
            for slot, length in enumerate(lengths)
        ],
        dim=1,
    )
    table = torch.randperm(batch * extent // block, dtype=torch.int32).reshape(batch, -1)
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        inputs[0], torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    results = {}
    preservation = {}
    for tp, cls in ((1, OptimizedDecoder), (4, MultichipDecoder)):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED if tp == 1 else ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, tp), trace_region_size=67108864)
        try:
            mapper = ttnn.ReplicateTensorToMesh(mesh)

            def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
                return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper)

            def read(x):
                parts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(x)]
                assert all(torch.equal(parts[0], part) for part in parts[1:])
                return parts[0]

            decoder = cls.from_state_dict(
                hf.state_dict(),
                hf_config=config,
                layer_idx=args.layer,
                mesh_device=mesh,
                **(
                    {
                        "expert_parallel": args.expert_parallel,
                        "hybrid_experts": args.hybrid_experts,
                        "attention_precision": args.attention_precision,
                        "fused_tail": args.fused_tail,
                        "optimized_shared": args.optimized_shared,
                        "shared_geometry": args.shared_geometry,
                        "grouped_moe_reduce": args.grouped_moe_reduce,
                        "qkv_fidelity": getattr(ttnn.MathFidelity, args.qkv_fidelity),
                        "output_fidelity": getattr(ttnn.MathFidelity, args.output_fidelity),
                        "attention_ccl_dtype": getattr(ttnn, args.attention_ccl_dtype),
                        "full_attention_ccl_dtype": (
                            getattr(ttnn, args.full_attention_ccl_dtype) if args.full_attention_ccl_dtype else None
                        ),
                    }
                    if tp == 4
                    else {}
                ),
            )
            if (
                tp == 4
                and args.sliding_sharded_rope is not None
                and config.layer_types[args.layer] == "sliding_attention"
            ):
                decoder.layer.self_attn.sharded_decode_rope = args.sliding_sharded_rope
            cfg = decoder.layer.self_attn.config
            cache = [
                upload(
                    torch.zeros(batch * extent // block, cfg.num_key_value_heads, block, cfg.head_dim), ttnn.bfloat8_b
                )
                for _ in range(2)
            ]
            page_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            ropes = tuple(upload(t.unsqueeze(0)) for t in (cos, sin))

            def snapshot():
                return [[ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(c)] for c in cache]

            before = snapshot()
            prefills = []
            preserved = True
            for slot, x in enumerate(inputs):
                # Split inside a page; continuation must preserve all prefix rows.
                outputs = []
                for start, end in ((0, 31), (31, lengths[slot])):
                    dx = upload(x[:, start:end].unsqueeze(0))
                    with device_only():
                        y = decoder.prefill_forward(
                            dx, rope_mats=ropes, page_table=page_table, kv_cache=cache, user_id=slot, start_pos=start
                        )
                    outputs.append(read(y))
                    after = snapshot()
                    for old_ranks, new_ranks in zip(before, after):
                        for old, new in zip(old_ranks, new_ranks):
                            other = table[torch.arange(batch) != slot].flatten().long()
                            assert torch.equal(old[other], new[other]), "Other request cache changed"

                            def logical(c):
                                return (
                                    c[table[slot].long()]
                                    .permute(1, 0, 2, 3)
                                    .reshape(cfg.num_key_value_heads, extent, cfg.head_dim)
                                )

                            assert torch.equal(logical(old)[:, :start], logical(new)[:, :start]), "Prefix cache changed"
                    before = after
                prefills.append(torch.cat(outputs, dim=2))
            token = upload(tokens.unsqueeze(0))
            pos = upload(torch.tensor([lengths], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            cache_pos = upload(torch.tensor(lengths, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            rope_decode = tuple(upload(t.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for t in (cos, sin))

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

            for _ in range(2):
                y = forward()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            y = forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            decoded = read(y)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(decoded, read(y))
            # Reassign logical requests to existing cache slots after capture.
            # Every mutable tensor keeps its address while its contents change.
            for value, destination, dtype, layout in (
                (table.flip(0), page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                (tokens.flip(1).unsqueeze(0), token, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                (torch.tensor([lengths[::-1]], dtype=torch.int32), pos, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                (torch.tensor(lengths[::-1], dtype=torch.int32), cache_pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            ):
                host = ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(host, destination)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(decoded.flip(2), read(y)), "Trace retained stale request/page ownership"
            ttnn.release_trace(mesh, trace)
            results[tp] = dict(prefill=prefills, decode=decoded)
            preservation[tp] = dict(
                prefix_unchanged=True,
                other_slots_unchanged=True,
                all_device_outputs_equal=True,
                repeat_trace_equal=True,
                refreshed_request_page_ownership=True,
            )
        finally:
            ttnn.close_mesh_device(mesh)
    prefills = [pcc(a, b) for a, b in zip(results[1]["prefill"], results[4]["prefill"])]
    decodes = [pcc(results[1]["decode"][:, :, i : i + 1], results[4]["decode"][:, :, i : i + 1]) for i in range(batch)]
    report = dict(
        attention_precision=args.attention_precision,
        layer_type=config.layer_types[args.layer],
        expert_parallel=args.expert_parallel,
        hybrid_experts=args.hybrid_experts,
        fused_tail=args.fused_tail,
        optimized_shared=args.optimized_shared,
        shared_geometry=args.shared_geometry,
        grouped_moe_reduce=args.grouped_moe_reduce,
        sliding_sharded_rope_override=args.sliding_sharded_rope,
        qkv_fidelity=args.qkv_fidelity,
        output_fidelity=args.output_fidelity,
        attention_ccl_dtype=args.attention_ccl_dtype,
        full_attention_ccl_dtype_override=args.full_attention_ccl_dtype,
        batch=batch,
        base_length=args.base_length,
        continuation_lengths=[length - 31 for length in lengths],
        lengths=lengths,
        prefill_pcc=prefills,
        decode_pcc=decodes,
        cache_preservation=preservation,
        runtime_audit="clean",
        passed=all(math.isfinite(value) and value >= 0.995 for value in prefills + decodes),
        runtime_sha256=hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
