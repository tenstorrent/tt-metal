# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare direct two-layer TP1/TP4 handoff and advancing traced decode."""

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
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import CollectiveBufferPool, MultichipDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def pcc(left, right):
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    return torch.corrcoef(torch.stack((left.flatten().double(), right.flatten().double())))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--sharded-residual", action="store_true")
    parser.add_argument("--pool-ccl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layers", type=int, nargs=2, default=[4, 5])
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expert_parallel and args.hybrid_experts:
        parser.error("--expert-parallel requires --no-hybrid-experts")
    if args.shared_geometry and not args.optimized_shared:
        parser.error("--no-optimized-shared requires --shared-geometry 0")
    if args.sharded_residual and args.grouped_moe_reduce:
        parser.error("--sharded-residual requires --no-grouped-moe-reduce")
    torch.set_num_threads(8)
    torch.manual_seed(47)
    root = Path(__file__).parents[1]
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("tt/multichip_decoder.py", "tt/optimized_decoder.py", "tests/test_multichip_stack.py")
    }
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    layers = tuple(args.layers)
    if not all(0 <= layer < len(config.layer_types) for layer in layers) or not 1 <= args.steps <= 128:
        parser.error("Valid layer indices and1..128 recorded decode steps are required")
    if not 1 <= args.length <= 4096:
        parser.error("Length must fit the recorded 4096-token fixture")
    length, steps, block = args.length, args.steps, 32
    extent = (length + steps + 127) // 128 * 128
    hf_layers = {layer: load_layer(config, layer, True) for layer in layers}
    fixture_stage = "optimized_multichip_decoder" if layers[0] == 4 else "optimized_decoder"
    fixture_length = length if layers[0] > 0 else 4096
    fixture_path = args.fixture or root / f"doc/{fixture_stage}/actual_text_layer{layers[0]}_{fixture_length}_128.pt"
    fixture = torch.load(fixture_path, weights_only=True)
    if fixture["metadata"]["layer"] != layers[0]:
        parser.error("Recorded activation boundary must match the first tested layer")
    if layers[0] > 0 and fixture["metadata"]["length"] != length:
        parser.error("Contextual HF activations must use their recorded prefill length")
    if fixture["decode"].shape[1] < steps:
        parser.error("Fixture has fewer decode activations than requested")
    fixture_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    prefill_input = fixture["prefill"][:, :length].unsqueeze(0)
    decode_inputs = fixture["decode"][:, :steps].unsqueeze(0)
    tables = {layer: torch.randperm(extent // block, dtype=torch.int32)[None] for layer in layers}
    ropes = {
        layer: Gemma4TextRotaryEmbedding(config)(
            fixture["prefill"][:, :length], torch.arange(extent)[None], layer_type=config.layer_types[layer]
        )
        for layer in layers
    }
    results = {}
    for tp, decoder_class in ((1, OptimizedDecoder), (4, MultichipDecoder)):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED if tp == 1 else ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, tp), trace_region_size=67108864)
        try:
            replicated = ttnn.ReplicateTensorToMesh(mesh)
            sharded = tp == 4 and args.sharded_residual
            input_mapper = ttnn.ShardTensorToMesh(mesh, dim=-1) if sharded else replicated

            def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper=replicated):
                return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper)

            def read(value, label):
                if sharded:
                    parts = [ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()]
                else:
                    parts = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(value)]
                nonfinite = [int((~torch.isfinite(part)).sum()) for part in parts]
                changed = [int((parts[0] != part).sum()) for part in parts]
                if any(nonfinite) or any(changed):
                    failure = dict(
                        label=label,
                        tp=tp,
                        nonfinite_per_rank=nonfinite,
                        changed_from_rank0=changed,
                        sharded_residual=sharded,
                        source_sha256=source_hashes,
                        passed=False,
                    )
                    args.output.with_suffix(".failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                    raise AssertionError(f"Stack output finite/replica check failed: {failure}")
                return parts[0]

            contexts = []
            pool = CollectiveBufferPool(mesh) if tp == 4 and args.pool_ccl else None
            for layer in layers:
                decoder = decoder_class.from_state_dict(
                    hf_layers[layer].state_dict(),
                    hf_config=config,
                    layer_idx=layer,
                    mesh_device=mesh,
                    **(
                        {
                            "expert_parallel": args.expert_parallel,
                            "attention_precision": args.attention_precision,
                            "collective_buffer_pool": pool,
                            "sharded_residual": args.sharded_residual,
                            "hybrid_experts": args.hybrid_experts,
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
                    and config.layer_types[layer] == "sliding_attention"
                ):
                    decoder.layer.self_attn.sharded_decode_rope = args.sliding_sharded_rope
                attention = decoder.layer.self_attn.config
                contexts.append(
                    dict(
                        decoder=decoder,
                        cache=[
                            upload(
                                torch.zeros(extent // block, attention.num_key_value_heads, block, attention.head_dim),
                                ttnn.bfloat8_b,
                            )
                            for _ in range(2)
                        ],
                        page_table=upload(tables[layer], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                        prefill_rope=tuple(upload(value.unsqueeze(0)) for value in ropes[layer]),
                        decode_rope=tuple(
                            upload(value.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for value in ropes[layer]
                        ),
                        position=upload(
                            torch.tensor([[length]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                        ),
                        cache_position=upload(
                            torch.tensor([length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                        ),
                    )
                )
            hidden = upload(prefill_input, mapper=input_mapper)
            prefill_outputs = []
            with device_only():
                for context in contexts:
                    # Pass the exact device output into the next layer. No host
                    # conversion or interlayer memory/layout transformation.
                    hidden = context["decoder"].prefill_forward(
                        hidden,
                        rope_mats=context["prefill_rope"],
                        page_table=context["page_table"],
                        kv_cache=context["cache"],
                    )
                    prefill_outputs.append(hidden)
            prefills = [read(value, f"prefill/layer{layer}") for layer, value in zip(layers, prefill_outputs)]
            assert all(tuple(value.shape) == (1, 1, length, config.hidden_size) for value in prefills)
            token = upload(decode_inputs[:, :, :1], mapper=input_mapper)

            def forward():
                hidden = token
                outputs = []
                with device_only():
                    for context in contexts:
                        hidden = context["decoder"].decode_forward(
                            hidden,
                            rope_mats=context["decode_rope"],
                            current_pos=context["position"],
                            cache_pos=context["cache_position"],
                            page_table=context["page_table"],
                            kv_cache=context["cache"],
                        )
                        outputs.append(hidden)
                return outputs

            for _ in range(2):
                forward()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            decode_outputs = forward()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            decoded = []
            try:
                for step in range(steps):
                    host_token = ttnn.from_torch(
                        decode_inputs[:, :, step : step + 1],
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=input_mapper,
                    )
                    ttnn.copy_host_to_device_tensor(host_token, token)
                    for context in contexts:
                        for value, destination, dtype in (
                            (torch.tensor([[length + step]], dtype=torch.int32), context["position"], ttnn.uint32),
                            (torch.tensor([length + step], dtype=torch.int32), context["cache_position"], ttnn.int32),
                        ):
                            host = ttnn.from_torch(
                                value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicated
                            )
                            ttnn.copy_host_to_device_tensor(host, destination)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    values = [
                        read(value, f"decode/step{step}/layer{layer}/first")
                        for layer, value in zip(layers, decode_outputs)
                    ]
                    assert all(tuple(value.shape) == (1, 1, 1, config.hidden_size) for value in values)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    repeated = [
                        read(value, f"decode/step{step}/layer{layer}/repeat")
                        for layer, value in zip(layers, decode_outputs)
                    ]
                    assert all(
                        torch.equal(first, second) for first, second in zip(values, repeated)
                    ), f"Stack replay differs at TP{tp} step{step}; changed per layer: " + str(
                        {layer: int((first != second).sum()) for layer, first, second in zip(layers, values, repeated)}
                    )
                    decoded.append(values)
            finally:
                ttnn.release_trace(mesh, trace)
            results[tp] = dict(prefill=prefills, decode=decoded)
            print("STACK_TP_DONE", tp, flush=True)
        finally:
            ttnn.close_mesh_device(mesh)

    comparisons = []
    for index, layer in enumerate(layers):
        comparisons.append(
            dict(phase="prefill", layer=layer, pcc=pcc(results[1]["prefill"][index], results[4]["prefill"][index]))
        )
        for step in range(steps):
            comparisons.append(
                dict(
                    phase="decode",
                    layer=layer,
                    position=length + step,
                    pcc=pcc(results[1]["decode"][step][index], results[4]["decode"][step][index]),
                )
            )
    output_hashes = {
        str(tp): {
            "prefill": [
                hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() for value in results[tp]["prefill"]
            ],
            "decode": [
                [hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() for value in step]
                for step in results[tp]["decode"]
            ],
        }
        for tp in results
    }
    report = dict(
        shared_collective_pool=args.pool_ccl,
        attention_precision=args.attention_precision,
        output_sha256=output_hashes,
        layers=list(layers),
        real_model_adjacency=layers[1] == layers[0] + 1,
        fixture_layer=fixture["metadata"]["layer"],
        fixture_sha256=fixture_hash,
        fixture_path=str(fixture_path),
        layer_types=[config.layer_types[layer] for layer in layers],
        length=length,
        decode_positions=list(range(length, length + steps)),
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
        sharded_residual=args.sharded_residual,
        direct_interlayer_handoff=True,
        independent_layer_caches=True,
        trace_scope="both layers in one trace",
        repeat_trace_equal=True,
        all_replicas_equal=not args.sharded_residual,
        runtime_audit="clean",
        source_sha256=source_hashes,
        comparisons=comparisons,
        passed=all(math.isfinite(value["pcc"]) and value["pcc"] >= 0.995 for value in comparisons),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
