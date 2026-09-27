# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay indexed TP experts on frozen real inputs from a boundary failure.

Default reupload preserves logical values and zeroes padding; --physical-input
restores saved physical BF16 tiles. An optional native gate prefix isolates one
preceding operation while expert routes and IDs remain frozen. Use --output-only
to run the original expert implementation without retention.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.diagnose_attention_ccl_boundaries import (
    BoundaryExperts,
    compare,
    snapshot,
)
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedExperts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--output-only", action="store_true")
    parser.add_argument(
        "--physical-input", action="store_true", help="Restore saved BF16 physical tile bits and logical views."
    )
    parser.add_argument(
        "--native-gate-core",
        type=int,
        choices=[0, 1],
        help="Prepend the native gate on this core; keep expert routes frozen.",
    )
    parser.add_argument("--read-output-only", action="store_true")
    parser.add_argument("--read-padding", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    root = Path(__file__).parents[1]
    fixture = torch.load(args.fixture, weights_only=True)
    recorded = fixture["reference"]
    for name in ("expert_input", "routes", "router_gate_indices", "routed_local"):
        assert len(recorded[name]) == 4, (name, len(recorded[name]))
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(config, args.layer, True)
    result = dict(
        command=sys.argv,
        runtime_sha256=hashlib.sha256((root / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        experts_sha256=hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        passed=False,
        layer=args.layer,
        output_only=args.output_only,
        physical_input=args.physical_input,
        native_gate_core=args.native_gate_core,
        read_output_only=args.read_output_only,
        read_padding=args.read_padding,
        repeats=args.repeats,
        completed_repeats=0,
        limitation=(
            "Host reupload preserves saved physical BF16 values; whole-layer allocator/program state differs."
            if args.physical_input
            else "Host reupload preserves logical values; producer padding and whole-layer allocator state differ."
        ),
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16777216)
    trace_id = None
    try:
        decoder = MultichipDecoder.from_state_dict(
            hf.state_dict(),
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            topology=ttnn.Topology.Linear,
            fused_tail=True,
            hybrid_experts=True,
            optimized_shared=True,
            shared_geometry=1,
            grouped_moe_reduce=True,
            attention_ccl_dtype=ttnn.bfloat8_b,
        )
        experts = decoder.layer.moe.experts.decode
        assert type(experts) is OptimizedExperts
        if not args.output_only:
            experts.__class__ = BoundaryExperts
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def upload(name, dtype, layout=ttnn.TILE_LAYOUT):
            physical = args.physical_input and layout == ttnn.TILE_LAYOUT
            key = name + ":physical" if physical else name
            host = torch.cat(recorded[key], dim=0)
            if physical:
                assert dtype == ttnn.bfloat16
                # The snapshots are BF16 expanded to FP32. Reconstruct the
                # upper 16 bits explicitly, preserving NaN payloads as well.
                host = (host.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
            value = ttnn.from_torch(
                host,
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=mapper,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if physical:
                value = ttnn.reshape(value, ttnn.Shape(recorded[name][0].shape), ttnn.Shape(recorded[key][0].shape))
            return value

        gate_prefixes = []
        if args.native_gate_core is not None:
            router = decoder.layer.moe.router
            rounded = torch.cat(recorded["router_rounded_scores"], dim=0).reshape(4, 128)
            face_host = torch.cat((rounded, torch.full_like(rounded, float("-inf"))), dim=-1).reshape(4, 16, 16)
            # Allocate both alternatives in the same order in every A/B run.
            # Only the selected kernel's executing core changes during replay.
            for core_x in (0, 1):
                core = ttnn.CoreCoord(core_x, 0)
                grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
                shard = ttnn.ShardSpec(grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
                memory = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
                buffers = {
                    field: ttnn.to_memory_config(getattr(router, field), memory)
                    for field in ("bias", "indices", "output", "output_indices")
                }
                buffers["face"] = ttnn.from_torch(
                    face_host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=mapper,
                    memory_config=memory,
                )
                gate_prefixes.append(buffers)
            result["native_gate_addresses"] = [
                {name: int(tensor.buffer_address()) for name, tensor in buffers.items()} for buffers in gate_prefixes
            ]

        x = upload("expert_input", ttnn.bfloat16)
        routing = upload("routes", ttnn.bfloat16)
        indices = upload("router_gate_indices", ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
        experts.indexed_router.last_decode_indices = indices
        if args.physical_input:
            uploaded = snapshot({"expert_input": x, "routes": routing}, include_padding=True)
            result["physical_input_comparison"] = compare({name: recorded[name] for name in uploaded}, uploaded)
            assert all(item["equal"] for item in result["physical_input_comparison"].values())
        result["configuration"] = dict(
            local_width=experts.width,
            gate_program=str(experts.gate_config),
            down_program=str(experts.down_config),
            mix_program=str(experts.mix_program),
            decode_compute=str(experts.decode_compute),
            mix_compute=str(experts.mix_compute),
            gate_weight_dtype=str(experts.gate_up.dtype),
            gate_weight_shape=list(experts.gate_up.shape),
            down_weight_dtype=str(experts.down.dtype),
            down_weight_shape=list(experts.down.shape),
        )

        def forward():
            with device_only():
                if args.native_gate_core is not None:
                    prefix = gate_prefixes[args.native_gate_core]
                    ttnn.experimental.deepseek.moe.generalized_moe_gate(
                        prefix["face"],
                        bias_tensor=prefix["bias"],
                        input_indices_tensor=prefix["indices"],
                        output_tensor=prefix["output"],
                        output_indices_tensor=prefix["output_indices"],
                        eps=1e-20,
                        scaling_factor=1.0,
                        enable_sigmoid=False,
                        topk=8,
                        output_softmax=True,
                        grouped=False,
                    )
                return experts(x, routing)

        for _ in range(2):
            y = forward()
        ttnn.synchronize_device(mesh)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        y = forward()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        retained = {"output": y} if args.output_only else experts.boundaries
        tensors = {"output": y} if args.read_output_only else retained
        result["boundaries"] = {
            name: dict(shape=list(tensor.shape), padded_shape=list(tensor.padded_shape), dtype=str(tensor.dtype))
            for name, tensor in retained.items()
        }
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        reference = snapshot(tensors, args.read_padding)
        result["recorded_output_comparison"] = compare(
            {"output": recorded["routed_local"]}, {"output": reference["output"]}
        )
        if "actual" in fixture and "routed_local" in fixture["actual"]:
            result["recorded_actual_output_comparison"] = compare(
                {"output": fixture["actual"]["routed_local"]}, {"output": reference["output"]}
            )
        if args.native_gate_core is not None:
            reference_path = args.output.with_suffix(".reference.pt")
            torch.save(reference, reference_path)
            result["reference_tensors"] = str(reference_path)
        for repeat in range(args.repeats):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            actual = snapshot(tensors, args.read_padding)
            comparisons = compare(reference, actual)
            first_difference = next((name for name, item in comparisons.items() if not item["equal"]), None)
            nonfinite = any(
                item["reference_nonfinite"] or item["actual_nonfinite"] for item in comparisons["output"]["ranks"]
            )
            result["completed_repeats"] = repeat + 1
            if first_difference is not None or nonfinite:
                result["failure"] = dict(
                    repeat=repeat,
                    first_different_boundary=first_difference,
                    output_nonfinite=nonfinite,
                    comparisons=comparisons,
                )
                tensor_path = args.output.with_suffix(".pt")
                torch.save(dict(reference=reference, actual=actual), tensor_path)
                result["failure_tensors"] = str(tensor_path)
                break
        else:
            result["passed"] = True
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert result["passed"], result.get("failure")
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
