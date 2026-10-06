# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent TP4 precision controls around the unchanged stack harness.

Example (device execution must be serialized by the invoking agent):
    python -m models.autoports.google_gemma_4_26b_a4b_it.doc.optimized_multichip_decoder.probe_stack_precision \
        --layers 0 1 --length 4096 --steps 128 --probe-expert-gate-bf8 --output /tmp/expert_gate8.json

Reload weight controls from the original state, preserving current TP packing.
Do not widen already quantized weights. Acceptance remains the harness's 0.995.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import test_multichip_stack as stack


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-layers", type=int, nargs="+", default=None)
    parser.add_argument("--probe-expert-gate-bf8", action="store_true")
    parser.add_argument("--probe-expert-activation-bf16", action="store_true")
    parser.add_argument("--probe-shared-gate-bf8", action="store_true")
    parser.add_argument("--probe-qkv-hifi2", action="store_true")
    parser.add_argument("--probe-output-hifi4", action="store_true")
    parser.add_argument("--probe-attention-precision", choices=("baseline", "qkv", "output", "both"))
    parser.add_argument("--probe-moe-bf16", action="store_true")
    args, stack_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *stack_args, "--output", str(args.output)]
    original = stack.MultichipDecoder.from_state_dict
    policies = []

    def factory(cls, state, **kwargs):
        selected = args.probe_layers is None or kwargs["layer_idx"] in args.probe_layers
        if selected and args.probe_attention_precision is not None:
            kwargs["attention_precision"] = args.probe_attention_precision
        if selected and args.probe_moe_bf16:
            kwargs["moe_ccl_bfp8"] = False
        if selected and args.probe_qkv_hifi2:
            kwargs["qkv_fidelity"] = ttnn.MathFidelity.HiFi2
        if selected and args.probe_output_hifi4:
            kwargs["output_fidelity"] = ttnn.MathFidelity.HiFi4
        decoder = original(state, **kwargs)
        mesh = kwargs["mesh_device"]
        experts = getattr(decoder.layer.moe.experts, "decode", decoder.layer.moe.experts)
        shared = decoder.layer.shared_mlp
        if selected and args.probe_expert_gate_bf8:
            if decoder.expert_parallel or experts.indexed_router is None or experts.expert_split:
                raise ValueError("Expert gate control requires packed indexed TP decode experts")
            gate, up = state["experts.gate_up_proj"].chunk(2, dim=-2)
            padding = 4 * experts.width - gate.shape[-2]
            if padding < 0:
                raise ValueError("Expert state is wider than the current TP packing")
            gate, up = (torch.nn.functional.pad(t.transpose(-2, -1), (0, padding)) for t in (gate, up))
            packed = torch.cat(
                [torch.cat((g, u), dim=-1) for g, u in zip(gate.chunk(4, dim=-1), up.chunk(4, dim=-1))],
                dim=-1,
            )
            weight = ttnn.from_torch(
                packed.unsqueeze(0),
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            if tuple(weight.shape) != tuple(experts.gate_up.shape):
                raise ValueError("Reloaded expert gate/up shape differs from the selected implementation")
            experts.gate_up = weight
        if selected and args.probe_expert_activation_bf16:
            experts.decode_activation_dtype = None
        if selected and args.probe_shared_gate_bf8:
            if shared.decode_weights is None:
                raise ValueError("Shared gate control requires optimized interleaved shared decode")
            gate, up = (state[name].transpose(-2, -1) for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight"))
            padding = 4 * shared.width - gate.shape[-1]
            if padding < 0:
                raise ValueError("Shared state is wider than the current TP packing")
            gate, up = (torch.nn.functional.pad(t, (0, padding)) for t in (gate, up))
            packed = torch.cat(
                [torch.cat((u, g), dim=-1) for u, g in zip(up.chunk(4, dim=-1), gate.chunk(4, dim=-1))],
                dim=-1,
            )
            weight = ttnn.from_torch(
                packed[None, None],
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            if tuple(weight.shape) != tuple(shared.decode_weights[0].shape):
                raise ValueError("Reloaded shared gate/up shape differs from the selected implementation")
            shared.decode_weights = (weight, shared.decode_weights[1])
        policy = dict(
            layer=decoder.layer_idx,
            controls_applied=selected,
            expert_gate_dtype=str(experts.gate_up.dtype),
            expert_down_dtype=str(experts.down.dtype),
            expert_activation_dtype=str(experts.decode_activation_dtype),
            expert_compute=str(experts.decode_compute),
            shared_gate_dtype=str(shared.decode_weights[0].dtype) if shared.decode_weights else None,
            shared_down_dtype=str(shared.decode_weights[1].dtype) if shared.decode_weights else None,
            attention_precision=decoder.attention_precision,
            attention_ccl_dtype=str(decoder.attention_ccl_dtype),
            moe_ccl_bfp8=decoder.moe_ccl_bfp8,
            qkv_fidelity=str(kwargs.get("qkv_fidelity")),
            output_fidelity=str(kwargs.get("output_fidelity")),
            cache_dtype=str(decoder.kv_cache_dtype),
        )
        policies.append(policy)
        print("PRECISION_CONTROL", json.dumps(policy), flush=True)
        return decoder

    stack.MultichipDecoder.from_state_dict = classmethod(factory)
    old_mtime = args.output.stat().st_mtime_ns if args.output.exists() else None
    try:
        stack.main()
    finally:
        if args.output.exists() and args.output.stat().st_mtime_ns != old_mtime:
            report = json.loads(args.output.read_text())
            report["precision_probe"] = dict(
                options={name: value for name, value in vars(args).items() if name != "output"},
                actual_tp4_policy=policies,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                original_stack_gate=0.995,
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
