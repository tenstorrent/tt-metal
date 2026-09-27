# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure prefill producer placement without changing decode or residual contracts."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_multichip_decoder as harness
from models.autoports.google_gemma_4_26b_a4b_it.tt import optimized_decoder


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--prefill-l1-role", nargs="+", choices=["control", "qkv", "output", "router", "shared"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args, rest = parser.parse_known_args()
    command = list(sys.argv)
    sys.argv = [sys.argv[0], *rest, "--output", str(args.output)]
    original_factory = harness.MultichipDecoder.from_state_dict
    original_mul = ttnn.mul
    original_concat = optimized_decoder.concat_heads
    targets = []
    meshes = []
    hits = {}
    memories = {}

    def placed_mul(*pos, **kw):
        a = pos[0] if pos else kw.get("input_tensor_a")
        b = pos[1] if len(pos) > 1 else kw.get("input_tensor_b")
        if hasattr(a, "shape") and a.shape[-2] > 1:
            for role, weight in targets:
                if b is weight:
                    kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
                    hits[role] = hits.get(role, 0) + 1
        return original_mul(*pos, **kw)

    def placed_concat(*pos, **kw):
        if (
            "output" in args.prefill_l1_role
            and not kw.get("is_decode_mode", False)
            and any(kw.get("mesh_device") is mesh for mesh in meshes)
        ):
            kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
            hits["output"] = hits.get("output", 0) + 1
        return original_concat(*pos, **kw)

    def factory(cls, state, **kw):
        decoder = original_factory(state, **kw)
        meshes.append(decoder.mesh_device)
        if "qkv" in args.prefill_l1_role:
            decoder.prefill_qkv_input_l1 = True
        if "router" in args.prefill_l1_role:
            targets.append(("router", decoder.layer.moe.router.original.source.scale))
        if "shared" in args.prefill_l1_role:
            targets.append(("shared", decoder.shared_norm_weight))
        projection = decoder.layer.self_attn.source.weights.wqkv
        original_prefill = projection.prefill

        def measured_prefill(x, **kwargs):
            memory = str(x.memory_config())
            memories[memory] = memories.get(memory, 0) + 1
            return original_prefill(x, **kwargs)

        projection.prefill = measured_prefill
        return decoder

    harness.MultichipDecoder.from_state_dict = classmethod(factory)
    ttnn.mul = placed_mul
    optimized_decoder.concat_heads = placed_concat
    previous = args.output.stat().st_mtime_ns if args.output.exists() else None
    try:
        harness.main()
    finally:
        if args.output.exists() and args.output.stat().st_mtime_ns != previous:
            report = json.loads(args.output.read_text())
            report["prefill_producer_l1_probe"] = dict(
                roles=args.prefill_l1_role,
                command=command,
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                producer_hits=hits,
                qkv_input_memories=memories,
                scope="TP4 prefill only; producers directly emit L1 without an added restore/conversion",
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
