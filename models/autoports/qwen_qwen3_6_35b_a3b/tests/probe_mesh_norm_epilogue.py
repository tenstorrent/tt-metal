# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in hardware component probe; the stage owner serializes its invocation.

Run with the normal lane environment and a bounded timeout. This checks the
experimental scalar inverse against the exact existing TTNN operation chain;
it does not replace layer0/layer34 PCC, trace, state or watcher validation.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_norm_epilogue_probe import CompensatedNormEpilogue


def native_inverse(total):
    """Literal scalar operation sequence from FusedDecoder.norm."""
    t = ttnn
    shifted = t.add(
        total,
        1e-6,
        input_tensor_a_activations=[t.UnaryWithParam(t.UnaryOpType.MUL_UNARY_SFPU, 1 / 2048)],
    )

    def split(value):
        high = t.unary_chain(
            value,
            [
                t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
                t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
            ],
        )
        return high, t.subtract(value, high)

    root = t.sqrt(shifted, fast_and_approximate_mode=False)
    high, low = split(root)
    residual = t.subtract(
        t.subtract(
            t.subtract(shifted, high, input_tensor_b_activations=[t.UnaryOpType.SQUARE]),
            t.multiply(
                high,
                low,
                input_tensor_a_activations=[t.UnaryWithParam(t.UnaryOpType.MUL_UNARY_SFPU, 2.0)],
            ),
        ),
        low,
        input_tensor_b_activations=[t.UnaryOpType.SQUARE],
    )
    root = t.add(
        root,
        t.multiply(
            residual,
            root,
            input_tensor_b_activations=[t.UnaryOpType.RECIP, t.UnaryWithParam(t.UnaryOpType.MUL_UNARY_SFPU, 0.5)],
        ),
    )
    inverse = t.reciprocal(root)
    rh, rl = split(root)
    ih, il = split(inverse)
    residual = t.subtract(
        t.subtract(
            t.subtract(
                t.multiply(rh, ih, activations=[t.UnaryWithParam(t.UnaryOpType.RSUB, 1.0)]),
                t.multiply(rh, il),
            ),
            t.multiply(rl, ih),
        ),
        t.multiply(rl, il),
    )
    return t.add(inverse, t.multiply(inverse, residual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("--rows must be at least2 to include zero and finite positive sums")
    # Include zero (epsilon-only) and broad finite scales. Model activations
    # still require the separate real-weight layer checks.
    host = torch.cat([torch.zeros(1), torch.logspace(-8, 8, args.rows - 1)]).reshape(args.rows, 1, 1, 1)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        total = ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        reference = native_inverse(total)
        candidate = CompensatedNormEpilogue(ttnn, mesh, 1e-6)(total)
        ttnn.synchronize_device(mesh)
        ranks = []
        for rank, (ref, got) in enumerate(zip(ttnn.get_device_tensors(reference), ttnn.get_device_tensors(candidate))):
            a, b = ttnn.to_torch(ref).float(), ttnn.to_torch(got).float()
            ranks.append(
                {
                    "rank": rank,
                    "bit_equal": torch.equal(a, b),
                    "finite": bool(torch.isfinite(b).all()),
                    "mismatched_elements": int((a != b).sum()),
                    "max_abs_error": float((a - b).abs().max()),
                }
            )
        result = {"rows": args.rows, "mesh": [1, 4], "ranks": ranks, "passed": all(r["bit_equal"] for r in ranks)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        assert result["passed"], "fused scalar epilogue differs from the native compensated chain"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
