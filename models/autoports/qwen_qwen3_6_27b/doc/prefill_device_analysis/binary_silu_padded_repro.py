# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused finite-range SiLU input-activation control; no model weights needed."""

import json
from pathlib import Path

import torch

import ttnn


def metrics(ref, actual):
    a, b = ref.float(), actual.float()
    finite = torch.isfinite(b)
    return {
        "nonfinite": int((~finite).sum()),
        "max_abs": float((a - b).abs().max()) if finite.all() else None,
        "relative_l2": float((a - b).norm() / a.norm()) if finite.all() else None,
        "range": [float(b[finite].min()), float(b[finite].max())],
    }


def main():
    torch.manual_seed(42)
    shape = (1, 12, 128, 128)
    gates = torch.linspace(-20, 20, 128).reshape(1, 1, 1, 128).expand(shape).contiguous().bfloat16()
    values = (torch.randn(shape) * 2).bfloat16()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    result = {
        "shape": shape,
        "input_ranges": {"gate": [-20, 20], "value": [float(values.min()), float(values.max())]},
        "cases": {},
    }
    try:

        def upload(x):
            return ttnn.from_torch(
                x,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def host(t):
            return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

        a = upload(values)
        b = ttnn.permute(upload(gates.permute(0, 2, 1, 3).contiguous()), (0, 2, 1, 3))
        result["padded_shapes"] = {"a": list(a.padded_shape), "b": list(b.padded_shape)}
        baseline = host(ttnn.multiply(a, ttnn.silu(b)))
        result["baseline"] = metrics(values.float() * torch.nn.functional.silu(gates.float()), baseline)
        calls = {
            "rhs_silu": lambda: ttnn.multiply(a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SILU]),
            "lhs_silu_commuted": lambda: ttnn.multiply(b, a, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            "rhs_silu_fp32_output": lambda: ttnn.multiply(
                a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.float32
            ),
        }
        for name, call in calls.items():
            out = host(call())
            result["cases"][name] = metrics(baseline, out)
            if not torch.isfinite(out).all():
                bad = torch.nonzero(~torch.isfinite(out))
                result["cases"][name]["bad_sample"] = [
                    {"index": idx.tolist(), "gate": float(gates[tuple(idx)]), "value": float(values[tuple(idx)])}
                    for idx in bad[:8]
                ]
            print(name, result["cases"][name], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    path = Path(__file__).parent / "artifacts/binary_silu_padded_repro.json"
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
