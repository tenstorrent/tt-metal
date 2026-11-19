# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate GR arithmetic with exact HF inputs at every operation boundary."""

import argparse
import json
from pathlib import Path

import torch

from .probe_batch_intermediates import fixture, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    _, _, _, requests, _ = fixture(args.snapshot, (3, 16, 23))
    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"scope": __doc__, "cases": []}
    try:

        def upload(value):
            return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)

        for request, item in requests.items():
            for phase, values in item["phases"].items():
                for kind in ("attn", "mlp"):
                    parts = values[kind + "_read_parts"]
                    device = {key: upload(value.unsqueeze(0)) for key, value in parts.items()}
                    for precise in (False, True):

                        def operation(fn, *operands, **kwargs):
                            converted = [
                                (
                                    ttnn.typecast(value, ttnn.float32)
                                    if precise and isinstance(value, ttnn.Tensor)
                                    else value
                                )
                                for value in operands
                            ]
                            result = fn(*converted, **kwargs)
                            return ttnn.typecast(result, ttnn.bfloat16) if precise else result

                        products = operation(ttnn.multiply, device["mixing"], device["normalized"])
                        count = parts["normalized"].shape[-2]
                        exact_products = ttnn.permute(
                            ttnn.reshape(device["product"], (1, count, 4, 2560)), (0, 2, 1, 3)
                        )
                        outputs = {
                            "scaled": operation(ttnn.multiply, device["down"], 0.25),
                            "activated": operation(ttnn.silu, device["scaled"]),
                            "mixing": operation(ttnn.sigmoid, device["up"]),
                            "product": products,
                            "mixed": operation(ttnn.mean, exact_products, dim=1, keepdim=True),
                            "injection_scaled": operation(ttnn.multiply, device["injection_logits"], 0.25),
                            "injection_sigmoid": operation(ttnn.sigmoid, device["injection_scaled"]),
                            "injection": operation(ttnn.multiply, device["injection_sigmoid"], 2),
                        }
                        result = {}
                        for name, tensor in outputs.items():
                            measured = metrics(parts[name], ttnn.to_torch(tensor).reshape(parts[name].shape))
                            result[name] = {
                                key: value for key, value in measured.items() if not key.startswith("per_row")
                            }
                        report["cases"].append(
                            dict(request=request, phase=phase, kind=kind, fp32_then_bf16=precise, operations=result)
                        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
