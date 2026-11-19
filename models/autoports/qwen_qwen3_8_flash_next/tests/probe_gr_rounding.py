# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-operand GR projection rounding analysis and compact fixture.

Compares literal HF BF16 F.linear with FP32 F.linear, FP64 dot products and
the implemented balanced FP32 product/addition tree. No reference is changed.
The FP64 comparison diagnoses BF16 midpoint sensitivity; it is not a proposed
acceptance oracle. This command never imports TTNN.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .probe_batch_intermediates import fixture


def tree_projection(value, weight):
    blocks = []
    for start in range(0, weight.shape[0], 32):
        product = value.float().unsqueeze(-2) * weight[start : start + 32].float()
        width, tails = product.shape[-1], []
        while width > 1:
            half = width // 2
            if width % 2:
                tails.append(product[..., -1:])
            product = product[..., :half] + product[..., half : 2 * half]
            width = half
        for tail in tails:
            product = product + tail
        blocks.append(product.squeeze(-1))
    return torch.cat(blocks, -1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    _, _, weights, requests, hashes = fixture(args.snapshot, (3, 16, 23, 29))
    report, saved = {"scope": __doc__, "cases": []}, {"cases": [], "weights": {}, "rng_hashes": hashes}
    for request, data in requests.items():
        for phase, values in data["phases"].items():
            for kind in ("attn", "mlp"):
                parts = values[kind + "_read_parts"]
                for name, operand_name in (("down", "normalized"), ("up", "activated")):
                    key = kind + "_hyper_connection.input_mix_weight_" + name + ".weight"
                    weight = weights[key]
                    operand, target = parts[operand_name], parts[name]
                    exact = F.linear(operand.double(), weight.double())
                    candidates = {
                        "torch_fp32": F.linear(operand.float(), weight.float()),
                        "fp32_tree": tree_projection(operand, weight),
                        "fp64": exact,
                    }
                    case = {"request": request, "phase": phase, "kind": kind, "projection": name, "candidates": {}}
                    saved["weights"][key] = weight.clone()
                    saved["cases"].append(
                        {
                            **{k: v for k, v in case.items() if k != "candidates"},
                            "weight_key": key,
                            "operand": operand.clone(),
                            "expected": target.clone(),
                        }
                    )
                    for policy, raw in candidates.items():
                        rounded = raw.bfloat16()
                        differing = torch.nonzero(rounded != target)
                        details = []
                        for index in differing[:12]:
                            idx = tuple(index.tolist())
                            details.append(
                                {
                                    "index": list(idx),
                                    "hf": target[idx].item(),
                                    "candidate": rounded[idx].item(),
                                    "raw": raw[idx].item(),
                                    "fp64": exact[idx].item(),
                                    "midpoint": (target[idx].double().item() + rounded[idx].double().item()) / 2,
                                }
                            )
                        case["candidates"][policy] = {
                            "different_values": len(differing),
                            "total_values": target.numel(),
                            "maximum_raw_error_vs_fp64": (raw.double() - exact).abs().max().item(),
                            "examples": details,
                        }
                    report["cases"].append(case)
    assert "ttnn" not in sys.modules
    torch.save(saved, args.fixture_output)
    report["fixture"] = str(args.fixture_output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
