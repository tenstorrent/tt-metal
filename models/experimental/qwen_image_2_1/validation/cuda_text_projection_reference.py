# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Derive CUDA targets inside Qwen Image 2.1's text projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = args.cuda_dir / "step_000/transformer"
    source = torch.load(base / "input/encoder_hidden_states.pt", map_location="cpu", weights_only=True).cuda()
    reference = torch.load(base / "txt_in.pt", map_location="cpu", weights_only=True).cuda()
    transformer = args.checkpoint / "transformer"
    mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def weight(name: str) -> torch.Tensor:
        with safe_open(transformer / mapping[name], framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(name).cuda()

    norm_weight = weight("txt_in.text_norm.weight")
    linear1_weight = weight("txt_in.in_layer.weight")
    linear2_weight = weight("txt_in.out_layer.weight")
    with torch.inference_mode():
        x = source.float()
        rrms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
        normalized = (x * rrms * (norm_weight.float() + 1)).to(source.dtype)
        linear1 = F.linear(normalized, linear1_weight)
        activated = F.gelu(linear1, approximate="tanh")
        linear2 = F.linear(activated, linear2_weight)
    difference = (linear2.float() - reference.float()).abs()
    report = {
        "mean_abs_error_to_pipeline": float(difference.mean()),
        "max_abs_error_to_pipeline": float(difference.max()),
        "exact_fraction_to_pipeline": float((linear2 == reference).float().mean()),
    }
    for name, tensor in (
        ("text_norm", normalized),
        ("linear1", linear1),
        ("gelu", activated),
        ("linear2", linear2),
    ):
        torch.save(tensor.cpu(), args.output_dir / f"{name}.pt")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not torch.allclose(linear2, reference, atol=0.01, rtol=0.001):
        raise AssertionError("derived text projection differs from the pipeline capture")


if __name__ == "__main__":
    main()
