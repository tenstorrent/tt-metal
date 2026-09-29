# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare the TT final DiT normalization/projection with CUDA captures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = args.cuda_dir / "step_000/transformer"

    def read(name: str) -> torch.Tensor:
        return torch.load(base / f"{name}.pt", map_location="cpu", weights_only=True)

    hidden = read("transformer_blocks.31")
    temb = read("time_text_embed")
    target_mask = torch.load(
        base / "transformer_blocks.0/input/target_token_mask.pt", map_location="cpu", weights_only=True
    )
    expected_norm, expected_output = read("norm_out"), read("proj_out")
    transformer = args.checkpoint / "transformer"
    mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def weight(name: str) -> torch.Tensor:
        with safe_open(transformer / mapping[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    norm_weight = weight("norm_out.linear.weight")
    projection_weight = weight("proj_out.weight")

    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_dit_components import to_device, to_host
    from models.experimental.qwen_image_2_1.tt.tt_output import output_head, prepare_weights, select_timestep_embedding

    device = None
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        weights = prepare_weights(norm_weight, projection_weight, device)
        selected_temb = select_timestep_embedding(temb, target_mask, device)
        normalized, output = output_head(to_device(hidden, device), selected_temb, weights)
        reports = []
        for name, result, expected in (
            ("norm_out", normalized, expected_norm),
            ("proj_out", output, expected_output),
        ):
            actual = to_host(result, tuple(expected.shape))
            destination = args.output_dir / f"step_000/transformer/{name}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actual, destination)
            difference = (actual.float() - expected.float()).abs()
            row = {
                "component": name,
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "relative_rms_error": float(
                    difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
                ),
            }
            reports.append(row)
            print(json.dumps(row), flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
