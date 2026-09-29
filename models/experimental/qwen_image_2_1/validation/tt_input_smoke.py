# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Check Qwen Image 2.1 input and timestep-conditioning components on TT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


WEIGHT_NAMES = (
    "img_in.weight",
    "txt_in.text_norm.weight",
    "txt_in.in_layer.weight",
    "txt_in.out_layer.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "modulation.1.weight",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--step", type=int, choices=(0, 1), default=0)
    parser.add_argument("--derived-text-dir", type=Path, help="CUDA-derived text projection stages")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step_dir = f"step_{args.step:03d}"
    base = args.cuda_dir / step_dir / "transformer"

    def read(name: str) -> torch.Tensor:
        return torch.load(base / f"{name}.pt", map_location="cpu", weights_only=True)

    latents = read("input/hidden_states")
    context = read("input/encoder_hidden_states")
    timestep = read("input/timestep")
    expected = {name: read(name) for name in ("img_in", "txt_in", "time_text_embed", "modulation")}
    transformer = args.checkpoint / "transformer"
    mapping = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for shard in sorted({mapping[name] for name in WEIGHT_NAMES}):
        with safe_open(transformer / shard, framework="pt", device="cpu") as source:
            for name in WEIGHT_NAMES:
                if mapping[name] == shard:
                    state[name] = source.get_tensor(name)

    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_dit_components import to_device, to_host
    from models.experimental.qwen_image_2_1.tt.tt_input import (
        image_projection,
        prepare_weights,
        sinusoidal_timestep,
        text_projection_stages,
        timestep_and_modulation,
    )

    device = None
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        weights = prepare_weights(state, device)
        image = image_projection(to_device(latents, device), weights)
        compute = weights.text_norm_compute
        text_stages = text_projection_stages(to_device(context, device), weights)
        text = text_stages[-1]
        features = sinusoidal_timestep(torch.cat((timestep, timestep.new_zeros(1))))
        time_embedding, modulation = timestep_and_modulation(to_device(features, device), weights)
        reports = []
        for name, result in (
            ("img_in", image),
            ("txt_in", text),
            ("time_text_embed", time_embedding),
            ("modulation", modulation),
        ):
            reference = expected[name]
            actual = to_host(result, tuple(reference.shape))
            destination = args.output_dir / step_dir / "transformer" / f"{name}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(actual, destination)
            difference = (actual.float() - reference.float()).abs()
            row = {
                "component": name,
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "relative_rms_error": float(
                    difference.square().mean().sqrt() / reference.float().square().mean().sqrt()
                ),
            }
            reports.append(row)
            print(json.dumps(row), flush=True)
        if args.derived_text_dir is not None:
            for name, result in zip(("text_norm", "linear1", "gelu", "linear2"), text_stages):
                reference = torch.load(args.derived_text_dir / f"{name}.pt", map_location="cpu", weights_only=True)
                actual = to_host(result, tuple(reference.shape))
                destination = args.output_dir / "derived_text" / f"{name}.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(actual, destination)
                difference = (actual.float() - reference.float()).abs()
                row = {
                    "component": f"text_{name}",
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "relative_rms_error": float(
                        difference.square().mean().sqrt() / reference.float().square().mean().sqrt()
                    ),
                }
                reports.append(row)
                print(json.dumps(row), flush=True)
            exact_linear1 = torch.load(args.derived_text_dir / "linear1.pt", map_location="cpu", weights_only=True)
            exact_text_norm = torch.load(args.derived_text_dir / "text_norm.pt", map_location="cpu", weights_only=True)
            expected_gelu = torch.load(args.derived_text_dir / "gelu.pt", map_location="cpu", weights_only=True)
            expected_linear2 = torch.load(args.derived_text_dir / "linear2.pt", map_location="cpu", weights_only=True)
            exact_linear1_tt = to_device(exact_linear1, device)
            isolated = (
                ("gelu_fast_exact_input", ttnn.gelu(exact_linear1_tt, fast_and_approximate_mode=True), expected_gelu),
                (
                    "gelu_accurate_exact_input",
                    ttnn.gelu(exact_linear1_tt, fast_and_approximate_mode=False),
                    expected_gelu,
                ),
                (
                    "linear2_exact_input",
                    ttnn.matmul(to_device(expected_gelu, device), weights.text_out, compute_kernel_config=compute),
                    expected_linear2,
                ),
                (
                    "linear1_default_exact_input",
                    ttnn.matmul(to_device(exact_text_norm, device), weights.text_in),
                    exact_linear1,
                ),
                (
                    "linear2_default_exact_input",
                    ttnn.matmul(to_device(expected_gelu, device), weights.text_out),
                    expected_linear2,
                ),
            )
            for name, result, reference in isolated:
                actual = to_host(result, tuple(reference.shape))
                difference = (actual.float() - reference.float()).abs()
                row = {
                    "component": name,
                    "max_abs_error": float(difference.max()),
                    "mean_abs_error": float(difference.mean()),
                    "relative_rms_error": float(
                        difference.square().mean().sqrt() / reference.float().square().mean().sqrt()
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
