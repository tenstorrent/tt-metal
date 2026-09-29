# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Check TT's first FlowMatch Euler step against the CUDA pipeline capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--next-sigma", type=float, default=0.02)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = args.cuda_dir / "step_000"

    def read(name: str) -> torch.Tensor:
        return torch.load(base / f"{name}.pt", map_location="cpu", weights_only=True)

    latents = read("transformer/input/hidden_states")
    prediction = read("transformer/output")[:, -latents.shape[1] :]
    expected = read("scheduler/updated_latents")

    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_dit_components import to_device, to_host
    from models.experimental.qwen_image_2_1.tt.tt_scheduler import flow_euler_step

    device = None
    try:
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
        result = flow_euler_step(
            to_device(latents, device), to_device(prediction, device), args.sigma, args.next_sigma, device
        )
        actual = to_host(result, tuple(expected.shape))
        destination = args.output_dir / "step_000/scheduler/updated_latents.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actual, destination)
        difference = (actual.float() - expected.float()).abs()
        report = {
            "sigma": args.sigma,
            "next_sigma": args.next_sigma,
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "exact_fraction": float((actual == expected).float().mean()),
            "relative_rms_error": float(difference.square().mean().sqrt() / expected.float().square().mean().sqrt()),
        }
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
