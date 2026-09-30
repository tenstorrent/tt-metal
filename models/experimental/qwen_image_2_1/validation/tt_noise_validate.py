# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate device Gaussian generation and export a paired CUDA oracle input."""

import argparse
import json
import os
from importlib.metadata import version
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-paired", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cuda_paired:
        data = torch.load(args.output_dir / "paired_inputs.pt", weights_only=True)
        radial = data["radial"].cuda()
        angular = data["angular"].cuda()
        reference = (
            (torch.sqrt(-2 * torch.log(radial.clamp_min(2.0**-24))) * torch.cos(angular * (2 * torch.pi)))
            .to(torch.bfloat16)
            .cpu()
        )
        actual = data["gaussian"].double().flatten()
        expected = reference.double().flatten()
        delta = actual - expected
        report = json.loads((args.output_dir / "report.json").read_text())
        report["paired_cuda"] = {
            "pcc": float(torch.corrcoef(torch.stack((actual, expected)))[0, 1]),
            "relative_rms_error": float(delta.norm() / expected.norm()),
            "max_abs_error": float(delta.abs().max()),
            "oracle": "CUDA Box-Muller applied to the same TT-generated uniforms",
        }
        torch.save(reference, args.output_dir / "cuda_gaussian.pt")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["paired_cuda"]["pcc"] >= 0.999
        print(json.dumps(report, indent=2))
        return report
    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn

    from models.experimental.qwen_image_2_1.tt.tt_noise import (
        gaussian_from_uniforms,
        gaussian_uniforms,
        initial_image_latents,
    )

    shape = (1, (args.height // 16) * (args.width // 16), 64)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0])
    try:
        radial, angular = gaussian_uniforms(shape, device, args.seed)
        gaussian = ttnn.typecast(gaussian_from_uniforms(radial, angular), ttnn.bfloat16)
        host = ttnn.to_torch(gaussian).reshape(shape)
        paired = {
            "radial": ttnn.to_torch(radial).reshape(shape),
            "angular": ttnn.to_torch(angular).reshape(shape),
            "gaussian": host,
        }
        torch.save(paired, args.output_dir / "paired_inputs.pt")
        repeated = ttnn.to_torch(initial_image_latents(device, args.height, args.width, args.seed)).reshape(shape)
        different = ttnn.to_torch(initial_image_latents(device, args.height, args.width, args.seed + 1)).reshape(shape)
        zero_a = ttnn.to_torch(initial_image_latents(device, args.height, args.width, 0)).reshape(shape)
        zero_b = ttnn.to_torch(initial_image_latents(device, args.height, args.width, 0)).reshape(shape)
        # A larger sample checks moments without requiring an enormous image.
        large_radial, large_angular = gaussian_uniforms((1, 4096, 64), device, args.seed)
        sample = ttnn.to_torch(gaussian_from_uniforms(large_radial, large_angular)).double().flatten()
        centered = sample - sample.mean()
        variance = centered.square().mean()
        report = {
            "shape": list(shape),
            "seed": args.seed,
            "same_seed_bitwise": torch.equal(host, repeated),
            "zero_seed_bitwise": torch.equal(zero_a, zero_b),
            "different_seed_changes_output": not torch.equal(host, different),
            "sample_count": sample.numel(),
            "finite": bool(torch.isfinite(sample).all()),
            "mean": float(sample.mean()),
            "std": float(sample.std()),
            "skewness": float(centered.pow(3).mean() / variance.pow(1.5)),
            "kurtosis": float(centered.pow(4).mean() / variance.square()),
            "p_abs_gt_2": float((sample.abs() > 2).double().mean()),
            "p_abs_gt_3": float((sample.abs() > 3).double().mean()),
            "ttnn_version": version("ttnn"),
            "device_bdf": args.device_bdf,
            "rng": "native TT uniforms and TT Box-Muller; no CPU/CUDA random tensor",
        }
        torch.save(host, args.output_dir / "initial_latents.pt")
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["same_seed_bitwise"] and report["zero_seed_bitwise"]
        assert report["different_seed_changes_output"] and report["finite"]
        assert abs(report["mean"]) < 0.02 and abs(report["std"] - 1) < 0.02
        assert abs(report["skewness"]) < 0.06 and abs(report["kurtosis"] - 3) < 0.12
        assert abs(report["p_abs_gt_2"] - 0.0455) < 0.004
        assert abs(report["p_abs_gt_3"] - 0.0027) < 0.001
        print(json.dumps(report, indent=2))
        return report
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
