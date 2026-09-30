# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate the TT VAE decoder against an independent CUDA decode capture."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image


def metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    # Large images need float64 reductions: float32 dot/norm paths can produce
    # noticeably inconsistent sums (including apparent PCC above one).
    a, b = actual.double().flatten(), expected.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    error = a - b
    return {
        "pcc": float(torch.dot(ac, bc) / (ac.norm() * bc.norm())),
        "relative_rms_error": float(error.square().mean().sqrt() / b.square().mean().sqrt()),
        "mean_abs_error": float(error.abs().mean()),
        "max_abs_error": float(error.abs().max()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-bdf", required=True)
    parser.add_argument("--stop-after", help="stop a fresh component diagnostic at this captured stage")
    parser.add_argument(
        "--isolated-only", action="store_true", help="compare individual components using exact CUDA inputs"
    )
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads((args.cuda_dir / "manifest.json").read_text())
    os.environ["TT_VISIBLE_DEVICES"] = args.device_bdf
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)
    import ttnn
    from models.experimental.qwen_image_2_1.tt.tt_vae import QwenImage21VAEDecoder

    reports = []
    started = time.monotonic()

    def progress(status, stage=None, error=None):
        path = args.output_dir / "progress.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "status": status,
                    "stage": stage,
                    "error": error,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_stages": len(reports),
                },
                indent=2,
            )
            + "\n"
        )
        temporary.replace(path)

    class StopAfter(Exception):
        pass

    def observe(name, value, height, width):
        progress("running", name)
        expected_path = args.cuda_dir / ("vae_input.pt" if name == "vae_input" else f"{name}/output.pt")
        if not expected_path.is_file():
            return
        actual = decoder.collect(value, height, width, value.shape[-1])
        expected = torch.load(expected_path, map_location="cpu", weights_only=True)
        if expected.ndim == 4:
            expected = expected.unsqueeze(2)
        row = {"stage": name, "shape": list(actual.shape), **metrics(actual, expected)}
        torch.save(actual, args.output_dir / f"{name}.pt")
        reports.append(row)
        (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if name == args.stop_after:
            raise StopAfter

    device = None
    try:
        progress("starting")
        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        decoder = QwenImage21VAEDecoder(args.checkpoint, device)
        if args.isolated_only:
            names = [row["name"] for row in manifest["stages"]]
            for name in names:
                source = torch.load(args.cuda_dir / name / "input.pt", map_location="cpu", weights_only=True)
                if source.ndim == 5:
                    source = source[:, :, 0]
                h, w = source.shape[-2:]
                value = decoder.upload(source.permute(0, 2, 3, 1).reshape(1, 1, h * w, source.shape[1]))
                if name + ".weight" in decoder.state:
                    value = decoder.conv(value, name, h, w)
                elif name + ".gamma" in decoder.state:
                    value = decoder.norm(value, name)
                elif name == "decoder.mid_block.resnets.0":
                    value = decoder.residual(value, name, h, w)
                elif name == "decoder.mid_block.attentions.0":
                    value = decoder.attention(value, name, h, w)
                elif name.endswith("upsampler.resample.0"):
                    value = decoder.nearest(value, h, w)
                    h, w = h * 2, w * 2
                elif name.endswith("avg_shortcut"):
                    block = int(name.split(".")[2])
                    out_channels = (1152, 1152, 576, 288)[block]
                    factors = decoder.config["temperal_downsample"][::-1]
                    value = decoder.duplicate_shortcut(
                        value, source.shape[1], out_channels, 2 if factors[block] else 1, h, w
                    )
                    h, w = h * 2, w * 2
                else:
                    continue
                observe(name, value, h, w)
                if reports[-1]["pcc"] < 0.99:
                    raise AssertionError(f"isolated component PCC below 0.99: {reports[-1]}")
            progress("complete", "isolated_components")
            return
        latents = torch.load(args.cuda_dir / "packed_latents.pt", map_location="cpu", weights_only=True)
        height, width = manifest["height"], manifest["width"]
        decoded = decoder.decode(decoder.upload(latents), height, width, observe)
        output = decoder.collect(decoded, height, width, decoder.config["out_channels"])
        expected = torch.load(args.cuda_dir / "output.pt", map_location="cpu", weights_only=True)
        row = {"stage": "output", "shape": list(output.shape), **metrics(output, expected)}
        reports.append(row)
        torch.save(output, args.output_dir / "output.pt")
        pixels = ((output[:, :, 0].float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        Image.fromarray(pixels[0].permute(1, 2, 0).numpy()).save(args.output_dir / "tt_vae.png")
        (args.output_dir / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        progress("complete", "output")
        if row["pcc"] < 0.98:
            raise AssertionError(f"decoded tensor PCC below 0.98: {row}")
    except StopAfter:
        progress("partial", args.stop_after)
    except Exception as error:
        progress("failed", error=str(error))
        raise
    finally:
        if device is not None:
            ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
