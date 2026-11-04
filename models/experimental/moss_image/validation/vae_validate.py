# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Replay captured CUDA VAE components and the complete TT decoder.

Run from the repository root using the built TT-Metal environment. The caller owns device reservation and
must set TT_VISIBLE_DEVICES before starting this process. Capture injection is
explicit here; this command is not an integrated text-to-image qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch


def metrics(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError("Invalid TT VAE output shape or nonfinite output")
    delta = actual - expected
    ac, ec = actual - actual.mean(), expected - expected.mean()
    denominator = ac.norm() * ec.norm()
    return {
        "pcc": float((ac @ ec / denominator).item()) if denominator else float(torch.equal(actual, expected)),
        "relative_rms": float(delta.norm() / expected.norm().clamp_min(1e-30)),
        "max_abs": float(delta.abs().max()),
        "reference_rms": float(expected.square().mean().sqrt()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", default="all", help="all, decoder, preprocess, or a captured stage name")
    parser.add_argument("--cards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--l1-small-size", type=int, default=131072)
    parser.add_argument("--min-pcc", type=float, default=0.99)
    parser.add_argument("--max-relative-rms", type=float, default=0.03)
    args = parser.parse_args()

    import ttnn
    from models.experimental.moss_image.tt.checkpoint import sha256, verify_checkpoint_manifest
    from models.experimental.moss_image.tt.vae import TTMossVAE

    manifest = json.loads((args.capture / "manifest.json").read_text())
    artifacts = manifest["artifacts"]
    verify_checkpoint_manifest(args.checkpoint, manifest, components=["vae"])

    def load(key):
        record = artifacts[key]
        path = (args.capture / record["path"]).resolve()
        if not path.is_relative_to(args.capture.resolve()):
            raise ValueError("Capture path escapes capture directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Capture digest mismatch: {key}")
        return torch.load(path, map_location="cpu", weights_only=True)

    def packed_latent():
        last = max(record["index"] for record in manifest["steps"])
        latent = load(f"steps/{last:03d}/updated_latents")
        return latent.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()

    stage_keys = sorted(key for key in artifacts if key.startswith("vae/stages/"))
    cases = [key.removeprefix("vae/stages/") for key in stage_keys]
    cases = [
        name
        for name in cases
        if name in ("post_quant_conv", "decoder.conv_in", "decoder.conv_norm_out", "decoder.conv_out")
        or ".resnets." in name
        or name.endswith(".attentions.0")
        or ".upsamplers.0.conv" in name
    ]
    cases = ["preprocess", *cases, "decoder"] if args.case == "all" else [args.case]
    report = {
        "capture": str(args.capture.resolve()),
        "cards": args.cards,
        "activation_injection": True,
        "scope": "CUDA component replay and TT full VAE decode",
        "thresholds": {"min_pcc": args.min_pcc, "max_relative_rms": args.max_relative_rms},
        "cases": [],
    }
    try:
        ttnn_version = importlib.metadata.version("ttnn")
    except importlib.metadata.PackageNotFoundError:
        ttnn_version = "source installation without package metadata"
    report["environment"] = {
        "torch": torch.__version__,
        "ttnn": ttnn_version,
        "tt_metal_home": os.environ.get("TT_METAL_HOME"),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "l1_small_size": args.l1_small_size,
        "sources_sha256": {
            str(path.relative_to(Path(__file__).parents[1])): sha256(path)
            for path in (Path(__file__), Path(__file__).parents[1] / "tt/vae.py")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, args.cards),
        physical_device_ids=list(range(args.cards)),
        # Full-resolution sliced convolutions retain per-slice halo metadata.
        # 32 KiB is insufficient for the complete 1024px decoder's config cache.
        l1_small_size=args.l1_small_size,
    )
    try:
        model = TTMossVAE(args.checkpoint, device)
        raw = load("vae/raw_output")
        height, width = raw.shape[-2:]
        for name in cases:
            begin = time.perf_counter()
            if name in ("preprocess", "decoder"):
                x = model.upload(packed_latent())
                if name == "preprocess":
                    expected = load("vae/input")
                    value = model.preprocess(x, height, width)
                else:
                    expected = raw.clamp(-1, 1)
                    value = model.decode(x, height, width)
            else:
                stage = load("vae/stages/" + name)
                reference_input, expected = stage["input"], stage["output"]
                if isinstance(expected, (tuple, list)):
                    expected = expected[0]
                n, channels, h, w = reference_input.shape
                x = model.upload(reference_input.permute(0, 2, 3, 1).reshape(n, 1, h * w, channels))
                if ".resnets." in name:
                    value = model.residual(x, name, h, w)
                elif ".attentions." in name:
                    value = model.attention(x, name, h, w)
                elif name == "decoder.conv_norm_out":
                    value = model.norm(x, name)
                else:
                    value = model.conv(x, name, h, w)
            n, channels, h, w = expected.shape
            value = ttnn.reshape(value, (n, h, w, channels))
            actual = model.collect(value)
            per_card = [
                metrics(ttnn.to_torch(local).float().permute(0, 3, 1, 2), expected.float())
                for local in ttnn.get_device_tensors(value)
            ]
            ttnn.synchronize_device(device)
            measured = metrics(actual, expected.float())
            passed = all(
                card["pcc"] >= args.min_pcc and card["relative_rms"] <= args.max_relative_rms for card in per_card
            )
            record = {
                "name": name,
                "shape": list(expected.shape),
                **measured,
                "passed": passed,
                "per_card": per_card,
                "seconds_including_compile_and_transfer": time.perf_counter() - begin,
            }
            report["cases"].append(record)
            if name == "decoder":
                torch.save(actual, args.output.parent / "tt_vae_output.pt")
                from PIL import Image

                pixels = ((actual[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
                Image.fromarray(pixels).save(args.output.parent / "tt_vae_output.png")
            report["passed"] = all(case["passed"] for case in report["cases"])
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    finally:
        ttnn.close_mesh_device(device)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
