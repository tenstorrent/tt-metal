# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only fallback visualization of already-saved stock repeatability latents."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKLFlux2, Flux2Pipeline
from diffusers.image_processor import VaeImageProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    report = json.loads((args.source / "report.json").read_text())
    assert report["status"] == "completed"
    output = args.source / "decoded-cpu"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    vae = AutoencoderKLFlux2.from_pretrained(args.checkpoint, subfolder="vae", torch_dtype=torch.float32).eval()
    processor = VaeImageProcessor()
    rows = []
    for record in report["rows"]:
        if record["mode"] == "capture":
            continue
        started = time.perf_counter()
        stem = f"{record['index']}-{record['mode']}"
        path = args.source / f"{stem}.pt"
        packed = torch.load(path, weights_only=True).float()
        assert list(packed.shape) == [1, 4096, 128] and bool(torch.isfinite(packed).all())
        with torch.no_grad():
            patched = packed.transpose(1, 2).reshape(1, 128, 64, 64)
            mean = vae.bn.running_mean.view(1, -1, 1, 1)
            std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
            z = Flux2Pipeline._unpatchify_latents(patched * std + mean)
            decoded = vae.decode(z, return_dict=False)[0]
            assert bool(torch.isfinite(decoded).all())
            image = processor.postprocess(decoded, output_type="pil")[0]
        image_path = output / f"{stem}.png"
        image.save(image_path)
        rows.append(
            dict(
                index=record["index"],
                mode=record["mode"],
                image=image_path.name,
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                latents_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        print("CPU_REPEAT_DECODE", rows[-1], flush=True)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="completed",
                source=str(args.source),
                checkpoint=args.checkpoint,
                decoder="Diffusers CPU FP32 VAE, no tiling; identical decoder for all six repeats",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
