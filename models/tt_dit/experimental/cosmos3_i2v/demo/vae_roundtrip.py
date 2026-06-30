# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""VAE encode→decode roundtrip diagnostic for Cosmos3-I2V.

Bypasses the transformer entirely. Loads the AutoencoderKLWan VAE used by
Cosmos3 (with z_dim=48 and the Cosmos3 latents_mean/std), encodes the
input image, applies the same normalization the pipeline does, immediately
reverses it, decodes, postprocesses, and writes one PNG frame.

What it tells you:
  - Black output  → VAE itself is broken on this diffusers pin / config
    (z_dim=48 not handled, latents_mean/std mismatch, kwarg mismatch, etc.)
  - Recognizable image → VAE roundtrip works. The black-output bug in the
    full demo lives in the transformer trunk or the scheduler, NOT the
    VAE. Next diagnostic should be output_type='latent' on the full demo
    to inspect what the trunk actually produces.

Run on a TT host or any CPU/GPU box that can load the HF repo:

    python -m models.tt_dit.experimental.cosmos3_i2v.demo.vae_roundtrip \\
        --image ../ref.jpg --out /tmp/vae_roundtrip.png

Does NOT need a TT device — VAE runs as host PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cosmos3 VAE encode/decode roundtrip diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, type=Path, help="Reference image (JPEG/PNG).")
    p.add_argument("--height", type=int, default=256, help="Encode/decode height (must be multiple of 16).")
    p.add_argument("--width", type=int, default=256, help="Encode/decode width (must be multiple of 16).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("vae_roundtrip.png"),
        help="Output PNG path. Save one decoded frame.",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="HF repo to pull VAE from (defaults to the project's HF_REPO).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise SystemExit(
            f"height/width must be multiples of 16 (the Wan VAE's spatial compression). Got {args.height}x{args.width}."
        )

    # The Cosmos3 pipeline uses `AutoencoderKLWan` from diffusers, but the actual repo subfolder
    # is just "vae" — load directly without going through the full pipeline (which would also
    # pull in the 64B transformer).
    import diffusers as _diffusers
    import numpy as np
    import torch
    from diffusers.video_processor import VideoProcessor
    from PIL import Image

    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO
    from models.tt_dit.experimental.cosmos3_i2v.reference.pipeline_cosmos3_omni import Cosmos3OmniPipeline  # noqa: F401

    hf_repo = args.hf_repo or HF_REPO
    vae_cls = _diffusers.AutoencoderKLWan
    print(f"[vae-roundtrip] Loading VAE from {hf_repo}/vae ...", flush=True)
    vae = vae_cls.from_pretrained(hf_repo, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()
    torch.set_grad_enabled(False)

    # Cosmos3 normalization stats. The pipeline reads these from vae.config; mirror that.
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype)
    latents_inv_std = 1.0 / torch.tensor(vae.config.latents_std, dtype=vae.dtype)
    z_dim = latents_mean.shape[0]
    print(f"[vae-roundtrip] z_dim={z_dim}, latents_mean.shape={tuple(latents_mean.shape)}", flush=True)
    print(
        f"[vae-roundtrip] vae.config: scale_factor_spatial={vae.config.scale_factor_spatial}, "
        f"scale_factor_temporal={vae.config.scale_factor_temporal}",
        flush=True,
    )

    # Preprocess image to [-1, 1] tensor matching the pipeline's path.
    vae_scale_factor_spatial = int(vae.config.scale_factor_spatial)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial, resample="bilinear")
    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    # video_processor.preprocess → [1, 3, H, W] in [-1, 1].
    frame_2d = video_processor.preprocess(image, height=args.height, width=args.width).to(dtype=vae.dtype)

    # The VAE wants [B, C, T, H, W]. Single-frame: T = scale_factor_temporal (so the encoded
    # latent has T_latent ≥ 1). Tile the image along time so we have something the temporal
    # encoder can compress.
    t_in = int(vae.config.scale_factor_temporal)
    frames_3d = frame_2d.unsqueeze(2).expand(-1, -1, t_in, -1, -1).contiguous()
    print(f"[vae-roundtrip] input frames shape: {tuple(frames_3d.shape)}", flush=True)

    # Encode
    enc = vae.encode(frames_3d)
    if hasattr(enc, "latent_dist"):
        raw_mu = enc.latent_dist.mode()
    elif hasattr(enc, "latents"):
        raw_mu = enc.latents
    else:
        raise RuntimeError(f"Could not extract latents from encoder output: {type(enc)}")
    print(
        f"[vae-roundtrip] encoded raw_mu shape: {tuple(raw_mu.shape)}, "
        f"min={raw_mu.float().min().item():.4f} max={raw_mu.float().max().item():.4f} "
        f"mean={raw_mu.float().mean().item():.4f}",
        flush=True,
    )

    # Apply Cosmos3's normalization then immediately reverse it (math is identity).
    mean = latents_mean.view(1, -1, 1, 1, 1)
    inv_std = latents_inv_std.view(1, -1, 1, 1, 1)
    latents = (raw_mu - mean) * inv_std
    print(
        f"[vae-roundtrip] normalized latents: "
        f"min={latents.float().min().item():.4f} max={latents.float().max().item():.4f} "
        f"mean={latents.float().mean().item():.4f} std={latents.float().std().item():.4f}",
        flush=True,
    )
    z_raw = latents / inv_std + mean

    # Decode
    decoded = vae.decode(z_raw).sample
    print(
        f"[vae-roundtrip] decoded shape: {tuple(decoded.shape)}, "
        f"min={decoded.float().min().item():.4f} max={decoded.float().max().item():.4f} "
        f"mean={decoded.float().mean().item():.4f}",
        flush=True,
    )

    # postprocess_video expects [B, C, T, H, W] → with output_type="np" returns float32 in [0,1]
    # shaped [T, H, W, 3]. Convert to uint8 for PIL.
    video = video_processor.postprocess_video(decoded, output_type="np")[0]
    print(
        f"[vae-roundtrip] postprocessed video: shape={tuple(video.shape)}, "
        f"dtype={video.dtype}, min={float(np.min(video)):.4f} max={float(np.max(video)):.4f} "
        f"mean={float(np.mean(video)):.4f}",
        flush=True,
    )

    video_u8 = (np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)
    middle = video_u8[video_u8.shape[0] // 2]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(middle).save(args.out)
    print(f"[vae-roundtrip] wrote {args.out}", flush=True)

    # Verdict
    span = float(np.max(video)) - float(np.min(video))  # in [0, 1]
    if span < 0.02:
        print(
            "[vae-roundtrip] VERDICT: VAE roundtrip is BROKEN. "
            f"Output dynamic range is {span:.4f} (on a 0-1 scale). The bug is in the VAE itself, the "
            "Cosmos3 latents_mean/std config, or the diffusers AutoencoderKLWan implementation on this pin.",
            flush=True,
        )
    else:
        print(
            "[vae-roundtrip] VERDICT: VAE roundtrip works. "
            f"Output dynamic range is {span:.4f} (on a 0-1 scale). The black-video bug in the full demo "
            "lives upstream — trunk output or scheduler, NOT the VAE. Next: re-run the demo with "
            "output_type='latent' and inspect the latent stats.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
