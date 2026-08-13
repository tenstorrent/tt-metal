# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TT VAE encode→decode smoke test for Cosmos3-I2V on BH/WH meshes.

Loads the AutoencoderKLWan (Cosmos3 z_dim=48, patch_size=(1,2,2)) and runs
its encode + decode entirely through `Cosmos3VAEEncoderAdapter` /
`Cosmos3VAEDecoderAdapter`. Skips the 64B transformer entirely so this
script is a cheap way to verify the TT-NN VAE adapters in isolation.

Compares the TT roundtrip to the host PyTorch reference:
  - Per-channel mean / std / min / max
  - Image-domain delta from input frame
  - Writes one decoded PNG frame to disk for visual inspection

Usage (BH Galaxy 4x8):

    python -m models.tt_dit.experimental.cosmos3_i2v.demo.vae_tt_smoke \\
        --image /path/to/ref.jpg --mesh-shape 4x8 --out /tmp/cosmos3_vae_tt.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cosmos3 TT-NN VAE encode/decode smoke test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, type=Path, help="Reference image (JPEG/PNG).")
    p.add_argument("--height", type=int, default=256, help="Encode height (multiple of 16).")
    p.add_argument("--width", type=int, default=256, help="Encode width (multiple of 16).")
    p.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Frames to tile the image to (must be 4k+1 for the Wan VAE; 5 keeps decode latent T=2).",
    )
    p.add_argument(
        "--mesh-shape",
        default="auto",
        help="Mesh shape RxC (e.g. 4x8). 'auto' picks the largest fitting shape.",
    )
    p.add_argument("--num-links", type=int, default=None, help="CCL num_links. Default = auto.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("cosmos3_vae_tt.png"),
        help="Where to save one decoded PNG frame.",
    )
    p.add_argument("--hf-repo", default=None, help="HF repo override.")
    p.add_argument(
        "--skip-host-ref",
        action="store_true",
        help="Skip running the host PyTorch reference (saves time / host RAM on small boxes).",
    )
    return p.parse_args(argv)


def _resolve_mesh_shape(spec: str, available: int) -> tuple[int, int]:
    if spec == "auto":
        if available >= 32:
            return (4, 8)
        if available >= 8:
            return (1, 8)
        if available >= 4:
            return (1, 4)
        if available >= 2:
            return (1, 2)
        return (1, 1)
    rows, cols = (int(x) for x in spec.lower().split("x"))
    if rows * cols > available:
        raise SystemExit(f"mesh {rows}x{cols} > available {available}")
    return (rows, cols)


def _stats(name: str, t) -> str:
    import torch

    f = t.detach().to(torch.float32).cpu()
    return (
        f"{name}: shape={tuple(t.shape)} min={f.min().item():.4f} max={f.max().item():.4f} "
        f"mean={f.mean().item():.4f} std={f.std().item():.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise SystemExit(f"height/width must be multiples of 16 (got {args.height}x{args.width})")
    if (args.num_frames - 1) % 4 != 0:
        raise SystemExit(f"--num-frames must be 4k+1 (got {args.num_frames}); try 5, 9, 13, ...")

    import diffusers as _diffusers
    import numpy as np
    import torch
    from diffusers.video_processor import VideoProcessor
    from PIL import Image

    import ttnn
    from models.tt_dit.experimental.cosmos3_i2v.demo.generate import close_mesh, open_mesh
    from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO
    from models.tt_dit.experimental.cosmos3_i2v.tokenizer.vae_cosmos3 import (
        Cosmos3VAEDecoderAdapter,
        Cosmos3VAEEncoderAdapter,
    )
    from models.tt_dit.parallel.config import VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager

    hf_repo = args.hf_repo or HF_REPO

    available = ttnn.get_num_devices()
    mesh_shape = _resolve_mesh_shape(args.mesh_shape, available)
    print(f"[vae-tt-smoke] mesh={mesh_shape} (of {available} available), hf_repo={hf_repo}", flush=True)

    print(f"[vae-tt-smoke] Loading host AutoencoderKLWan from {hf_repo}/vae ...", flush=True)
    torch_vae = _diffusers.AutoencoderKLWan.from_pretrained(
        hf_repo, subfolder="vae", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    torch_vae.eval()
    torch.set_grad_enabled(False)
    print(
        f"[vae-tt-smoke] config: z_dim={torch_vae.config.z_dim}, patch_size={torch_vae.config.patch_size}", flush=True
    )

    sf_s = int(torch_vae.config.scale_factor_spatial)
    video_processor = VideoProcessor(vae_scale_factor=sf_s, resample="bilinear")
    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    frame_2d = video_processor.preprocess(image, height=args.height, width=args.width).to(dtype=torch.bfloat16)
    frames_BCTHW = frame_2d.unsqueeze(2).expand(-1, -1, args.num_frames, -1, -1).contiguous()
    print(_stats("[vae-tt-smoke] input video", frames_BCTHW), flush=True)

    mesh = open_mesh(mesh_shape)
    try:
        # Resolve num_links the way build_cosmos3_i2v_native_pipeline does.
        if args.num_links is not None:
            num_links = args.num_links
        elif ttnn.device.is_blackhole():
            num_links = 2
        elif mesh_shape == (4, 8):
            num_links = 4
        else:
            num_links = 1
        print(f"[vae-tt-smoke] num_links={num_links}", flush=True)

        tp_axis = max(range(len(mesh_shape)), key=lambda i: mesh_shape[i])
        sp_axis = 1 - tp_axis
        parallel_config = VaeHWParallelConfig.from_tuples(
            height=(mesh_shape[tp_axis], tp_axis),
            width=(mesh_shape[sp_axis], sp_axis),
        )
        ccl_manager = CCLManager(mesh_device=mesh, num_links=num_links, topology=ttnn.Topology.Linear)

        enc = Cosmos3VAEEncoderAdapter(
            checkpoint_name=hf_repo,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            encoder_t_chunk_size=None,
            vae_dtype=ttnn.bfloat16,
            torch_vae=torch_vae,
        )

        # If TT_DIT_VAE_DEBUG=1, also dump host-side per-level stats from the
        # torch encoder by hooking each down_block / mid_block forward.
        if os.environ.get("TT_DIT_VAE_DEBUG") == "1":
            hooks = []
            host_log: list[tuple[str, tuple, float, float, float, float]] = []
            torch_encoder = torch_vae.encoder

            def _make_hook(label: str):
                def _h(_mod, _inp, out):
                    o = out.detach().to(torch.float32).cpu()
                    host_log.append(
                        (label, tuple(o.shape), o.min().item(), o.max().item(), o.mean().item(), o.std().item())
                    )

                return _h

            hooks.append(torch_encoder.conv_in.register_forward_hook(_make_hook("host:post_conv_in")))
            for i, db in enumerate(torch_encoder.down_blocks):
                hooks.append(db.register_forward_hook(_make_hook(f"host:down_block[{i}]:{type(db).__name__}")))
            hooks.append(torch_encoder.mid_block.register_forward_hook(_make_hook("host:post_mid_block")))

            try:
                with torch.no_grad():
                    _ = torch_vae.encode(frames_BCTHW).latent_dist.mode()
            finally:
                for h in hooks:
                    h.remove()
            for label, shape, mn, mx, mean, std in host_log:
                print(
                    f"[wan-encoder-dbg] {label}: shape={shape} min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={std:.4f}",
                    flush=True,
                )

        t0 = time.time()
        tt_mu = enc.encode(frames_BCTHW)
        print(f"[vae-tt-smoke] TT encode done in {time.time() - t0:.2f}s", flush=True)
        print(_stats("[vae-tt-smoke] tt raw_mu", tt_mu), flush=True)

        # Build decoder sized to the pixel-domain dims (matches the lazy-init
        # path used inside build_cosmos3_i2v_native_pipeline).
        dec = Cosmos3VAEDecoderAdapter(
            checkpoint_name=hf_repo,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            vae_t_chunk_size=None,
            vae_dtype=ttnn.bfloat16,
            torch_vae=torch_vae,
        )

        t1 = time.time()
        tt_video = dec.decode(tt_mu, output_type="pt")
        print(f"[vae-tt-smoke] TT decode done in {time.time() - t1:.2f}s", flush=True)
        print(_stats("[vae-tt-smoke] tt decoded video", tt_video), flush=True)

        # Optional host reference comparison.
        if not args.skip_host_ref:
            t2 = time.time()
            ref_mu = torch_vae.encode(frames_BCTHW).latent_dist.mode()
            ref_video = torch_vae.decode(ref_mu).sample
            print(f"[vae-tt-smoke] host roundtrip done in {time.time() - t2:.2f}s", flush=True)

            tt_mu_t = tt_mu[
                : ref_mu.shape[0], : ref_mu.shape[1], : ref_mu.shape[2], : ref_mu.shape[3], : ref_mu.shape[4]
            ]
            tt_video_t = tt_video[
                : ref_video.shape[0],
                : ref_video.shape[1],
                : ref_video.shape[2],
                : ref_video.shape[3],
                : ref_video.shape[4],
            ]

            def _delta(name, a, b):
                d = (a.float() - b.float()).abs()
                rel = d.mean() / (b.float().abs().mean() + 1e-6)
                print(
                    f"[vae-tt-smoke] Δ {name}: abs_mean={d.mean().item():.4f} "
                    f"abs_max={d.max().item():.4f} rel_mean={rel.item():.4f}",
                    flush=True,
                )

            _delta("raw_mu (TT vs host)", tt_mu_t, ref_mu)
            _delta("decoded video (TT vs host)", tt_video_t, ref_video)

        # Save one frame for eyeballing.
        out = video_processor.postprocess_video(tt_video, output_type="np")[0]
        out_u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        middle = out_u8[out_u8.shape[0] // 2]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(middle).save(args.out)
        print(f"[vae-tt-smoke] wrote {args.out}", flush=True)

        # Verdict: dynamic range should be non-trivial.
        span = float(out.max()) - float(out.min())
        if span < 0.02:
            print(
                f"[vae-tt-smoke] VERDICT: BROKEN. Output dynamic range {span:.4f} too small — TT VAE is dead.",
                flush=True,
            )
            return 1
        print(f"[vae-tt-smoke] VERDICT: roundtrip works (dynamic range {span:.4f}).", flush=True)
        return 0

    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    sys.exit(main())
