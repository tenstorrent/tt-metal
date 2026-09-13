# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-stage ground truth for the ttnn DiffVAE port.

Final pixels alone cannot localise a bug in a 24-block network — the Gemma-4 investigation
made that point expensively. So this drives upstream's decoder one stage at a time and saves
every boundary a ttnn module will have to reproduce.

Two things it does that ``decoder.decode_video`` does not:

* **Injects the noise.** Stage 5 predicts x0 from ``randn`` noise, so the noise is an input,
  not an implementation detail. Matching seeds across torch-on-host and ttnn-on-device would
  not produce the same tensor, so it is drawn once here and written out to be uploaded.
* **Bypasses tiling.** Single tile, single temporal group, so a ttnn stage can be compared
  against one contiguous tensor. Tiling is a separate concern to validate later.

The stage sequence mirrors ``_decode_pixels`` for the untiled single-tile case; if upstream
changes that order this will drift, so keep it next to the source when updating.

  PYTHONPATH=~/LTX-2/packages/ltx-core/src:. python capture_stages.py \
      latents/latent_0_1x128x4x34x60.pt --crop 10 --out stages/crop10.safetensors

The latent argument also accepts ``randn:BxCxFxHxW``, which draws one instead of reading a dump.
A capture only has to exercise the decoder on a numerically sane input for parity to mean
something, so a real latent is not required to check the port against upstream — but the two
differ in distribution, so a synthetic capture is not evidence about output quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from host_ref import load


ENDPOINTS = ("input.latent", "stage5.noise", "output.pixels")


def capture(decoder, latent: torch.Tensor, *, seed: int, pixels_only: bool = False) -> dict[str, torch.Tensor]:
    """Run the decoder stage by stage, returning every intermediate a port must match.

    ``pixels_only`` keeps just the decoder's endpoints (latent in, noise, pixels out). The
    intermediates are what localise a bug, but they are also ~40x the volume, so a run only
    needed for an end-to-end number does not have to pay for them.
    """
    from ltx_core.model.video_vae import diffusion_tiling

    out: dict[str, torch.Tensor] = {}

    def record(name: str, tensor: torch.Tensor) -> None:
        if not pixels_only or name in ENDPOINTS:
            out[name] = tensor.clone()

    record("input.latent", latent)

    # Same two shape adjustments the real decode path applies before stage 1: a floor so the
    # NA kernels are satisfied, then trailing-frame replication for the NATTEN border.
    latent, pads = diffusion_tiling.ensure_min_latent_shape(latent, decoder.stage_min_tile_sizes)
    padded = diffusion_tiling.pad_trailing_latent_for_natten_border(latent, decoder._natten_trailing_pad_latent_frames)
    record("input.latent_padded", padded)
    print(f"latent {tuple(latent.shape)} -> padded {tuple(padded.shape)}  pads={pads}", flush=True)

    hidden = decoder.per_channel_statistics.un_normalize(padded).permute(0, 2, 3, 4, 1)
    hidden = decoder.conv_in(hidden)
    record("stage0.conv_in", hidden)

    # Stages 1-3 individually rather than through forward_stages_1_to_3, so each NA stack and
    # its upsample land in the dump separately.
    for stage in range(3):
        for depth, block in enumerate(decoder.det_stages[stage]):
            hidden = block(hidden)
            record(f"det{stage}.block{depth}", hidden)
        hidden = decoder.upsamples[stage](hidden, drop_leading_frame=True)
        record(f"det{stage}.upsampled", hidden)
        print(f"  after det stage {stage}: {tuple(hidden.shape)}", flush=True)

    context = decoder.forward_stage_4(hidden, drop_leading_frame=True, pad_trailing=True)
    record("stage4.context", context)
    print(f"  stage-4 context: {tuple(context.shape)}", flush=True)

    # Stage-5 canvas: what _decode_temporal_group_isolated would draw noise into for a single
    # tile covering the whole volume.
    batch = latent.shape[0]
    frames, height, width = context.shape[1], context.shape[2], context.shape[3]
    noise = torch.randn(
        (batch, decoder.out_channels, frames, height * decoder.patch_size, width * decoder.patch_size),
        generator=torch.Generator().manual_seed(seed),
        dtype=context.dtype,
    )
    record("stage5.noise", noise)
    print(f"  stage-5 noise: {tuple(noise.shape)}", flush=True)

    timestep = decoder.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
    t_now = timestep[:, -1]
    t_emb = decoder.t_embedder(decoder.timestep_scale_multiplier * t_now, hidden_dtype=context.dtype)
    record("stage5.t_emb", t_emb)
    # AdaLNZero returns 7 chunks (scale/shift/gate for msa and mlp, plus gate_ctx), each already
    # broadcast to (B,1,1,1,C). Saved separately so a modulation bug is attributable to a chunk.
    for name, chunk in zip(
        ("scale_msa", "shift_msa", "gate_msa", "scale_mlp", "shift_mlp", "gate_mlp", "gate_ctx"),
        decoder.shared_adaln(t_emb),
    ):
        record(f"stage5.mod.{name}", chunk)

    combined = decoder._context_and_x_for_diff_step(context, noise)
    record("stage5.context_and_x", combined)

    x_half = combined[..., decoder.context_channels :]
    modulation = decoder.shared_adaln(t_emb)
    for depth, block in enumerate(decoder.diff_blocks):
        x_half.copy_(block.forward_combined(combined, modulation))
        record(f"diff.block{depth}", x_half)

    from ltx_core.model.video_vae.ops import unpatchify

    normed = decoder.norm_out(x_half)
    record("stage5.norm_out", normed)
    projected = decoder.conv_out(normed)
    record("stage5.conv_out", projected)
    pixels = unpatchify(projected.permute(0, 4, 1, 2, 3).contiguous(), patch_size_hw=decoder.patch_size, patch_size_t=1)
    record("output.pixels", pixels)
    print(f"  pixels: {tuple(pixels.shape)}  range [{pixels.min():.3f}, {pixels.max():.3f}]", flush=True)

    # model_output_type is "x0" on this checkpoint, so the single-step prediction *is* the
    # image. Assert rather than branch: a "v" checkpoint would need an Euler step here and
    # should fail loudly instead of yielding quietly wrong targets.
    assert decoder.model_output_type == "x0", f"unhandled model_output_type={decoder.model_output_type}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("latent", help="path to a latent .pt, or randn:BxCxFxHxW to draw one")
    parser.add_argument("--latent-frames", type=int, default=None)
    parser.add_argument("--crop", type=int, default=None, help="centre-crop latent to NxN (keeps captures small)")
    parser.add_argument(
        "--crop-hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="centre-crop latent to HxW in latent units; each unit is 32 pixels, so 16 24 is 512x768",
    )
    parser.add_argument("--out", type=Path, default=Path("stages/stages.safetensors"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pixels-only",
        action="store_true",
        help="keep only latent/noise/pixels; for end-to-end numbers at resolutions where the "
        "intermediates would be tens of GB",
    )
    args = parser.parse_args()

    if args.latent.startswith("randn:"):
        shape = tuple(int(d) for d in args.latent.removeprefix("randn:").split("x"))
        # Seeded off --seed but offset, so the latent and stage-5's noise are independent draws.
        latent = torch.randn(shape, generator=torch.Generator().manual_seed(args.seed + 1))
    else:
        latent = torch.load(Path(args.latent), map_location="cpu")
    if args.latent_frames is not None:
        latent = latent[:, :, : args.latent_frames]
    crop_hw = tuple(args.crop_hw) if args.crop_hw else ((args.crop, args.crop) if args.crop else None)
    if crop_hw is not None:
        crop_h, crop_w = crop_hw
        _, _, _, lh, lw = latent.shape
        top, left = (lh - crop_h) // 2, (lw - crop_w) // 2
        latent = latent[:, :, :, top : top + crop_h, left : left + crop_w]
    print(f"latent {tuple(latent.shape)}  (pixels {latent.shape[3] * 32}x{latent.shape[4] * 32})")

    decoder = load()
    with torch.no_grad():
        captured = capture(decoder, latent.float(), seed=args.seed, pixels_only=args.pixels_only)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in captured.items()}, str(args.out))
    total = sum(v.numel() * v.element_size() for v in captured.values())
    print(f"\nwrote {len(captured)} tensors ({total / 1e9:.2f} GB) to {args.out}")
    for name, value in captured.items():
        print(f"  {name:26s} {tuple(value.shape)}")


if __name__ == "__main__":
    main()
