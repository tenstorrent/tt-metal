# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end text-to-audio demo for the AudioX bringup.

Loads an AudioX (HKUSTAudio/AudioX) checkpoint, runs the CPU reference
conditioner stack to build the cross-attn context, runs the DiT denoiser
through ``sample_rf`` (rectified flow, discrete Euler), and decodes the
resulting latent through the Oobleck VAE decoder into a stereo waveform.

For text-to-audio the video and audio prompts are empty, so CLIP and
AudioAutoencoder both take the ``empty_*_feat`` fast path and only T5 does
real work on the conditioning side.

Run from the tt-metal root:

    python -m models.experimental.audiox.demo.demo \
        --checkpoint /path/to/audiox.safetensors \
        --prompt "a soft piano loop" \
        --output /tmp/out.wav
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

from models.experimental.audiox.demo.media import (
    load_audio_prompt,
    load_image_prompt,
    load_video_prompt,
    resample_output_audio,
)
from models.experimental.audiox.reference.conditioners import (
    AudioAutoencoderConditioner,
    CLIPConditioner,
    MultiConditioner,
    T5Conditioner,
)
from models.experimental.audiox.reference.dit import DiffusionTransformer
from models.experimental.audiox.reference.oobleck import OobleckDecoder, OobleckEncoder
from models.experimental.audiox.reference.sampler import sample_rf
from models.experimental.audiox.utils.loader import (
    load_audiox_checkpoint,
    load_into,
    remap_conditioner_state_dict,
    remap_dit_state_dict,
    remap_oobleck_encoder_state_dict,
    remap_oobleck_decoder_state_dict,
)


# AudioX HF config snapshots — copied from HKUSTAudio/AudioX so the demo runs
# without parsing the upstream JSON config. If/when those values change in a
# future release we'll re-derive them from the config rather than guessing.
_HF_CONFIG = {
    "sample_rate": 44100,
    "output_sample_rate": 16000,
    "downsample": 2048,  # prod(decoder strides)
    "duration_seconds": 10,
    "io_channels": 64,
    "embed_dim": 1536,
    "depth": 24,
    "num_heads": 24,
    "cond_token_dim": 768,
    "decoder_channels": 128,
    "decoder_latent_dim": 64,
    "decoder_c_mults": (1, 2, 4, 8, 16),
    "decoder_strides": (2, 4, 4, 8, 8),
    "decoder_out_channels": 2,
}


def _resolve_duration_seconds(duration_seconds: int | None) -> int:
    return _HF_CONFIG["duration_seconds"] if duration_seconds is None else int(duration_seconds)


class _ZeroPretransform(torch.nn.Module):
    """Stand-in for the Oobleck VAE encoder used by the audio-prompt
    conditioner. The text-to-audio path always feeds zero audio so the
    encoder never runs — we just need ``encoded_channels`` for the
    AudioAutoencoderConditioner constructor."""

    def __init__(self, channels: int):
        super().__init__()
        self.encoded_channels = channels

    def encode(self, _audio):  # pragma: no cover — text-to-audio never hits this
        raise RuntimeError("text-to-audio demo never calls the audio encoder")


def _build_audio_pretransform() -> OobleckEncoder:
    return OobleckEncoder(
        in_channels=_HF_CONFIG["decoder_out_channels"],
        channels=_HF_CONFIG["decoder_channels"],
        latent_dim=_HF_CONFIG["decoder_latent_dim"],
        c_mults=_HF_CONFIG["decoder_c_mults"],
        strides=_HF_CONFIG["decoder_strides"],
    )


def _build_conditioners(audio_pretransform: torch.nn.Module | None = None) -> MultiConditioner:
    text = T5Conditioner(output_dim=_HF_CONFIG["cond_token_dim"])
    video = CLIPConditioner(output_dim=_HF_CONFIG["cond_token_dim"])
    audio = AudioAutoencoderConditioner(
        pretransform=audio_pretransform or _ZeroPretransform(channels=_HF_CONFIG["decoder_latent_dim"]),
        output_dim=_HF_CONFIG["cond_token_dim"],
    )
    return MultiConditioner({"text_prompt": text, "video_prompt": video, "audio_prompt": audio})


def _build_dit() -> DiffusionTransformer:
    return DiffusionTransformer(
        io_channels=_HF_CONFIG["io_channels"],
        embed_dim=_HF_CONFIG["embed_dim"],
        depth=_HF_CONFIG["depth"],
        num_heads=_HF_CONFIG["num_heads"],
        cond_token_dim=_HF_CONFIG["cond_token_dim"],
    )


def _build_decoder() -> OobleckDecoder:
    return OobleckDecoder(
        out_channels=_HF_CONFIG["decoder_out_channels"],
        channels=_HF_CONFIG["decoder_channels"],
        latent_dim=_HF_CONFIG["decoder_latent_dim"],
        c_mults=_HF_CONFIG["decoder_c_mults"],
        strides=_HF_CONFIG["decoder_strides"],
    )


def _empty_video_for_text_only(batch: int, *, duration_seconds: int | None = None) -> torch.Tensor:
    """All-zero video that triggers the CLIPConditioner empty-feat shortcut."""
    fps = 5
    duration = _resolve_duration_seconds(duration_seconds)
    return torch.zeros(batch, fps * duration, 3, 224, 224)


def _empty_audio_for_text_only(batch: int, *, duration_seconds: int | None = None) -> torch.Tensor:
    """All-zero audio that triggers the AudioAutoencoderConditioner shortcut."""
    duration = _resolve_duration_seconds(duration_seconds)
    samples = _HF_CONFIG["sample_rate"] * duration
    return torch.zeros(batch, 2, samples)


def _build_metadata_batch(prompt: str) -> list:
    return _build_metadata_batch_with_inputs(prompt=prompt)


def _build_metadata_batch_with_inputs(
    prompt: str,
    *,
    video_prompt: torch.Tensor | None = None,
    audio_prompt: torch.Tensor | None = None,
    duration_seconds: int | None = None,
) -> list:
    return [
        {
            "text_prompt": prompt,
            "video_prompt": video_prompt
            if video_prompt is not None
            else _empty_video_for_text_only(1, duration_seconds=duration_seconds),
            "audio_prompt": audio_prompt
            if audio_prompt is not None
            else _empty_audio_for_text_only(1, duration_seconds=duration_seconds),
        }
    ]


def _load_visual_prompt(
    video_path: Path | None,
    image_path: Path | None,
    *,
    duration_seconds: int | None = None,
) -> torch.Tensor | None:
    if video_path is not None and image_path is not None:
        raise ValueError("pass at most one of --video or --image")
    if video_path is None and image_path is None:
        return None

    target_frames = 5 * _resolve_duration_seconds(duration_seconds)
    if video_path is not None:
        return load_video_prompt(video_path, target_frames=target_frames)
    return load_image_prompt(image_path, target_frames=target_frames)


def _load_audio_prompt(audio_path: Path | None, *, duration_seconds: int | None = None) -> torch.Tensor | None:
    if audio_path is None:
        return None
    duration = _resolve_duration_seconds(duration_seconds)
    target_samples = _HF_CONFIG["sample_rate"] * duration
    return load_audio_prompt(
        audio_path,
        target_sample_rate=_HF_CONFIG["sample_rate"],
        target_samples=target_samples,
        target_channels=_HF_CONFIG["decoder_out_channels"],
    )


def _resolve_audio_prompt(
    audio_path: Path | None,
    audio_prompt_tensor: torch.Tensor | None,
    *,
    duration_seconds: int | None = None,
) -> torch.Tensor | None:
    if audio_prompt_tensor is not None:
        return audio_prompt_tensor
    return _load_audio_prompt(audio_path, duration_seconds=duration_seconds)


def _make_cross_attn_cond(multi_out: dict) -> torch.Tensor:
    """Concatenate per-conditioner embeds along the sequence dim, in the
    order AudioX uses: ``video_prompt``, ``text_prompt``, ``audio_prompt``."""
    return torch.cat(
        [multi_out["video_prompt"][0], multi_out["text_prompt"][0], multi_out["audio_prompt"][0]],
        dim=1,
    )


def run_demo(
    checkpoint: Path,
    prompt: str,
    output: Path,
    video_path: Path | None = None,
    image_path: Path | None = None,
    audio_path: Path | None = None,
    video_prompt_tensor: torch.Tensor | None = None,
    audio_prompt_tensor: torch.Tensor | None = None,
    steps: int = 100,
    seed: int = 0,
    device: str = "cpu",
    duration_seconds: int | None = None,
    return_details: bool = False,
) -> Path | dict:
    """Generate one stereo audio clip for ``prompt`` and save to ``output``.
    Returns the output path on success."""
    duration_seconds = _resolve_duration_seconds(duration_seconds)
    torch.manual_seed(seed)
    if video_prompt_tensor is not None and (video_path is not None or image_path is not None):
        raise ValueError("pass either a visual path or video_prompt_tensor, not both")
    if audio_prompt_tensor is not None and audio_path is not None:
        raise ValueError("pass either an audio path or audio_prompt_tensor, not both")

    # Load checkpoint and split into per-module state dicts.
    raw_sd = load_audiox_checkpoint(checkpoint)
    dit_sd = remap_dit_state_dict(raw_sd)
    encoder_sd = remap_oobleck_encoder_state_dict(raw_sd, n_blocks=len(_HF_CONFIG["decoder_c_mults"]))
    decoder_sd = remap_oobleck_decoder_state_dict(raw_sd, n_blocks=len(_HF_CONFIG["decoder_c_mults"]))

    # Build modules and load weights. Conditioners get only the matching id;
    # T5/CLIP HF visual encoder weights live outside state_dict, so missing
    # keys for them are expected.
    audio_prompt = _resolve_audio_prompt(audio_path, audio_prompt_tensor, duration_seconds=duration_seconds)
    audio_pretransform = _build_audio_pretransform() if audio_prompt is not None else None
    if audio_pretransform is not None:
        load_into(audio_pretransform, encoder_sd, label="encoder")

    multi = _build_conditioners(audio_pretransform=audio_pretransform)
    for cid in ("text_prompt", "video_prompt", "audio_prompt"):
        cond_sd = remap_conditioner_state_dict(raw_sd, conditioner_id=cid)
        load_into(multi.conditioners[cid], cond_sd, label=f"cond:{cid}")

    dit = _build_dit().eval()
    load_into(dit, dit_sd, label="dit")

    decoder = _build_decoder().eval()
    load_into(decoder, decoder_sd, label="decoder")

    multi = multi.to(device)
    dit = dit.to(device)
    decoder = decoder.to(device)

    # Build cross-attn context once per generation.
    visual_prompt = (
        video_prompt_tensor
        if video_prompt_tensor is not None
        else _load_visual_prompt(video_path, image_path, duration_seconds=duration_seconds)
    )
    cond_out = multi(
        _build_metadata_batch_with_inputs(
            prompt=prompt,
            video_prompt=visual_prompt,
            audio_prompt=audio_prompt,
            duration_seconds=duration_seconds,
        ),
        device,
    )
    cross_attn_cond = _make_cross_attn_cond(cond_out)

    samples = _HF_CONFIG["sample_rate"] * duration_seconds
    t_latent = -(-samples // _HF_CONFIG["downsample"])  # ceil division
    noise = torch.randn(1, _HF_CONFIG["io_channels"], t_latent, device=device)

    def model_fn(x, t):
        return dit(x, t, cross_attn_cond=cross_attn_cond)

    latent = sample_rf(model_fn, noise, steps=steps)

    # Decode latent -> stereo audio. Decoder upsamples by `downsample`.
    audio = decoder(latent).clamp(-1.0, 1.0).detach().cpu()
    audio = resample_output_audio(
        audio,
        input_sample_rate=_HF_CONFIG["sample_rate"],
        output_sample_rate=_HF_CONFIG["output_sample_rate"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), audio[0], _HF_CONFIG["output_sample_rate"])
    if return_details:
        return {
            "output_path": output,
            "latent": latent.detach().cpu(),
            "cross_attn_cond": cross_attn_cond.detach().cpu(),
            "conditioning_tokens": int(cross_attn_cond.shape[1]),
            "t_latent": int(t_latent),
            "duration_seconds": int(duration_seconds),
        }
    return output


def _parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AudioX text/video/image-to-audio demo (CPU reference path)")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to AudioX .safetensors")
    p.add_argument("--prompt", type=str, default="", help="Text prompt to condition on")
    p.add_argument("--output", type=Path, default=Path("audiox_out.wav"), help="Output WAV path")
    visual = p.add_mutually_exclusive_group()
    visual.add_argument("--video", type=Path, help="Optional video prompt for video-to-audio or video-to-music")
    visual.add_argument("--image", type=Path, help="Optional image prompt, repeated across the visual timeline")
    p.add_argument("--audio", type=Path, help="Optional audio prompt for audio-conditioned generation")
    p.add_argument("--steps", type=int, default=100, help="Number of diffusion steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--duration-seconds", type=int, default=_HF_CONFIG["duration_seconds"])
    args = p.parse_args(argv)
    if not args.prompt and args.video is None and args.image is None and args.audio is None:
        p.error("at least one of --prompt, --video, --image, or --audio is required")
    return args


def main(argv: list = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    out = run_demo(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        output=args.output,
        video_path=args.video,
        image_path=args.image,
        audio_path=args.audio,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        duration_seconds=args.duration_seconds,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
