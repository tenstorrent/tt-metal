# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end text/video/image-to-audio demo for the AudioX bringup, running on TT.

Same generation flow as ``demo.demo`` but the DiT denoiser and Oobleck VAE
decoder run on a Tenstorrent device through the TTNN ports. The conditioner
stack stays on CPU since it runs once per generation and is <0.1% of total
compute (porting it to TTNN wouldn't move the needle).

Run from the tt-metal root with a connected device:

    python -m models.experimental.audiox.demo.tt_demo \
        --checkpoint /path/to/audiox.safetensors \
        --prompt "a soft piano loop" \
        --output /tmp/out.wav
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
import ttnn

from models.experimental.audiox.demo.demo import (
    _HF_CONFIG,
    _build_audio_pretransform,
    _build_conditioners,
    _build_decoder,
    _build_dit,
    _build_metadata_batch_with_inputs,
    _load_visual_prompt,
    _make_cross_attn_cond,
    _resolve_audio_prompt,
)
from models.experimental.audiox.demo.media import resample_output_audio
from models.experimental.audiox.tt.dit import TtDiffusionTransformer
from models.experimental.audiox.tt.oobleck import TtOobleckDecoder
from models.experimental.audiox.tt.sampler import sample_rf as tt_sample_rf
from models.experimental.audiox.utils.loader import (
    load_audiox_checkpoint,
    load_into,
    remap_conditioner_state_dict,
    remap_dit_state_dict,
    remap_oobleck_encoder_state_dict,
    remap_oobleck_decoder_state_dict,
)


def _should_use_local_mesh() -> bool:
    try:
        return ttnn.get_num_devices() > ttnn.get_num_pcie_devices()
    except Exception:
        return False


def open_tt_device(device_id: int):
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    if _should_use_local_mesh():
        dispatch_core_config = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 1),
                physical_device_ids=[device_id],
                dispatch_core_config=dispatch_core_config,
            )
        except RuntimeError:
            pass
    return ttnn.open_device(device_id=device_id)


def _to_tt(t: torch.Tensor, device, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device)


def _nct_to_nhwc(x_nct: ttnn.Tensor) -> ttnn.Tensor:
    """[B, C, T] -> [B, T, 1, C] for the Oobleck decoder."""
    x_btc = ttnn.transpose(x_nct, 1, 2)  # [B, T, C]
    return ttnn.unsqueeze(x_btc, 2)  # [B, T, 1, C]


def run_tt_demo(
    checkpoint: Path,
    prompt: str,
    output: Path,
    device,
    video_path: Path | None = None,
    image_path: Path | None = None,
    audio_path: Path | None = None,
    video_prompt_tensor: torch.Tensor | None = None,
    audio_prompt_tensor: torch.Tensor | None = None,
    steps: int = 100,
    seed: int = 0,
    return_details: bool = False,
) -> Path | dict:
    """Generate one stereo audio clip on TT and save to ``output``.

    Conditioners run on CPU; DiT + decoder run on ``device``."""
    torch.manual_seed(seed)
    if video_prompt_tensor is not None and (video_path is not None or image_path is not None):
        raise ValueError("pass either a visual path or video_prompt_tensor, not both")
    if audio_prompt_tensor is not None and audio_path is not None:
        raise ValueError("pass either an audio path or audio_prompt_tensor, not both")

    # 1. Load checkpoint and split per module.
    raw_sd = load_audiox_checkpoint(checkpoint)
    dit_sd = remap_dit_state_dict(raw_sd)
    encoder_sd = remap_oobleck_encoder_state_dict(raw_sd, n_blocks=len(_HF_CONFIG["decoder_c_mults"]))
    decoder_sd = remap_oobleck_decoder_state_dict(raw_sd, n_blocks=len(_HF_CONFIG["decoder_c_mults"]))

    # 2. Conditioners on CPU. Build CPU reference modules and load weights.
    audio_prompt = _resolve_audio_prompt(audio_path, audio_prompt_tensor)
    audio_pretransform = _build_audio_pretransform() if audio_prompt is not None else None
    if audio_pretransform is not None:
        load_into(audio_pretransform, encoder_sd, label="encoder")

    multi = _build_conditioners(audio_pretransform=audio_pretransform)
    for cid in ("text_prompt", "video_prompt", "audio_prompt"):
        cond_sd = remap_conditioner_state_dict(raw_sd, conditioner_id=cid)
        load_into(multi.conditioners[cid], cond_sd, label=f"cond:{cid}")

    visual_prompt = video_prompt_tensor if video_prompt_tensor is not None else _load_visual_prompt(video_path, image_path)
    cond_out = multi(
        _build_metadata_batch_with_inputs(prompt=prompt, video_prompt=visual_prompt, audio_prompt=audio_prompt),
        "cpu",
    )
    cross_attn_cond_torch = _make_cross_attn_cond(cond_out)

    # 3. Build TT modules. We seed them from CPU reference state_dicts so the
    # TT modules pick up pretrained weights directly.
    cpu_dit = _build_dit().eval()
    load_into(cpu_dit, dit_sd, label="dit")
    tt_dit = TtDiffusionTransformer(
        mesh_device=device,
        state_dict=cpu_dit.state_dict(),
        depth=_HF_CONFIG["depth"],
        num_heads=_HF_CONFIG["num_heads"],
        io_channels=_HF_CONFIG["io_channels"],
        embed_dim=_HF_CONFIG["embed_dim"],
    )

    cpu_decoder = _build_decoder().eval()
    load_into(cpu_decoder, decoder_sd, label="decoder")
    tt_decoder = TtOobleckDecoder(
        mesh_device=device,
        state_dict=cpu_decoder.state_dict(),
        out_channels=_HF_CONFIG["decoder_out_channels"],
        channels=_HF_CONFIG["decoder_channels"],
        latent_dim=_HF_CONFIG["decoder_latent_dim"],
        c_mults=_HF_CONFIG["decoder_c_mults"],
        strides=_HF_CONFIG["decoder_strides"],
    )

    # 4. Set up noise + cond on device.
    samples = _HF_CONFIG["sample_rate"] * _HF_CONFIG["duration_seconds"]
    t_latent = -(-samples // _HF_CONFIG["downsample"])
    noise = torch.randn(1, _HF_CONFIG["io_channels"], t_latent)

    tt_noise = _to_tt(noise, device)
    tt_cond = _to_tt(cross_attn_cond_torch, device)

    # 5. Sample on device. The sampler calls tt_dit each step with the current
    # latent + a [batch] timestep tensor.
    def model_fn(x, t, **kwargs):
        return tt_dit(x, t, cross_attn_cond=tt_cond)

    tt_latent = tt_sample_rf(model_fn, tt_noise, mesh_device=device, steps=steps)
    tt_latent_torch = ttnn.to_torch(tt_latent).detach().cpu()

    # 6. Decode on device. tt_decoder expects [B, T, 1, C].
    tt_audio_nhwc = tt_decoder(_nct_to_nhwc(tt_latent))

    # 7. Pull audio back to CPU, drop the H=1 dim, transpose [B, T, 1, C] -> [B, C, T].
    audio = ttnn.to_torch(tt_audio_nhwc).squeeze(2).transpose(1, 2).clamp(-1.0, 1.0).detach().float().cpu()
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
            "latent": tt_latent_torch,
            "cross_attn_cond": cross_attn_cond_torch.detach().cpu(),
            "conditioning_tokens": int(cross_attn_cond_torch.shape[1]),
            "t_latent": int(t_latent),
        }
    return output


def _parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AudioX text/video/image-to-audio demo (TT path)")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to AudioX .safetensors")
    p.add_argument("--prompt", type=str, default="", help="Text prompt to condition on")
    p.add_argument("--output", type=Path, default=Path("audiox_tt_out.wav"))
    visual = p.add_mutually_exclusive_group()
    visual.add_argument("--video", type=Path, help="Optional video prompt for video-to-audio or video-to-music")
    visual.add_argument("--image", type=Path, help="Optional image prompt, repeated across the visual timeline")
    p.add_argument("--audio", type=Path, help="Optional audio prompt for audio-conditioned generation")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.prompt and args.video is None and args.image is None and args.audio is None:
        p.error("at least one of --prompt, --video, --image, or --audio is required")
    return args


def main(argv: list = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    device = open_tt_device(device_id=0)
    try:
        out = run_tt_demo(
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            output=args.output,
            device=device,
            video_path=args.video,
            image_path=args.image,
            audio_path=args.audio,
            steps=args.steps,
            seed=args.seed,
        )
    finally:
        ttnn.close_device(device)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
