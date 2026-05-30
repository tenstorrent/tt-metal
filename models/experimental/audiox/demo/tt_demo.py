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
import time
from pathlib import Path

import torch
import torchaudio
import ttnn

from models.common.auto_compose import to_torch_auto_compose
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


def _build_tt_dit(cpu_state_dict: dict, device):
    try:
        return TtDiffusionTransformer(
            mesh_device=device,
            state_dict=cpu_state_dict,
            depth=_HF_CONFIG["depth"],
            num_heads=_HF_CONFIG["num_heads"],
            io_channels=_HF_CONFIG["io_channels"],
            embed_dim=_HF_CONFIG["embed_dim"],
            lazy_layers=False,
        )
    except RuntimeError:
        return TtDiffusionTransformer(
            mesh_device=device,
            state_dict=cpu_state_dict,
            depth=_HF_CONFIG["depth"],
            num_heads=_HF_CONFIG["num_heads"],
            io_channels=_HF_CONFIG["io_channels"],
            embed_dim=_HF_CONFIG["embed_dim"],
            lazy_layers=True,
        )


def _should_use_local_mesh() -> bool:
    try:
        return ttnn.get_num_devices() > ttnn.get_num_pcie_devices()
    except Exception:
        return False


def _local_mesh_physical_device_ids(device_id: int) -> list[int]:
    num_devices = ttnn.get_num_devices()
    num_pcie_devices = ttnn.get_num_pcie_devices()
    num_local_devices = max(1, num_devices // max(1, num_pcie_devices))
    start = (device_id // max(1, num_local_devices)) * num_local_devices
    return list(range(start, min(start + num_local_devices, num_devices)))


def _local_mesh_width(device_id: int) -> int:
    available = len(_local_mesh_physical_device_ids(device_id))
    requested = int(os.getenv("AUDIOX_TT_LOCAL_MESH_WIDTH", "1"))
    return max(1, min(requested, available))


def open_tt_device(device_id: int):
    os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    if os.getenv("AUDIOX_TT_OPEN_MODE", "mesh") == "direct":
        return ttnn.open_device(device_id=device_id)
    if _should_use_local_mesh():
        dispatch_core_config = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
        physical_device_ids = _local_mesh_physical_device_ids(device_id)
        mesh_width = _local_mesh_width(device_id)
        physical_device_ids = physical_device_ids[:mesh_width]
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, len(physical_device_ids)),
                physical_device_ids=physical_device_ids,
                dispatch_core_config=dispatch_core_config,
            )
        except RuntimeError:
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


def _tt_to_torch(tensor: ttnn.Tensor, device) -> torch.Tensor:
    if isinstance(device, ttnn.MeshDevice):
        if device.get_num_devices() == 1:
            return ttnn.to_torch(tensor)
        return to_torch_auto_compose(tensor, device=device)
    return ttnn.to_torch(tensor)


def close_tt_device(device) -> None:
    if isinstance(device, ttnn.MeshDevice):
        ttnn.close_mesh_device(device)
    else:
        ttnn.close_device(device)


def _synchronize_tt_device(device) -> None:
    ttnn.synchronize_device(device)


def _deallocate_tt(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None:
            ttnn.deallocate(tensor, force=True)


def _nct_to_nhwc(x_nct: ttnn.Tensor) -> ttnn.Tensor:
    """[B, C, T] -> [B, T, 1, C] for the Oobleck decoder."""
    x_btc = ttnn.transpose(x_nct, 1, 2)  # [B, T, C]
    return ttnn.unsqueeze(x_btc, 2)  # [B, T, 1, C]


class TtAudioXSession:
    def __init__(self, checkpoint: Path, device, seed: int | None = None):
        self.device = device
        self.setup_timings = {}
        self.audio_pretransform = None
        self._conditioning_cache_key = None
        self._conditioning_cache_value = None

        if seed is not None:
            torch.manual_seed(seed)

        started_at = time.perf_counter()
        raw_sd = load_audiox_checkpoint(checkpoint)
        self.dit_sd = remap_dit_state_dict(raw_sd)
        self.encoder_sd = remap_oobleck_encoder_state_dict(raw_sd, n_blocks=len(_HF_CONFIG["decoder_c_mults"]))
        decoder_sd = remap_oobleck_decoder_state_dict(raw_sd, n_blocks=len(_HF_CONFIG["decoder_c_mults"]))
        self.setup_timings["checkpoint_seconds"] = time.perf_counter() - started_at

        started_at = time.perf_counter()
        self.multi = _build_conditioners(audio_pretransform=None)
        for cid in ("text_prompt", "video_prompt", "audio_prompt"):
            cond_sd = remap_conditioner_state_dict(raw_sd, conditioner_id=cid)
            load_into(self.multi.conditioners[cid], cond_sd, label=f"cond:{cid}")

        cpu_dit = _build_dit().eval()
        load_into(cpu_dit, self.dit_sd, label="dit")
        self.tt_dit_state_dict = cpu_dit.state_dict()
        self.tt_dit = _build_tt_dit(self.tt_dit_state_dict, device)

        cpu_decoder = _build_decoder().eval()
        load_into(cpu_decoder, decoder_sd, label="decoder")
        self.tt_decoder = TtOobleckDecoder(
            mesh_device=device,
            state_dict=cpu_decoder.state_dict(),
            out_channels=_HF_CONFIG["decoder_out_channels"],
            channels=_HF_CONFIG["decoder_channels"],
            latent_dim=_HF_CONFIG["decoder_latent_dim"],
            c_mults=_HF_CONFIG["decoder_c_mults"],
            strides=_HF_CONFIG["decoder_strides"],
        )
        _synchronize_tt_device(device)
        self.setup_timings["tt_module_build_seconds"] = time.perf_counter() - started_at

    def _ensure_tt_dit(self) -> None:
        if self.tt_dit is None:
            self.tt_dit = _build_tt_dit(self.tt_dit_state_dict, self.device)
            _synchronize_tt_device(self.device)

    def _conditioning_key(
        self,
        *,
        prompt: str,
        video_path: Path | None,
        image_path: Path | None,
        audio_path: Path | None,
        video_prompt_tensor: torch.Tensor | None,
        audio_prompt_tensor: torch.Tensor | None,
    ) -> tuple:
        return (
            prompt,
            None if video_path is None else str(video_path),
            None if image_path is None else str(image_path),
            None if audio_path is None else str(audio_path),
            None if video_prompt_tensor is None else (id(video_prompt_tensor), tuple(video_prompt_tensor.shape)),
            None if audio_prompt_tensor is None else (id(audio_prompt_tensor), tuple(audio_prompt_tensor.shape)),
        )

    def run(
        self,
        prompt: str,
        output: Path,
        *,
        video_path: Path | None = None,
        image_path: Path | None = None,
        audio_path: Path | None = None,
        video_prompt_tensor: torch.Tensor | None = None,
        audio_prompt_tensor: torch.Tensor | None = None,
        steps: int = 100,
        seed: int = 0,
        return_details: bool = False,
    ) -> Path | dict:
        if video_prompt_tensor is not None and (video_path is not None or image_path is not None):
            raise ValueError("pass either a visual path or video_prompt_tensor, not both")
        if audio_prompt_tensor is not None and audio_path is not None:
            raise ValueError("pass either an audio path or audio_prompt_tensor, not both")

        timings = dict(self.setup_timings)

        started_at = time.perf_counter()
        conditioning_key = self._conditioning_key(
            prompt=prompt,
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
            video_prompt_tensor=video_prompt_tensor,
            audio_prompt_tensor=audio_prompt_tensor,
        )
        if conditioning_key == self._conditioning_cache_key and self._conditioning_cache_value is not None:
            cross_attn_cond_torch = self._conditioning_cache_value
        else:
            audio_prompt = _resolve_audio_prompt(audio_path, audio_prompt_tensor)
            if audio_prompt is not None and self.audio_pretransform is None:
                self.audio_pretransform = _build_audio_pretransform()
                load_into(self.audio_pretransform, self.encoder_sd, label="encoder")
                self.multi.conditioners["audio_prompt"].pretransform = self.audio_pretransform
            visual_prompt = (
                video_prompt_tensor if video_prompt_tensor is not None else _load_visual_prompt(video_path, image_path)
            )
            cond_out = self.multi(
                _build_metadata_batch_with_inputs(prompt=prompt, video_prompt=visual_prompt, audio_prompt=audio_prompt),
                "cpu",
            )
            cross_attn_cond_torch = _make_cross_attn_cond(cond_out)
            self._conditioning_cache_key = conditioning_key
            self._conditioning_cache_value = cross_attn_cond_torch
        timings["conditioning_seconds"] = time.perf_counter() - started_at

        started_at = time.perf_counter()
        samples = _HF_CONFIG["sample_rate"] * _HF_CONFIG["duration_seconds"]
        t_latent = -(-samples // _HF_CONFIG["downsample"])
        noise = torch.randn(1, _HF_CONFIG["io_channels"], t_latent)

        tt_noise = _to_tt(noise, self.device)
        tt_cond = _to_tt(cross_attn_cond_torch, self.device)
        _synchronize_tt_device(self.device)
        timings["tt_input_setup_seconds"] = time.perf_counter() - started_at

        self._ensure_tt_dit()

        def model_fn(x, t, **kwargs):
            return self.tt_dit(x, t, cross_attn_cond=tt_cond)

        started_at = time.perf_counter()
        tt_latent = tt_sample_rf(model_fn, tt_noise, mesh_device=self.device, steps=steps)
        _synchronize_tt_device(self.device)
        timings["sampling_seconds"] = time.perf_counter() - started_at

        started_at = time.perf_counter()
        tt_latent_torch = _tt_to_torch(tt_latent, self.device).detach().cpu()
        _deallocate_tt(tt_noise, tt_cond)
        if hasattr(self.tt_dit, "deallocate"):
            self.tt_dit.deallocate()
        self.tt_dit = None
        timings["latent_copy_seconds"] = time.perf_counter() - started_at

        started_at = time.perf_counter()
        tt_audio_nhwc = self.tt_decoder(_nct_to_nhwc(tt_latent))
        _synchronize_tt_device(self.device)
        timings["decode_seconds"] = time.perf_counter() - started_at
        _deallocate_tt(tt_latent)

        timings["generation_seconds"] = (
            timings["conditioning_seconds"]
            + timings["tt_input_setup_seconds"]
            + timings["sampling_seconds"]
            + timings["latent_copy_seconds"]
            + timings["decode_seconds"]
        )

        started_at = time.perf_counter()
        audio = _tt_to_torch(tt_audio_nhwc, self.device).squeeze(2).transpose(1, 2).clamp(-1.0, 1.0).detach().float().cpu()
        _deallocate_tt(tt_audio_nhwc)
        audio = resample_output_audio(
            audio,
            input_sample_rate=_HF_CONFIG["sample_rate"],
            output_sample_rate=_HF_CONFIG["output_sample_rate"],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output), audio[0], _HF_CONFIG["output_sample_rate"])
        timings["save_seconds"] = time.perf_counter() - started_at
        if return_details:
            return {
                "output_path": output,
                "latent": tt_latent_torch,
                "cross_attn_cond": cross_attn_cond_torch.detach().cpu(),
                "conditioning_tokens": int(cross_attn_cond_torch.shape[1]),
                "t_latent": int(t_latent),
                "timings": timings,
            }
        return output


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
    session = TtAudioXSession(checkpoint=checkpoint, device=device, seed=seed)
    return session.run(
        prompt=prompt,
        output=output,
        video_path=video_path,
        image_path=image_path,
        audio_path=audio_path,
        video_prompt_tensor=video_prompt_tensor,
        audio_prompt_tensor=audio_prompt_tensor,
        steps=steps,
        seed=seed,
        return_details=return_details,
    )


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
        close_tt_device(device)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
