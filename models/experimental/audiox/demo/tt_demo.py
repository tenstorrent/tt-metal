# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end text/video/image-to-audio demo for the AudioX bringup, running on TT.

Same generation flow as ``demo.demo`` but the DiT denoiser runs on a
Tenstorrent device through the TTNN port. The decoder defaults to CPU
(``AUDIOX_TT_CPU_DECODE=1``) and can be forced onto TT with
``AUDIOX_TT_CPU_DECODE=0``. The conditioner stack stays on CPU since it runs
once per generation and is <0.1% of total compute.

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
    _resolve_duration_seconds,
    _resolve_audio_prompt,
)
from models.experimental.audiox.demo.media import resample_output_audio, save_output_audio
from models.experimental.audiox.demo.tt_runtime import apply_tt_env_overrides, restore_tt_env, tt_open_kwargs_from_env
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
    open_kwargs = tt_open_kwargs_from_env()
    if os.getenv("AUDIOX_TT_OPEN_MODE", "mesh") == "direct":
        return ttnn.open_device(device_id=device_id, **open_kwargs)
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
                **open_kwargs,
            )
        except RuntimeError:
            try:
                return ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(1, 1),
                    physical_device_ids=[device_id],
                    dispatch_core_config=dispatch_core_config,
                    **open_kwargs,
                )
            except RuntimeError:
                pass
    return ttnn.open_device(device_id=device_id, **open_kwargs)


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


def _should_use_cpu_decode() -> bool:
    return os.environ.get("AUDIOX_TT_CPU_DECODE", "1") == "1"


def _build_cpu_decoder_fused(decoder_sd: dict):
    cpu_decoder = _build_decoder().eval()
    load_into(cpu_decoder, decoder_sd, label="decoder")
    for m in cpu_decoder.modules():
        if hasattr(m, "weight_g"):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except Exception:
                pass
    return cpu_decoder


class TtAudioXSession:
    def __init__(self, checkpoint: Path, device, seed: int | None = None):
        self.device = device
        self.setup_timings = {}
        self.audio_pretransform = None
        self._conditioning_cache_key = None
        self._conditioning_cache_value = None
        self.use_cpu_decode = _should_use_cpu_decode()

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

        if self.use_cpu_decode:
            self.cpu_decoder = _build_cpu_decoder_fused(decoder_sd)
            self.tt_decoder = None
        else:
            self.cpu_decoder = None
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

    def deallocate(self) -> None:
        if self.tt_dit is not None and hasattr(self.tt_dit, "deallocate"):
            self.tt_dit.deallocate()
            self.tt_dit = None
        if self.tt_decoder is not None and hasattr(self.tt_decoder, "deallocate"):
            self.tt_decoder.deallocate()
        self.cpu_decoder = None
        self._conditioning_cache_key = None
        self._conditioning_cache_value = None

    def _conditioning_key(
        self,
        *,
        prompt: str,
        video_path: Path | None,
        image_path: Path | None,
        audio_path: Path | None,
        video_prompt_tensor: torch.Tensor | None,
        audio_prompt_tensor: torch.Tensor | None,
        duration_seconds: int,
    ) -> tuple:
        return (
            prompt,
            None if video_path is None else str(video_path),
            None if image_path is None else str(image_path),
            None if audio_path is None else str(audio_path),
            None if video_prompt_tensor is None else (id(video_prompt_tensor), tuple(video_prompt_tensor.shape)),
            None if audio_prompt_tensor is None else (id(audio_prompt_tensor), tuple(audio_prompt_tensor.shape)),
            int(duration_seconds),
        )

    def prepare_conditioning(
        self,
        *,
        prompt: str,
        video_path: Path | None = None,
        image_path: Path | None = None,
        audio_path: Path | None = None,
        video_prompt_tensor: torch.Tensor | None = None,
        audio_prompt_tensor: torch.Tensor | None = None,
        duration_seconds: int | None = None,
    ) -> torch.Tensor:
        duration_seconds = _resolve_duration_seconds(duration_seconds)
        conditioning_key = self._conditioning_key(
            prompt=prompt,
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
            video_prompt_tensor=video_prompt_tensor,
            audio_prompt_tensor=audio_prompt_tensor,
            duration_seconds=duration_seconds,
        )
        if conditioning_key == self._conditioning_cache_key and self._conditioning_cache_value is not None:
            return self._conditioning_cache_value

        audio_prompt = _resolve_audio_prompt(
            audio_path,
            audio_prompt_tensor,
            duration_seconds=duration_seconds,
        )
        if audio_prompt is not None and self.audio_pretransform is None:
            self.audio_pretransform = _build_audio_pretransform()
            load_into(self.audio_pretransform, self.encoder_sd, label="encoder")
            self.multi.conditioners["audio_prompt"].pretransform = self.audio_pretransform
        visual_prompt = (
            video_prompt_tensor
            if video_prompt_tensor is not None
            else _load_visual_prompt(video_path, image_path, duration_seconds=duration_seconds)
        )
        cond_out = self.multi(
            _build_metadata_batch_with_inputs(
                prompt=prompt,
                video_prompt=visual_prompt,
                audio_prompt=audio_prompt,
                duration_seconds=duration_seconds,
            ),
            "cpu",
        )
        cross_attn_cond_torch = _make_cross_attn_cond(cond_out)
        self._conditioning_cache_key = conditioning_key
        self._conditioning_cache_value = cross_attn_cond_torch
        return cross_attn_cond_torch

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
        cross_attn_cond_torch: torch.Tensor | None = None,
        steps: int = 100,
        seed: int = 0,
        duration_seconds: int | None = None,
        return_details: bool = False,
    ) -> Path | dict:
        duration_seconds = _resolve_duration_seconds(duration_seconds)
        if video_prompt_tensor is not None and (video_path is not None or image_path is not None):
            raise ValueError("pass either a visual path or video_prompt_tensor, not both")
        if audio_prompt_tensor is not None and audio_path is not None:
            raise ValueError("pass either an audio path or audio_prompt_tensor, not both")

        timings = dict(self.setup_timings)

        if cross_attn_cond_torch is None:
            started_at = time.perf_counter()
            cross_attn_cond_torch = self.prepare_conditioning(
                prompt=prompt,
                video_path=video_path,
                image_path=image_path,
                audio_path=audio_path,
                video_prompt_tensor=video_prompt_tensor,
                audio_prompt_tensor=audio_prompt_tensor,
                duration_seconds=duration_seconds,
            )
            timings["conditioning_seconds"] = time.perf_counter() - started_at
        else:
            timings["conditioning_seconds"] = 0.0

        started_at = time.perf_counter()
        samples = _HF_CONFIG["sample_rate"] * duration_seconds
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
        if self.use_cpu_decode and self.cpu_decoder is not None:
            latent_nct = tt_latent_torch.float()  # [B, C, T]
            with torch.no_grad():
                audio_nct = self.cpu_decoder(latent_nct)  # [B, 2, T_audio]
            timings["decode_seconds"] = time.perf_counter() - started_at
            timings["decode_backend"] = "cpu"
            timings["decoder_profile_present"] = False
            _deallocate_tt(tt_latent)

            timings["generation_seconds"] = (
                timings["conditioning_seconds"]
                + timings["tt_input_setup_seconds"]
                + timings["sampling_seconds"]
                + timings["latent_copy_seconds"]
                + timings["decode_seconds"]
            )

            started_at = time.perf_counter()
            audio = audio_nct.clamp(-1.0, 1.0).detach().float().cpu()
        else:
            tt_audio_nhwc = self.tt_decoder(_nct_to_nhwc(tt_latent))
            _synchronize_tt_device(self.device)
            timings["decode_seconds"] = time.perf_counter() - started_at
            timings["decode_backend"] = "tt"
            timings["decoder_profile_present"] = getattr(self.tt_decoder, "last_profile", None) is not None
            if timings["decoder_profile_present"]:
                timings["decoder_profile"] = self.tt_decoder.last_profile
            _deallocate_tt(tt_latent)

            timings["generation_seconds"] = (
                timings["conditioning_seconds"]
                + timings["tt_input_setup_seconds"]
                + timings["sampling_seconds"]
                + timings["latent_copy_seconds"]
                + timings["decode_seconds"]
            )

            started_at = time.perf_counter()
            if isinstance(tt_audio_nhwc, list):
                audio_chunks = []
                for tt_chunk in tt_audio_nhwc:
                    audio_chunks.append(
                        _tt_to_torch(tt_chunk, self.device)
                        .squeeze(2)
                        .transpose(1, 2)
                        .clamp(-1.0, 1.0)
                        .detach()
                        .float()
                        .cpu()
                    )
                    _deallocate_tt(tt_chunk)
                audio = torch.cat(audio_chunks, dim=2)
            else:
                audio = (
                    _tt_to_torch(tt_audio_nhwc, self.device)
                    .squeeze(2)
                    .transpose(1, 2)
                    .clamp(-1.0, 1.0)
                    .detach()
                    .float()
                    .cpu()
                )
                _deallocate_tt(tt_audio_nhwc)

        audio = resample_output_audio(
            audio,
            input_sample_rate=_HF_CONFIG["sample_rate"],
            output_sample_rate=_HF_CONFIG["output_sample_rate"],
        )
        save_output_audio(output, audio[0], sample_rate=_HF_CONFIG["output_sample_rate"])
        timings["save_seconds"] = time.perf_counter() - started_at
        if return_details:
            return {
                "output_path": output,
                "latent": tt_latent_torch,
                "cross_attn_cond": cross_attn_cond_torch.detach().cpu(),
                "conditioning_tokens": int(cross_attn_cond_torch.shape[1]),
                "t_latent": int(t_latent),
                "duration_seconds": int(duration_seconds),
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
    duration_seconds: int | None = None,
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
        duration_seconds=duration_seconds,
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
    p.add_argument("--duration-seconds", type=int, default=_HF_CONFIG["duration_seconds"])
    p.add_argument("--tt-device-id", type=int, default=0)
    p.add_argument("--tt-open-mode", choices=("mesh", "direct"))
    p.add_argument("--tt-local-mesh-width", type=int)
    p.add_argument("--tt-l1-small-size", type=int)
    p.add_argument("--tt-trace-region-size", type=int)
    p.add_argument("--tt-num-command-queues", type=int)
    p.add_argument("--tt-worker-l1-size", type=int)
    p.add_argument("--tt-conv-transpose-input-chunk", type=int)
    p.add_argument("--tt-conv1d-width-slices", type=int)
    p.add_argument("--tt-conv-transpose-height-slices", type=int)
    p.add_argument("--tt-conv-transpose-stride2-act-block-h", type=int)
    p.add_argument("--tt-conv-transpose-stride4-act-block-h", type=int)
    args = p.parse_args(argv)
    if not args.prompt and args.video is None and args.image is None and args.audio is None:
        p.error("at least one of --prompt, --video, --image, or --audio is required")
    return args


def main(argv: list = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    previous_tt_env = apply_tt_env_overrides(
        open_mode=args.tt_open_mode,
        local_mesh_width=args.tt_local_mesh_width,
        l1_small_size=args.tt_l1_small_size,
        trace_region_size=args.tt_trace_region_size,
        num_command_queues=args.tt_num_command_queues,
        worker_l1_size=args.tt_worker_l1_size,
        conv_transpose_input_chunk=args.tt_conv_transpose_input_chunk,
        conv1d_width_slices=args.tt_conv1d_width_slices,
        conv_transpose_height_slices=args.tt_conv_transpose_height_slices,
        conv_transpose_stride2_act_block_h=args.tt_conv_transpose_stride2_act_block_h,
        conv_transpose_stride4_act_block_h=args.tt_conv_transpose_stride4_act_block_h,
    )
    device = open_tt_device(device_id=args.tt_device_id)
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
            duration_seconds=args.duration_seconds,
        )
    finally:
        close_tt_device(device)
        restore_tt_env(previous_tt_env)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
