# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stable-Video-Infinity (SVI) 2.0 Pro pipeline for Wan2.2 I2V.

Chains short I2V clips into long videos with latent-space continuity.
See ``experimental/models/Wan2_2_SVI.md`` for the regime parameters,
upstream-workflow comparison, and the documented scheduler gap.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_lora import LoRASpec, WanPipelineI2VLora
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.solvers import EulerSolver, UniPCSolver, UniPCVariant

Regime = Literal["python", "comfyui"]


_REGIMES = {
    "python": {
        "flow_shift": 5.0,
        "call_defaults": dict(num_inference_steps=50, guidance_scale=5.0, guidance_scale_2=5.0),
    },
    "comfyui": {
        "flow_shift": 8.0,
        "call_defaults": dict(num_inference_steps=6, guidance_scale=1.5, guidance_scale_2=1.5),
    },
}


@dataclass(frozen=True)
class _ClipSpec:
    prompt: str
    seed: int
    guidance_scale: float
    guidance_scale_2: float


class WanPipelineSVI(WanPipelineI2VLora):
    """Wan2.2 I2V with SVI 2.0 Pro LoRA + autoregressive clip-chaining driver."""

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        svi_high: str,
        svi_low: str,
        lightx2v_high: Optional[str] = None,
        lightx2v_low: Optional[str] = None,
        regime: Regime = "python",
        num_motion_latent: int = 1,
        num_overlap_frame: int = 4,
    ) -> None:
        if regime not in _REGIMES:
            raise ValueError(f"regime must be one of {sorted(_REGIMES)}, got {regime!r}")

        for label, path in [("svi_high", svi_high), ("svi_low", svi_low)]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label}: file does not exist: {path}")

        self._regime: Regime = regime
        self._num_motion_latent = int(num_motion_latent)
        self._num_overlap_frame = int(num_overlap_frame)
        # Lives for the duration of one generate_long_video call; cleared in
        # that method's finally block.
        self._anchor_latents_cache: Optional[tuple] = None

        lora_high, lora_low, scheduler = self._configure_regime(
            regime=regime,
            svi_high=svi_high,
            svi_low=svi_low,
            lightx2v_high=lightx2v_high,
            lightx2v_low=lightx2v_low,
        )

        super().__init__(
            device=device,
            config=config,
            lora_high=lora_high,
            lora_low=lora_low,
        )
        # Override the base's scheduler/solver — breaks tracing.
        self._scheduler = scheduler
        if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
            self._solver = EulerSolver()
        else:
            self._solver = UniPCSolver(
                order=scheduler.config.solver_order,
                variant=UniPCVariant(scheduler.config.solver_type),
            )

    @staticmethod
    def _configure_regime(
        *,
        regime: "Regime",
        svi_high: str,
        svi_low: str,
        lightx2v_high: Optional[str],
        lightx2v_low: Optional[str],
    ) -> tuple[list[LoRASpec], list[LoRASpec], object]:
        """Return ``(lora_high, lora_low, scheduler)`` per regime."""
        shift = _REGIMES[regime]["flow_shift"]

        if regime == "python":
            if lightx2v_high is not None or lightx2v_low is not None:
                raise ValueError("lightx2v_* are only valid in regime='comfyui'; drop them for 'python'.")
            # Matches diffsynth's FlowMatchScheduler("Wan") — plain Euler on a
            # flow-matching schedule, dispatched to EulerSolver downstream.
            return (
                [LoRASpec(svi_high, 1.0)],
                [LoRASpec(svi_low, 1.0)],
                FlowMatchEulerDiscreteScheduler(shift=shift),
            )

        # comfyui regime: requires LightX2V LoRAs stacked under SVI.
        if lightx2v_high is None or lightx2v_low is None:
            raise ValueError(
                "regime='comfyui' requires lightx2v_high and lightx2v_low LoRA paths. "
                "See experimental/models/Wan2_2_SVI.md for the recommended HF repo."
            )
        for label, path in [("lightx2v_high", lightx2v_high), ("lightx2v_low", lightx2v_low)]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label}: file does not exist: {path}")
        # Per upstream docs/svi/comfyui.md: LightX2V at 1.0 on the high-noise
        # expert hurts SVI; 0.5 is recommended.
        # ComfyUI workflow uses k-diffusion dpm++_sde; tt-metal has no
        # on-device port, so we substitute UniPC at the same flow_shift.
        # Visually close, not bit-exact.
        return (
            [LoRASpec(lightx2v_high, 0.5), LoRASpec(svi_high, 1.0)],
            [LoRASpec(lightx2v_low, 1.0), LoRASpec(svi_low, 1.0)],
            UniPCMultistepScheduler(
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                flow_shift=shift,
            ),
        )

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
        svi_high: str,
        svi_low: str,
        lightx2v_high: Optional[str] = None,
        lightx2v_low: Optional[str] = None,
        regime: Regime = "python",
        num_motion_latent: int = 1,
        num_overlap_frame: int = 4,
    ) -> WanPipelineSVI:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=WanPipelineI2VLora.BASE_DIFFUSERS_REPO,
            height=height,
            width=width,
            num_frames=num_frames,
            num_links=num_links,
            topology=topology,
            dynamic_load=dynamic_load,
            is_fsdp=is_fsdp,
            boundary_ratio=0.875,
            model_type="i2v",
        )
        return cls(
            device=mesh_device,
            config=config,
            svi_high=svi_high,
            svi_low=svi_low,
            lightx2v_high=lightx2v_high,
            lightx2v_low=lightx2v_low,
            regime=regime,
            num_motion_latent=num_motion_latent,
            num_overlap_frame=num_overlap_frame,
        )

    def __call__(
        self,
        *args,
        anchor_image: Optional[Image.Image] = None,
        prev_last_latent: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._svi_anchor_image = anchor_image
        self._svi_prev_last_latent = prev_last_latent
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self._svi_anchor_image = None
            self._svi_prev_last_latent = None

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype=None,
        device=None,
        anchor_image: Optional[Image.Image] = None,
        prev_last_latent: Optional[torch.Tensor] = None,
    ):
        if anchor_image is None:
            anchor_image = getattr(self, "_svi_anchor_image", None)
        if prev_last_latent is None:
            prev_last_latent = getattr(self, "_svi_prev_last_latent", None)

        # WanPipelineI2V.__init__'s warmup_buffers invokes __call__ with no
        # SVI state — defer to the plain I2V path for that one pass.
        if anchor_image is None and prev_last_latent is None:
            return super().prepare_latents(
                batch_size=batch_size,
                image_prompt=image_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=dtype,
                device=device,
            )

        anchor_pil = _as_anchor_image(anchor_image if anchor_image is not None else image_prompt)

        # Cache the encoded anchor + mask across clips; the encoder is the
        # dominant non-DiT cost per clip (~3-8s on BH 2x4) and the result is
        # invariant under (anchor, H, W, T, dtype).
        cache_key = (id(anchor_pil), height, width, num_frames, num_channels_latents, dtype)
        cached = self._anchor_latents_cache
        if cached is not None and cached[0] == cache_key:
            tt_y_template = cached[1]
            latents, _ = WanPipeline.prepare_latents(
                self,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=dtype,
                device=device,
            )
            tt_y = tt_y_template.clone()
        else:
            latents, tt_y = super().prepare_latents(
                batch_size=batch_size,
                image_prompt=[ImagePrompt(image=anchor_pil, frame_pos=0)],
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=dtype,
                device=device,
            )
            # Cloned because the motion-latent splice below mutates tt_y in place.
            self._anchor_latents_cache = (cache_key, tt_y.clone())

        # tt_y layout: (B, 4 mask channels + 16 image-latent channels, T_lat, H_lat, W_lat).
        t_lat = tt_y.shape[2]
        n = 0
        if prev_last_latent is not None and self._num_motion_latent > 0:
            n = min(self._num_motion_latent, prev_last_latent.shape[2], t_lat - 1)
            if n > 0:
                motion = prev_last_latent[:, :, -n:, :, :].to(tt_y.device, tt_y.dtype)
                if motion.shape[-2:] != tt_y.shape[-2:]:
                    logger.warning(
                        f"prev_last_latent spatial shape {tuple(motion.shape[-2:])} != "
                        f"conditioning {tuple(tt_y.shape[-2:])}; skipping motion splice."
                    )
                    n = 0
                else:
                    tt_y[:, 4:, 1 : 1 + n, :, :] = motion

        if 1 + n < t_lat:
            tt_y[:, 4:, 1 + n :, :, :] = 0.0

        return latents, tt_y

    def generate_long_video(
        self,
        *,
        prompt: Union[str, List[str]],
        num_clips: int,
        anchor_image: Optional[Image.Image] = None,
        image_prompt=None,
        base_seed: int = 0,
        seed_stride: int = 42,
        partial_output_path: Optional[str] = None,
        **call_kwargs,
    ):
        """Run ``num_clips`` chained I2V generations and return the concat.

        ``anchor_image`` is the persistent SVI reference frame. ``image_prompt``
        is accepted for compatibility with the base I2V test harness.
        """
        if num_clips < 1:
            raise ValueError("num_clips must be >= 1")
        call_kwargs.setdefault("output_type", "pt_with_last_latent")

        defaults = _REGIMES[self._regime]["call_defaults"]
        for k, v in defaults.items():
            call_kwargs.setdefault(k, v)

        anchor_pil = _as_anchor_image(anchor_image if anchor_image is not None else image_prompt)
        clip_specs = _make_clip_specs(
            prompt=prompt,
            guidance_scale=call_kwargs.pop("guidance_scale"),
            guidance_scale_2=call_kwargs.pop("guidance_scale_2"),
            num_clips=num_clips,
            base_seed=base_seed,
            seed_stride=seed_stride,
        )

        try:
            return self._generate_clips_loop(
                clip_specs=clip_specs,
                anchor_image=anchor_pil,
                partial_output_path=partial_output_path,
                call_kwargs=call_kwargs,
            )
        finally:
            # Drop the anchor-encode cache so the next generate_long_video
            # call can re-encode with a different anchor / shape.
            self._anchor_latents_cache = None

    def _generate_clips_loop(
        self,
        *,
        clip_specs: List[_ClipSpec],
        anchor_image: Image.Image,
        partial_output_path,
        call_kwargs,
    ):
        prev_last_latent: Optional[torch.Tensor] = None
        clips: List[torch.Tensor] = []

        for clip_idx, spec in enumerate(clip_specs):
            logger.info(
                f"SVI clip {clip_idx + 1}/{len(clip_specs)} (seed={spec.seed}, "
                f"num_motion_latent={self._num_motion_latent}, "
                f"cfg={spec.guidance_scale}/{spec.guidance_scale_2}, "
                f"prompt={spec.prompt[:80]!r}...)"
            )

            result = self(
                prompts=[spec.prompt],
                image_prompt=[ImagePrompt(image=anchor_image, frame_pos=0)],
                seed=spec.seed,
                guidance_scale=spec.guidance_scale,
                guidance_scale_2=spec.guidance_scale_2,
                anchor_image=anchor_image,
                prev_last_latent=prev_last_latent,
                **call_kwargs,
            )

            if not (isinstance(result, tuple) and len(result) == 2):
                raise RuntimeError("SVI requires output_type='pt_with_last_latent' from the inner pipeline.")
            frames, prev_last_latent = result

            clips.append(frames[0] if frames.ndim == 5 else frames)

            # Save partial concat so a mid-run hang still leaves a usable
            # video covering the clips that completed.
            if partial_output_path is not None:
                partial = _concat_with_overlap(clips, overlap=self._num_overlap_frame)
                torch.save(partial.detach(), partial_output_path)
                logger.info(
                    f"saved partial output after clip {clip_idx + 1}/{len(clip_specs)} "
                    f"({partial.shape[0]} frames) to {partial_output_path}"
                )

        return _concat_with_overlap(clips, overlap=self._num_overlap_frame)


def _make_clip_specs(
    *,
    prompt: Union[str, List[str]],
    guidance_scale,
    guidance_scale_2,
    num_clips: int,
    base_seed: int,
    seed_stride: int,
) -> List[_ClipSpec]:
    prompts = _broadcast_prompt(prompt, num_clips)
    cfgs = _broadcast_float(guidance_scale, num_clips, "guidance_scale")
    cfg2s = _broadcast_float(guidance_scale_2, num_clips, "guidance_scale_2")
    return [
        _ClipSpec(
            prompt=prompts[i],
            seed=base_seed + i * seed_stride,
            guidance_scale=cfgs[i],
            guidance_scale_2=cfg2s[i],
        )
        for i in range(num_clips)
    ]


def _broadcast_prompt(prompt: Union[str, List[str]], num_clips: int) -> List[str]:
    if isinstance(prompt, list):
        if len(prompt) != num_clips:
            raise ValueError(f"prompt list length {len(prompt)} != num_clips {num_clips}")
        return prompt
    return [prompt] * num_clips


def _broadcast_float(value, num_clips: int, name: str) -> List[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != num_clips:
            raise ValueError(f"{name} list length {len(value)} != num_clips {num_clips}")
        return [float(v) for v in value]
    return [float(value)] * num_clips


def _as_anchor_image(value) -> Image.Image:
    """Accept a PIL image or single ImagePrompt/list wrapper from the I2V API."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("SVI supports a single anchor image; got list of length != 1")
        return _as_anchor_image(value[0])
    if isinstance(value, ImagePrompt):
        return value.image
    if not isinstance(value, Image.Image):
        raise TypeError("SVI requires anchor_image or image_prompt to be a PIL image")
    return value


def _concat_with_overlap(clips: List[torch.Tensor], *, overlap: int) -> torch.Tensor:
    """Temporal-concat decoded clips, dropping ``overlap`` boundary frames between adjacent clips."""
    if not clips:
        raise ValueError("no clips to concatenate")
    pieces = [clips[0]] + [c[overlap:] if overlap > 0 else c for c in clips[1:]]
    return torch.cat(pieces, dim=0)
