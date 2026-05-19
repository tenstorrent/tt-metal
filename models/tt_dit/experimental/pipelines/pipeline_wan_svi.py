# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Stable-Video-Infinity (SVI) 2.0 Pro pipeline for Wan2.2 I2V.

Generates long videos by autoregressively chaining short Wan2.2 I2V clips.
Continuity across clips is handled entirely in latent space — there is no
rolling pixel-frame handoff.

Per upstream's ``docs/svi/svi_2.0_pro.md``, each clip's I2V conditioning ``y``
is structured as::

    y = concat([anchor_latent, motion_latent, zero_padding], dim=temporal)

- ``anchor_latent``: the user-provided first image, VAE-encoded as a single
  latent frame. Same across every clip in the sequence.
- ``motion_latent``: the last ``num_motion_latent`` latent frames from the
  previous clip's denoised output (None on the first clip).
- ``zero_padding``: literal-zero latents for the remaining frame slots.

The mask is set at latent frame 0 only (the anchor position). Continuity
between clips is produced by the motion-latent splice plus the SVI LoRA's
trained behavior; the original "use decoded last frame as next clip's
image_prompt" mechanism from SVI 2.0 is explicitly NOT used in 2.0 Pro.

Two sampling regimes:

- ``python``: matches upstream ``inference_svi_2.0_pro.py``. 50 steps,
  CFG=5.0, ``FlowMatchEulerDiscreteScheduler`` at sigma_shift=5, SVI LoRA
  alone at 1.0/1.0 (high/low expert).
- ``comfyui``: matches the upstream ComfyUI workflow at
  ``comfyui_workflow/SVI-Wan22-1210-*-Clips.json``. 6 steps, CFG=1.5,
  ``flow_shift=8``, SVI LoRA at 1.0/1.0 stacked on top of LightX2V LoRA at
  0.5/1.0 (per upstream's ``docs/svi/comfyui.md``: LightX2V at 1.0 hurts
  SVI on the high-noise expert, so 0.5 is recommended there). Solver-wise,
  upstream uses k-diffusion ``dpm++_sde`` with Karras-fixed sigmas; tt-metal
  has no device-side ``DPMSolverSDESolver`` today, so we approximate with
  ``UniPCMultistepScheduler`` at the same flow_shift. Output is functionally
  equivalent but not bit-identical.

Upstream: https://github.com/vita-epfl/Stable-Video-Infinity (svi_wan22).
LoRA weights: HF ``vita-video-gen/svi-model``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

import PIL
import torch
from loguru import logger

from models.tt_dit.experimental.pipelines.pipeline_wan_lora import WanPipelineI2VLora
from models.tt_dit.experimental.utils.lora import LoRASpec
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt

Regime = Literal["python", "comfyui"]


class WanPipelineSVI(WanPipelineI2VLora):
    """Wan2.2 I2V with SVI 2.0 Pro LoRA + autoregressive clip-chaining driver.

    Args:
        svi_high, svi_low: Paths to SVI LoRA safetensors for the high- and
            low-noise experts respectively. Required for both regimes.
        lightx2v_high, lightx2v_low: Paths to LightX2V LoRA safetensors.
            Required for ``regime="comfyui"`` (which stacks them under SVI);
            must be ``None`` for ``regime="python"``.
        regime: ``"python"`` or ``"comfyui"`` (see module docstring).
        num_motion_latent: Number of latent frames from the previous clip's
            denoised output to splice into the next clip's conditioning at
            positions 1..N. Default 1 matches upstream's argparse default.
            Set to 0 to collapse SVI 2.0 Pro to SVI 2.0 behavior (anchor
            alone, no latent handoff).
        num_overlap_frame: Number of decoded pixel frames to drop on concat
            between adjacent clips. Driver-only — does not affect per-clip
            generation. Default 4 matches upstream's argparse default.
        sigma_shift: Optional override for the flow-shift parameter of the
            scheduler. Defaults are 5.0 for ``python`` and 8.0 for
            ``comfyui`` (both match upstream).
    """

    # Defaults applied to generate_long_video's call_kwargs per regime.
    # boundary_ratio is an __init__ parameter (passed through WanPipeline
    # create_pipeline) and is not in these dicts.
    PYTHON_CALL_DEFAULTS = dict(
        num_inference_steps=50,
        guidance_scale=5.0,
        guidance_scale_2=5.0,
    )
    COMFYUI_CALL_DEFAULTS = dict(
        num_inference_steps=6,
        guidance_scale=1.5,
        guidance_scale_2=1.5,
    )
    PYTHON_FLOW_SHIFT = 5.0
    COMFYUI_FLOW_SHIFT = 8.0

    def __init__(
        self,
        *args,
        svi_high: str,
        svi_low: str,
        lightx2v_high: Optional[str] = None,
        lightx2v_low: Optional[str] = None,
        regime: Regime = "python",
        num_motion_latent: int = 1,
        num_overlap_frame: int = 4,
        sigma_shift: Optional[float] = None,
        **kwargs,
    ):
        if regime not in ("python", "comfyui"):
            raise ValueError(f"regime must be 'python' or 'comfyui', got {regime!r}")
        if regime == "comfyui":
            if lightx2v_high is None or lightx2v_low is None:
                raise ValueError(
                    "regime='comfyui' requires lightx2v_high and lightx2v_low LoRA paths. "
                    "See experimental/models/Wan2_2_SVI.md for the recommended HF repo."
                )
            for label, path in [("lightx2v_high", lightx2v_high), ("lightx2v_low", lightx2v_low)]:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"{label}: file does not exist: {path}")
        elif lightx2v_high is not None or lightx2v_low is not None:
            raise ValueError(
                "lightx2v LoRA paths are only valid in regime='comfyui'. " "Drop them for regime='python'."
            )

        for label, path in [("svi_high", svi_high), ("svi_low", svi_low)]:
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label}: file does not exist: {path}")

        self._regime: Regime = regime
        self._num_motion_latent = int(num_motion_latent)
        self._num_overlap_frame = int(num_overlap_frame)

        # Build the LoRA stacks and pick the scheduler per regime.
        if regime == "python":
            kwargs["lora_high"] = [LoRASpec(svi_high, 1.0)]
            kwargs["lora_low"] = [LoRASpec(svi_low, 1.0)]
            shift = self.PYTHON_FLOW_SHIFT if sigma_shift is None else float(sigma_shift)
            if "scheduler" not in kwargs:
                from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

                # Matches diffsynth's FlowMatchScheduler("Wan") at sigma_shift=5
                # (plain Euler step on a flow-matching schedule).
                kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=shift)
        else:  # comfyui
            # Per upstream docs/svi/comfyui.md: LightX2V at 1.0 on the
            # high-noise expert "hurts SVI"; 0.5 is recommended.
            kwargs["lora_high"] = [
                LoRASpec(lightx2v_high, 0.5),
                LoRASpec(svi_high, 1.0),
            ]
            kwargs["lora_low"] = [
                LoRASpec(lightx2v_low, 1.0),
                LoRASpec(svi_low, 1.0),
            ]
            shift = self.COMFYUI_FLOW_SHIFT if sigma_shift is None else float(sigma_shift)
            if "scheduler" not in kwargs:
                from diffusers.schedulers import UniPCMultistepScheduler

                # Upstream uses k-diffusion dpm++_sde with Karras-fixed sigmas;
                # tt-metal has no DPMSolverSDESolver yet, so we approximate
                # with UniPC at the same flow_shift. Not bit-exact.
                kwargs["scheduler"] = UniPCMultistepScheduler(
                    use_flow_sigmas=True,
                    prediction_type="flow_prediction",
                    flow_shift=shift,
                )

        super().__init__(*args, **kwargs)

    @staticmethod
    def create_pipeline(*args, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        return WanPipeline.create_pipeline(*args, pipeline_class=WanPipelineSVI, **kwargs)

    # ------------------------------------------------------------------ #
    # __call__ wrapper. The base WanPipeline.__call__ does not accept
    # anchor_image / prev_last_latent and does not forward arbitrary kwargs
    # to prepare_latents. We stash them on `self` so the prepare_latents
    # override below can pick them up, then clear on exit.
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        *args,
        anchor_image: Optional[PIL.Image.Image] = None,
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

    # ------------------------------------------------------------------ #
    # Conditioning: matches upstream WanVideoUnit_ImageEmbedderVAE.
    # ------------------------------------------------------------------ #

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
        latents=None,
        anchor_image: Optional[PIL.Image.Image] = None,
        prev_last_latent: Optional[torch.Tensor] = None,
    ):
        """Build (latents, tt_y) matching upstream SVI 2.0 Pro conditioning.

        Calls the base ``WanPipelineI2V.prepare_latents`` with a single
        ``ImagePrompt(anchor, frame_pos=0)`` so the mask channels and the
        anchor-at-frame-0 image latent are correct, then:

        - splices ``prev_last_latent[:, :, -num_motion_latent:]`` into the
          image-latent channels at temporal positions 1..num_motion_latent;
        - exact-zeros the remaining image-latent slots (frames
          num_motion_latent+1..end) to match upstream's
          ``padding = torch.zeros(...)``.

        Falls back to the base behavior when neither ``anchor_image`` nor
        ``prev_last_latent`` is set (so this method stays usable as a
        vanilla I2V prepare_latents).
        """
        if anchor_image is None:
            anchor_image = getattr(self, "_svi_anchor_image", None)
        if prev_last_latent is None:
            prev_last_latent = getattr(self, "_svi_prev_last_latent", None)

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
                latents=latents,
            )

        # Encoder runs once with the anchor at frame 0, producing the correct
        # mask packing and the encoded anchor at latent frame 0. Subsequent
        # frames of tt_y get overwritten below.
        anchor_pil = anchor_image if anchor_image is not None else _unwrap_image(image_prompt)
        latents, tt_y = super().prepare_latents(
            batch_size=batch_size,
            image_prompt=[ImagePrompt(image=anchor_pil, frame_pos=0)],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
            latents=latents,
        )

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

    # ------------------------------------------------------------------ #
    # Autoregressive long-video driver.
    # ------------------------------------------------------------------ #

    def generate_long_video(
        self,
        *,
        prompt: Union[str, List[str]],
        image_prompt,
        num_clips: int,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        base_seed: int = 0,
        seed_stride: int = 42,
        **call_kwargs,
    ):
        """Run ``num_clips`` chained I2V generations and return the concat.

        Anchor stays constant across all clips; continuity comes from
        ``prev_last_latent`` spliced into each next clip's conditioning.

        ``prompt`` may be a single string (used for every clip) or a
        ``List[str]`` of length ``num_clips`` (one per clip, mirrors
        upstream's per-clip ``prompt.txt``).

        Returns a tensor (or numpy array, depending on the inner
        ``output_type``) of shape::

            (num_clips * num_frames - (num_clips - 1) * num_overlap_frame,
             ...spatial dims..., 3)
        """
        if num_clips < 1:
            raise ValueError("num_clips must be >= 1")
        call_kwargs.setdefault("output_type", "pt")

        # Regime-specific defaults for the inner __call__ (steps, CFG, etc.).
        defaults = self.PYTHON_CALL_DEFAULTS if self._regime == "python" else self.COMFYUI_CALL_DEFAULTS
        for k, v in defaults.items():
            call_kwargs.setdefault(k, v)

        anchor_pil = _unwrap_image(image_prompt)

        if isinstance(prompt, list):
            if len(prompt) != num_clips:
                raise ValueError(f"prompt list length {len(prompt)} != num_clips {num_clips}")
            prompts_per_clip = list(prompt)
        else:
            prompts_per_clip = [prompt] * num_clips

        prev_last_latent: Optional[torch.Tensor] = None
        clips: List[torch.Tensor] = []

        for clip_idx, clip_prompt in enumerate(prompts_per_clip):
            seed = base_seed + clip_idx * seed_stride
            logger.info(
                f"SVI clip {clip_idx + 1}/{num_clips} (seed={seed}, "
                f"num_motion_latent={self._num_motion_latent}, "
                f"prompt={clip_prompt[:80]!r}...)"
            )

            result = self(
                prompt=clip_prompt,
                image_prompt=[ImagePrompt(image=anchor_pil, frame_pos=0)],
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                return_last_latent=True,
                anchor_image=anchor_pil,
                prev_last_latent=prev_last_latent,
                **call_kwargs,
            )

            frames = result.frames if hasattr(result, "frames") else result[0]
            prev_last_latent = getattr(result, "last_latent", None)
            if prev_last_latent is None:
                raise RuntimeError("SVI requires return_last_latent=True from the inner pipeline.")

            # frames is (B, T, ...) for pt/np outputs — drop the batch dim.
            clips.append(frames[0] if frames.ndim == 5 else frames)

        return _concat_with_overlap(clips, overlap=self._num_overlap_frame)


def _unwrap_image(image_prompt) -> PIL.Image.Image:
    """Coerce the user's image_prompt (PIL, ImagePrompt, or list of either)
    to a single PIL.Image for use as the SVI anchor."""
    if isinstance(image_prompt, list):
        if len(image_prompt) != 1:
            raise ValueError("SVI supports a single anchor image; got list of length != 1")
        return _unwrap_image(image_prompt[0])
    if isinstance(image_prompt, ImagePrompt):
        return image_prompt.image
    return image_prompt


def _concat_with_overlap(clips: List[torch.Tensor], *, overlap: int):
    """Temporal-concat decoded clips, dropping ``overlap`` boundary frames
    between adjacent clips. Accepts torch.Tensor or numpy.ndarray clips."""
    if not clips:
        raise ValueError("no clips to concatenate")
    if len(clips) == 1:
        return clips[0]
    pieces = [clips[0]] + [c[overlap:] if overlap > 0 else c for c in clips[1:]]
    if isinstance(pieces[0], torch.Tensor):
        return torch.cat(pieces, dim=0)
    import numpy as np

    return np.concatenate(pieces, axis=0)
