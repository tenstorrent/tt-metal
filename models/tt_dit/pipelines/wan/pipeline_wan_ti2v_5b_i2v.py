# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B image-to-video on a single BH Galaxy (4x8).

TI2V-5B conditions on an image differently from Wan2.2-14B I2V. The 14B model widens the
transformer to 36 channels and concatenates the conditioning latents onto the channel axis.
TI2V-5B keeps 48 channels in and out, has `image_dim: null`, and ships no image encoder;
instead it **pins latent frame 0 to the conditioning latent and gives that frame's tokens a
timestep of 0** while every other token sees `t`. So this subclasses the 5B T2V pipeline and
overrides the conditioning hooks rather than reusing `WanPipelineI2V`.

The reference is `diffusers.pipelines.wan.pipeline_wan_i2v.WanImageToVideoPipeline` with
`expand_timesteps=True`; the host-side math lives in `ti2v_5b_i2v_math` and is unit-tested
against it in `tests/models/wan2_2/test_ti2v_5b_i2v_math.py`.

The conditioning image is encoded on the **host**, with the torch `AutoencoderKLWan` the VAE
adapter already holds. The TT VAE encoder cannot do it: `WanEncoder3D` asserts
`not is_residual` (`models/vae/vae_wan2_1.py:1760`) and only the residual *decoder* was ported.
That is a deliberate first-cut choice, not an oversight -- it encodes one frame rather than the
81 the decoder emits, it *is* the reference so it removes a variable from correctness
debugging, and it gives a future TT encoder its acceptance golden. Its cost is timed and
logged (`last_image_encode_seconds`) and falls inside the `prepare_latents` section
`WanPipeline.__call__` already emits, so it can be measured rather than guessed at.
"""

from __future__ import annotations

import os
import time

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from loguru import logger
from PIL import Image

import ttnn

from ...utils import tensor
from ...utils.tensor import float32_tensor
from .pipeline_wan import WanPipelineConfig
from .pipeline_wan_i2v import ImagePrompt, _parse_image_prompts
from .pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from .ti2v_5b_i2v_math import blend_condition, first_frame_mask, normalize_latents, pad_timesteps_for_sequence_parallel


class WanTI2V5BI2VPipeline(WanTI2V5BPipeline):
    """Wan2.2 TI2V-5B with image conditioning (per-token timestep + latent pinning)."""

    @classmethod
    def _config_overrides(cls) -> dict[str, object]:
        return {
            **super()._config_overrides(),
            # Switches `_step` onto the per-token timestep branch. Requires `prepare_mask`,
            # `prepare_timestep_tensor`, `get_model_input` and `finalize_latents` below.
            "expand_timesteps": True,
        }

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        scheduler: SchedulerMixin | None = None,
        run_warmup: bool = True,
        lora_enabled: bool = False,
    ) -> None:
        # Per-run conditioning state, consumed by the hooks below. Set before super().__init__
        # because the warmup call reaches prepare_latents.
        self._first_frame_mask: torch.Tensor | None = None
        self._condition_latent: torch.Tensor | None = None
        self._mask_1BND: ttnn.Tensor | None = None
        self._masks_installed = False
        self.last_image_encode_seconds: float = 0.0

        # Warm up ourselves, after our own state exists, and with an image_prompt -- otherwise
        # `condition_buffer` is never allocated (see pipeline_wan.py:436-438, :756-759).
        super().__init__(device=device, config=config, scheduler=scheduler, run_warmup=False, lora_enabled=lora_enabled)
        if run_warmup:
            logger.info("Pipeline allocation run (I2V)...")
            self._warmup()

    def _warmup(self) -> None:
        """Allocate buffers with a sample image sized to the target resolution.

        `guidance_scale_2` stays None: the 5B checkpoint is dense (`boundary_ratio is None`),
        and `__call__` rejects a second guidance scale in that case.
        """
        self(
            prompts=["warmup"],
            image_prompt=Image.new("RGB", (self._width, self._height)),
            num_inference_steps=2,
            guidance_scale=2 if self._cfg_enabled else 1,
            guidance_scale_2=None,
        )

    # ------------------------------------------------------------------ conditioning

    def encode_condition_image(
        self,
        image: Image.Image,
        *,
        height: int,
        width: int,
        device: torch.device | None,
    ) -> torch.Tensor:
        """Conditioning image -> one normalized 48-channel latent frame, shape (1, 48, 1, h, w).

        Mirrors diffusers `pipeline_wan_i2v.py:425-460`: preprocess to [-1, 1], add a singleton
        temporal axis, encode, take the posterior **mode** (not a sample, so the result is
        deterministic), then normalize into the transformer's latent space.

        This is the swap point for a TT residual encoder: keep the signature and return shape
        and nothing else has to change.
        """
        torch_vae = self._vae._torch_vae

        pixels = self.video_processor.preprocess(image, height=height, width=width)
        pixels = pixels.to(device=device, dtype=torch_vae.dtype)
        video_condition = pixels.unsqueeze(2)  # (1, 3, 1, H, W) -- just the seed frame

        # bf16 autocast on the host encode. Measured on this host (EPYC 9354P, avx512_bf16,
        # 1280x704): 3.386s -> 2.770s median of 3, a 0.616s saving at a latent PCC of 0.9999880
        # against fp32 (max abs deviation 0.0146 on a latent whose std is 0.4435). Set
        # WAN5B_I2V_ENCODE_FP32=1 to force fp32 -- worth doing if the conditioning itself is
        # ever under suspicion, since fp32 is bit-faithful to the diffusers reference.
        use_fp32 = os.environ.get("WAN5B_I2V_ENCODE_FP32") == "1"
        with torch.no_grad():
            if use_fp32:
                latent = torch_vae.encode(video_condition).latent_dist.mode()
            else:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    latent = torch_vae.encode(video_condition).latent_dist.mode()
            latent = latent.to(torch.float32)

        mean = self._vae._latents_mean.to(latent.device, latent.dtype)
        std = self._vae._latents_std.to(latent.device, latent.dtype)
        return normalize_latents(latent, mean, std).to(torch.float32)

    def prepare_latents(
        self,
        batch_size: int,
        image_prompt=None,
        num_channels_latents: int = 48,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Noise latents plus the conditioning latent, as ``(latents, cond_latents)``.

        The returned conditioning is **expanded to the full frame count**. diffusers keeps it
        at ``T == 1`` and relies on broadcasting, but here it is patchified by
        ``preprocess_spatial_input_host`` and must therefore yield the same token count ``N``
        as the latents. Expanding on the host costs ~21 MB at 1280x704/121f.
        """
        assert batch_size == 1, "Only batch size 1 is currently supported for TI2V-5B I2V"
        if image_prompt is None:
            msg = "TI2V-5B I2V requires an image_prompt"
            raise ValueError(msg)

        prompts = _parse_image_prompts(image_prompt, num_frames)
        if len(prompts) != 1 or prompts[0].frame_pos != 0:
            positions = [p.frame_pos for p in prompts]
            msg = (
                f"TI2V-5B conditions only on latent frame 0 (its mask pins frame 0 and nothing "
                f"else), so exactly one ImagePrompt at frame_pos=0 is representable; got {positions}"
            )
            raise ValueError(msg)

        latents, _ = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            device=device,
        )

        # The host encode lands inside the `prepare_latents` section `__call__` already emits,
        # so the perf test can read its cost there; log it separately too, since it is the one
        # part of the I2V pipeline that a TT encoder port would remove.
        encode_start = time.perf_counter()
        condition = self.encode_condition_image(
            prompts[0].image, height=height, width=width, device=torch.device(device) if device else None
        )
        self.last_image_encode_seconds = time.perf_counter() - encode_start
        logger.info(f"I2V host image encode: {self.last_image_encode_seconds:.3f}s")

        num_latent_frames, latent_h, latent_w = latents.shape[2], latents.shape[3], latents.shape[4]
        assert condition.shape[-2:] == (latent_h, latent_w), (
            f"conditioning latent {tuple(condition.shape)} does not match latent grid "
            f"{(latent_h, latent_w)}; check the VAE spatial scale factor"
        )

        # Cached for prepare_mask / finalize_latents, which run later in the same __call__.
        self._condition_latent = condition
        self._first_frame_mask = first_frame_mask(
            num_latent_frames, latent_h, latent_w, dtype=torch.float32, device=condition.device
        )

        cond_latents = condition.expand(-1, -1, num_latent_frames, -1, -1).contiguous()
        return latents, cond_latents

    # ------------------------------------------------------------------ hooks

    def prepare_mask(self, latents: torch.Tensor, *, device: torch.device | None) -> torch.Tensor:
        """The first-frame mask built in `prepare_latents`, not the base all-ones mask."""
        assert self._first_frame_mask is not None, "prepare_latents must run before prepare_mask"
        return self._first_frame_mask.to(device=device) if device is not None else self._first_frame_mask

    def prepare_timestep_tensor(self, timestep: torch.Tensor, *, t: float, traced: bool) -> ttnn.Tensor:
        """Emit the two distinct timestep values and install the per-token expansion masks.

        `_step` hands us a `(batch, N)` per-token schedule, but for TI2V-5B it only ever holds
        two values: 0 on the conditioned frame and `t` everywhere else. The timestep embedder is
        pointwise in the token axis, so embedding those two rows and expanding through a mask is
        mathematically identical to embedding all N -- while running the MLP at M=32 (tile-padded
        from 2) instead of M=N/SP. At 1280x704/81f that is 2 rows instead of 2336, and it is the
        same shape the scalar path already uses, so it needs no matmul blocking entries and
        cannot overflow L1 at any resolution.

        The 2-row tensor is replicated and tiny, so it follows the proven scalar upload pattern
        (host tensor when traced, so the tracer copies it in per step). Only the masks are
        sequence-parallel, and they are persistent so a trace binds them by address.
        """
        assert timestep.shape[0] == 1, "batch size 1 only"
        self._install_per_token_masks()

        # Row order must match the mask convention: 0 -> row 0, t -> row 1.
        rows = torch.tensor([0.0, float(t)], dtype=torch.float32).reshape(1, 1, 2, 1)
        return float32_tensor(rows, device=(None if traced else self.mesh_device))

    def _install_per_token_masks(self) -> None:
        """Build and cache the temb / timestep-projection expansion masks on the transformer."""
        if self._masks_installed:
            return
        dim = self.transformer.dim
        tp = self.parallel_config.tensor_parallel.factor
        # fp32 to match the embedder's output: the scalar path adds temb straight to the fp32
        # scale_shift_table and only typecasts the gates to bf16 afterwards.
        temb_mask = self._token_mask(dim // tp, dtype=ttnn.float32)
        proj_mask = self._token_mask(6 * (dim // tp), dtype=ttnn.float32)
        self.transformer.set_per_token_timestep_masks(temb_mask, proj_mask)
        self._masks_installed = True

    def _token_mask(self, width: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        """Per-token binary mask at `width`, sequence-parallel sharded.

        0 on the conditioned frame's tokens, 1 elsewhere -- including the sequence-parallel pad
        tail, which must read as `t` to match the scalar path. Token order is frame-major
        (`preprocess_spatial_input_host` permutes to `(B, patch_F, patch_H, patch_W, ...)`), so
        the conditioned frame's tokens are the contiguous head of the global sequence; note that
        head lands only on the first shard, which is why this is a mask rather than a slice.
        """
        assert self._first_frame_mask is not None, "prepare_latents must run first"
        sp = self.parallel_config.sequence_parallel
        _, ph, pw = self.transformer.patch_size
        tokens = self._first_frame_mask[0, 0, :, ::ph, ::pw].reshape(-1)
        tokens = pad_timesteps_for_sequence_parallel(tokens, sp.factor, fill=1.0)
        mask = tokens.reshape(1, 1, -1, 1).expand(-1, -1, -1, width).contiguous()
        logger.info(f"I2V per-token mask: {tuple(mask.shape)} (width {width})")
        return tensor.from_torch(
            mask,
            device=self.mesh_device,
            mesh_axes=[None, None, sp.mesh_axis, None],
            dtype=dtype,
        )

    def get_model_input(self, latents: ttnn.Tensor, cond_latents: ttnn.Tensor | None) -> ttnn.Tensor:
        """Blend the conditioning into the latents: `(1 - mask) * cond + mask * latents`.

        Mirrors diffusers `pipeline_wan_i2v.py:758`. Pins latent frame 0's tokens to the
        conditioning latent at every step; `finalize_latents` repeats it once after the loop.
        The solver still steps on the *unblended* latents, matching the reference.
        """
        assert cond_latents is not None, "I2V requires conditioning latents"
        latents = super().get_model_input(latents, None)  # fp32 -> bf16

        mask = self._device_token_mask(latents)
        # lerp(a, b, w) = a + w * (b - a) = (1 - w) * a + w * b. A full-shape weight avoids
        # relying on broadcast semantics for the tensor-weight overload.
        return ttnn.lerp(cond_latents, latents, mask)

    def _device_token_mask(self, like: ttnn.Tensor) -> ttnn.Tensor:
        """Per-token mask matching the model input width, built once and cached."""
        if self._mask_1BND is None:
            self._mask_1BND = self._token_mask(like.shape[-1], dtype=like.dtype)
        return self._mask_1BND

    def finalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Re-pin the conditioned frame after the denoise loop.

        Mirrors diffusers `pipeline_wan_i2v.py:813-814`. Without this, latent frame 0 is
        whatever the solver last produced rather than exactly the conditioning latent, and the
        decoded first frame drifts from the seed image.
        """
        if self._first_frame_mask is None or self._condition_latent is None:
            return latents
        mask = self._first_frame_mask.to(device=latents.device, dtype=latents.dtype)
        condition = self._condition_latent.to(device=latents.device, dtype=latents.dtype)
        return blend_condition(condition, latents, mask)


__all__ = ["ImagePrompt", "WanTI2V5BI2VPipeline"]
