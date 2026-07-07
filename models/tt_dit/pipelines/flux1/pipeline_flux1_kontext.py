# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-Kontext-dev pipeline (instruction-based image editing).

This extends the text-to-image ``Flux1Pipeline`` with the four pipeline-layer
deltas that turn FLUX.1-dev into FLUX.1-Kontext-dev. The transformer, text
encoders, VAE decoder, and scheduler are reused unchanged (Kontext shares the
diffusers ``FluxTransformer2DModel`` with dev, so ``Flux1Checkpoint`` loads the
Kontext weights via ``checkpoint_name`` alone).

Deltas vs ``pipeline_flux1.py`` (see design doc for the diffusers ground truth):
  D1  VAE-encode the reference image (mode/argmax), normalize with
      (z - shift_factor) * scaling_factor, pack -> ``image_latents``.
  D2  RoPE ids: noise tokens keep channel 0 = 0; reference-image tokens set
      channel 0 = 1 (row/col coords deliberately overlap).
  D3  Each step feed the transformer ``cat([noise, image_latents], dim=seq)``
      (noise first) with the correspondingly concatenated RoPE.
  D4  Slice the transformer output back to the noise tokens before the solver.

Sequence-parallel note (design doc §4): noise and image are fractured
*separately* on the sp axis and concatenated *on device* so that each shard is
``[noise_shard | image_shard]``. Slicing the first ``n // sp`` tokens per shard
then recovers exactly the global noise sequence. For ``sp == 1`` this reduces to
a plain concat + ``[:, :n]`` slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.models.transformers.transformer_flux1 import Flux1Checkpoint
from models.tt_dit.models.vae.vae_sd35 import VAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, distribute_cfg
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.flux1.pipeline_flux1 import (
    _PRESETS_BH,
    _PRESETS_WH,
    _VAE_SCALE_FACTOR,
    _calculate_shift,
    _latent_image_ids,
    _pack_latents,
    _unpack_latents,
)
from models.tt_dit.pipelines.flux1.text_encoder import TextEncoder
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils.tensor import from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

# Kontext preferred resolutions (diffusers PREFERRED_KONTEXT_RESOLUTIONS).
PREFERRED_KONTEXT_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
)


def _snap_to_preferred_resolution(width: int, height: int) -> tuple[int, int]:
    """Snap (width, height) to the closest-aspect Kontext preferred resolution."""
    aspect = width / height
    _, best_w, best_h = min((abs(aspect - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
    return best_w, best_h


@dataclass(frozen=True, kw_only=True)
class Flux1KontextPipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    enable_t5_text_encoder: bool
    use_torch_t5_text_encoder: bool
    use_torch_clip_text_encoder: bool

    height: int
    width: int
    cfg_enabled: bool

    checkpoint_name: str

    # Optional FLUX.1 LoRA fused into the transformer weights at load time.
    lora_path: str | None = None
    lora_scale: float = 1.0

    @classmethod
    def default(
        cls,
        *,
        mesh_shape: ttnn.MeshShape,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        num_links: int | None = None,
        dit_parallel_config: DiTParallelConfig | None = None,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        enable_t5_text_encoder: bool = True,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = False,
        checkpoint_name: str,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> Flux1KontextPipelineConfig:
        preset_dict = _PRESETS_BH if is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=preset["sp"], tp=preset["tp"])
        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig.from_tuple(preset["encoder_tp"])
        if vae_parallel_config is None:
            vae_parallel_config = VAEParallelConfig.from_tuple(preset["vae_tp"])

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            enable_t5_text_encoder=enable_t5_text_encoder,
            use_torch_t5_text_encoder=use_torch_t5_text_encoder,
            use_torch_clip_text_encoder=use_torch_clip_text_encoder,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )


class Flux1KontextPipeline(PipelineAPIMixin):
    """FLUX.1-Kontext-dev: edit an input image following a text instruction."""

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = False,
        checkpoint_name: str,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> Flux1KontextPipeline:
        config = Flux1KontextPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: Flux1KontextPipelineConfig) -> None:
        self._mesh_device = device
        self._parallel_config = config.dit_parallel_config
        self._encoder_parallel_config = config.encoder_parallel_config
        self._vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        self._submesh_devices = create_submeshes(self._mesh_device, config.dit_parallel_config)

        self._ccl_managers = [
            CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
            for submesh_device in self._submesh_devices
        ]
        self._cfg_combiner = CFGCombiner(self._submesh_devices)

        self.encoder_device = self._submesh_devices[0]
        self.vae_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0
        self.vae_submesh_idx = 0

        logger.info("creating TT-NN transformer...")
        checkpoint_name = config.checkpoint_name
        checkpoint = Flux1Checkpoint(checkpoint_name, lora_path=config.lora_path, lora_scale=config.lora_scale)
        self.transformers = [
            checkpoint.build(ccl_manager=mgr, parallel_config=config.dit_parallel_config) for mgr in self._ccl_managers
        ]
        self.synchronize_devices()

        self._tracers = [Tracer(self._traced_step, device=device, prep_run=False) for device in self._submesh_devices]
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self._solvers = [EulerSolver() for _ in self._submesh_devices]

        self._pos_embed = checkpoint.pos_embed
        self._num_channels_latents = checkpoint.num_channels_latents
        self._joint_attention_dim = checkpoint.joint_attention_dim
        self._patch_size = checkpoint.patch_size
        self._with_guidance_embeds = checkpoint.with_guidance_embeds

        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

        logger.info("creating text encoder...")
        self._text_encoder = TextEncoder(
            checkpoint_name=checkpoint_name,
            device=self.encoder_device,
            ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
            parallel_config=self._encoder_parallel_config,
            enable_t5=config.enable_t5_text_encoder,
            joint_attention_dim=self._joint_attention_dim,
            use_torch_clip=config.use_torch_clip_text_encoder,
            use_torch_t5=config.use_torch_t5_text_encoder,
        )
        ttnn.synchronize_device(self.encoder_device)

        logger.info("creating VAE decoder...")
        self._vae = VAEDecoderAdapter(
            checkpoint_name=checkpoint_name,
            parallel_config=self._vae_parallel_config,
            ccl_manager=self._ccl_managers[self.vae_submesh_idx],
            use_torch=False,
        )

        # D1: reference-image VAE *encoder*. Host-side torch for the first port
        # (one encode per request is negligible vs the denoising loop). A future
        # optimization can move this on-device (see stable_diffusion_xl_base tt_encoder).
        logger.info("creating host VAE encoder (reference image)...")
        self._vae_encoder = AutoencoderKL.from_pretrained(
            checkpoint_name, subfolder="vae", torch_dtype=torch.bfloat16
        ).eval()
        self._vae_shift = self._vae_encoder.config.shift_factor
        self._vae_scale = self._vae_encoder.config.scaling_factor

        logger.info("Pipeline allocation run...")
        # Warmup with a blank reference image to trigger compile/trace allocation.
        from PIL import Image as _Image

        self(
            image=_Image.new("RGB", (self._width, self._height)),
            prompts=[""],
            num_inference_steps=2,
            cfg_scale=2 if config.cfg_enabled else 1,
            traced=False,
        )

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        *,
        image: Image.Image | None = None,
        width: int | None = None,
        height: int | None = None,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 1,
        guidance_scale: float = 3.5,  # Kontext default
        prompts: Sequence[str],
        prompts_2: Sequence[str] | None = None,
        negative_prompts: Sequence[str] | None = None,
        negative_prompts_2: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        traced: bool = False,
        vae_traced: bool | None = False,
        encoder_traced: bool | None = None,
        clip_skip: int = 0,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        on_event = on_event if on_event is not None else null_callback
        prompts_2 = prompts_2 if prompts_2 is not None else prompts
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * len(prompts)
        negative_prompts_2 = negative_prompts_2 if negative_prompts_2 is not None else negative_prompts

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        prompt_count = len(prompts)
        # Per-request resolution: default to the configured size, then snap to the
        # nearest Kontext preferred bucket. Untraced runs recompile kernels for a
        # new shape on first use, so any preferred resolution is usable.
        width = width if width is not None else self._width
        height = height if height is not None else self._height
        width, height = _snap_to_preferred_resolution(width, height)
        # image is optional: with a reference image this is instruction editing
        # (concat noise+image, D1-D4); without one it degrades to plain FLUX.1-dev
        # text-to-image (noise-only sequence, no image concat/ids/slice offset).
        has_image = image is not None

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        sp_factor = self._parallel_config.sequence_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"
        if cfg_scale > 1 and not self._cfg_enabled:
            raise ValueError("cfg_scale > 1 requires CFG to be enabled")

        on_event(SectionStart("total"))
        assert height % (_VAE_SCALE_FACTOR * self._patch_size) == 0
        assert width % (_VAE_SCALE_FACTOR * self._patch_size) == 0

        latents_height = height // _VAE_SCALE_FACTOR
        latents_width = width // _VAE_SCALE_FACTOR
        latents_sequence_length = latents_height * latents_width  # noise token count (n)

        # ---- encode prompts (unchanged) ----
        logger.info("encoding prompts...")
        on_event(SectionStart("encoder"))
        torch_context, torch_pooled = self._text_encoder.encode_cfg(
            (prompts, prompts_2),
            (negative_prompts, negative_prompts_2),
            num_images_per_prompt=num_images_per_prompt,
            cfg_enabled=self._cfg_enabled,
            clip_skip=clip_skip,
            traced=encoder_traced,
            on_event=on_event,
        )
        _, prompt_sequence_length, _ = torch_context.shape
        on_event(SectionEnd("encoder"))

        # ---- D1: VAE-encode the reference image on host, pack (edit mode only) ----
        if has_image:
            logger.info("encoding reference image (host VAE)...")
            image_latents_torch, img_h, img_w = self._encode_reference_image(image, height, width)
            image_sequence_length = img_h * img_w
        else:
            logger.info("no reference image -> text-to-image generation")
            image_latents_torch, img_h, img_w = None, 0, 0
            image_sequence_length = 0

        # ---- timesteps (unchanged; scheduler shift uses noise seq len) ----
        self._scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            mu=_calculate_shift(latents_sequence_length, self._scheduler),
        )
        sigmas = self._scheduler.sigmas.tolist()
        for solver in self._solvers:
            solver.set_schedule(sigmas)
        timesteps = self._scheduler.timesteps

        torch_guidance = (
            torch.full([prompt_count * num_images_per_prompt], fill_value=guidance_scale)
            if self._with_guidance_embeds
            else None
        )

        # ---- D2: ids. text=0, noise ch0=0, image ch0=1 (row/col overlap) ----
        logger.info("preparing latents & rope ids...")
        text_ids = torch.zeros([prompt_sequence_length, 3])
        latent_ids = _latent_image_ids(height=latents_height, width=latents_width)  # ch0 = 0
        if has_image:
            image_ids = _latent_image_ids(height=img_h, width=img_w)
            image_ids[..., 0] = 1  # the single Kontext-specific line
            ids = torch.cat((text_ids, latent_ids, image_ids), dim=0)
        else:
            ids = torch.cat((text_ids, latent_ids), dim=0)
        torch_rope_cos, torch_rope_sin = self._pos_embed.forward(ids)

        p = prompt_sequence_length
        n = latents_sequence_length
        prompt_cos, prompt_sin = torch_rope_cos[:p], torch_rope_sin[:p]
        noise_cos, noise_sin = torch_rope_cos[p : p + n], torch_rope_sin[p : p + n]
        image_cos, image_sin = torch_rope_cos[p + n :], torch_rope_sin[p + n :]  # empty when no image

        # ---- distribute to devices ----
        context = distribute_cfg(torch_context, devices=self._submesh_devices)
        pooled = distribute_cfg(torch_pooled, devices=self._submesh_devices)
        latents = self._random_latents(
            batch_size=prompt_count * num_images_per_prompt, seed=seed, width=width, height=height
        )  # noise (updated)

        # image latents fractured on the sp axis exactly like noise, kept constant across steps.
        if has_image:
            image_latents = from_torch_to_devices(
                image_latents_torch, devices=self._submesh_devices, mesh_axes=[None, sp_axis, None]
            )
        else:
            image_latents = [None] * len(self._submesh_devices)

        guidance = (
            from_torch_to_devices(torch_guidance.unsqueeze(-1), devices=self._submesh_devices)
            if torch_guidance is not None
            else [None] * len(self._submesh_devices)
        )

        # D3: combined spatial RoPE = per-shard [noise | image] (§4 SP-safe).
        # Text-to-image (no reference) uses the noise-only RoPE with no image concat.
        noise_cos_dev = from_torch_to_devices(noise_cos, devices=self._submesh_devices, mesh_axes=[sp_axis, None])
        noise_sin_dev = from_torch_to_devices(noise_sin, devices=self._submesh_devices, mesh_axes=[sp_axis, None])
        if has_image:
            image_cos_dev = from_torch_to_devices(image_cos, devices=self._submesh_devices, mesh_axes=[sp_axis, None])
            image_sin_dev = from_torch_to_devices(image_sin, devices=self._submesh_devices, mesh_axes=[sp_axis, None])
            spatial_cos = [
                ttnn.concat([noise_cos_dev[i], image_cos_dev[i]], dim=0) for i in range(len(self._submesh_devices))
            ]
            spatial_sin = [
                ttnn.concat([noise_sin_dev[i], image_sin_dev[i]], dim=0) for i in range(len(self._submesh_devices))
            ]
        else:
            spatial_cos = noise_cos_dev
            spatial_sin = noise_sin_dev

        prompt_rope_cos = from_torch_to_devices(prompt_cos, devices=self._submesh_devices)
        prompt_rope_sin = from_torch_to_devices(prompt_sin, devices=self._submesh_devices)

        combined_sequence_length = n + image_sequence_length
        noise_local_len = n // sp_factor  # tokens to keep per shard after forward (D4)

        # ---- denoising ----
        logger.info("denoising...")
        on_event(SectionStart("denoising"))
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            on_event(SectionStart(f"denoising_step_{i}"))
            velocity_preds = []
            for idx, tracer in enumerate(self._tracers):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self._submesh_devices[idx],
                )
                inputs = tracer.inputs
                velocity_preds.append(
                    tracer(
                        cfg_enabled=self._cfg_enabled,
                        submesh_idx=idx,
                        latents=latents[idx],  # noise only; updated each step
                        image_latents=image_latents[idx] if i == 0 else inputs["image_latents"],
                        prompt=context[idx] if i == 0 else inputs["prompt"],
                        pooled=pooled[idx] if i == 0 else inputs["pooled"],
                        timestep=timestep,
                        guidance=guidance[idx] if i == 0 and guidance else inputs["guidance"],
                        spatial_rope=(spatial_cos[idx], spatial_sin[idx]) if i == 0 else inputs["spatial_rope"],
                        prompt_rope=(prompt_rope_cos[idx], prompt_rope_sin[idx]) if i == 0 else inputs["prompt_rope"],
                        combined_sequence_length=combined_sequence_length,
                        prompt_sequence_length=prompt_sequence_length,
                        noise_local_len=noise_local_len,
                        traced=traced,
                        tracer_blocking_execution=False,
                    )
                )
                latents[idx] = tracer.inputs["latents"]

            if self._cfg_enabled:
                velocity_preds = self._cfg_combiner.combine(velocity_preds, cfg_scale)

            latents = [
                solver.step(step=i, latent=latents[idx], velocity_pred=velocity_preds[idx])
                for idx, solver in enumerate(self._solvers)
            ]
            self.synchronize_devices()
            on_event(SectionEnd(f"denoising_step_{i}"))
        on_event(SectionEnd("denoising"))

        # ---- decode (unchanged) ----
        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        output = self._decode_latents(latents[self.vae_submesh_idx], traced=vae_traced, width=width, height=height)
        on_event(SectionEnd("vae"))
        on_event(SectionEnd("total"))
        return output

    # ------------------------------------------------------------------ #
    def _encode_reference_image(self, image: Image.Image, height: int, width: int) -> tuple[torch.Tensor, int, int]:
        """D1: preprocess -> VAE encode (mode) -> normalize -> pack. Returns (packed, img_h, img_w)."""
        ref = self._image_processor.preprocess(image, height=height, width=width)  # [-1,1], BCHW float
        with torch.no_grad():
            z = self._vae_encoder.encode(ref.to(torch.bfloat16)).latent_dist.mode()
        z = (z - self._vae_shift) * self._vae_scale
        b, c, h2, w2 = z.shape
        img_h, img_w = h2 // 2, w2 // 2
        packed = _pack_latents(z, b, self._num_channels_latents, img_h, img_w)  # [B, img_h*img_w, C*4]
        return packed, img_h, img_w

    def synchronize_devices(self) -> None:
        for submesh_device in self._submesh_devices:
            ttnn.synchronize_device(submesh_device)

    def _random_latents(
        self, *, batch_size: int, seed: int, width: int | None = None, height: int | None = None
    ) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)
        latents_height = (height if height is not None else self._height) // _VAE_SCALE_FACTOR
        latents_width = (width if width is not None else self._width) // _VAE_SCALE_FACTOR
        shape = [batch_size, self._num_channels_latents, latents_height * 2, latents_width * 2]
        latents = _pack_latents(
            torch.randn(shape, dtype=torch.bfloat16),
            batch_size,
            self._num_channels_latents,
            latents_height,
            latents_width,
        )
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        return from_torch_to_devices(latents, devices=self._submesh_devices, mesh_axes=[None, sp_axis, None])

    def _decode_latents(
        self, tt_latents: ttnn.Tensor, *, traced: bool, width: int | None = None, height: int | None = None
    ) -> list[Image.Image]:
        ttnn.synchronize_device(self.vae_device)
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
            tt_latents, dim=1, mesh_axis=sp_axis, use_hyperparams=True
        )
        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        unpack_h = height if height is not None else self._height
        unpack_w = width if width is not None else self._width
        torch_latents = _unpack_latents(torch_latents, unpack_h, unpack_w, _VAE_SCALE_FACTOR)
        torch_latents = torch_latents.permute(0, 2, 3, 1)  # BCHW -> NHWC
        decoded_output = self._vae.decode(torch_latents, traced=traced)
        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)
        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

    def _traced_step(
        self,
        *,
        cfg_enabled: bool,
        submesh_idx: int,
        latents: ttnn.Tensor,
        image_latents: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        combined_sequence_length: int,
        noise_local_len: int,
        **kwargs: Any,
    ) -> ttnn.Tensor:
        # D3: concat noise + reference-image tokens on device (noise first).
        # Text-to-image has no image tokens, so the sequence is noise-only.
        spatial = ttnn.concat([latents, image_latents], dim=1) if image_latents is not None else latents
        if cfg_enabled and not self._parallel_config.cfg_parallel.factor > 1:
            spatial = ttnn.concat([spatial, spatial])

        velocity = self.transformers[submesh_idx].forward(
            spatial=spatial,
            spatial_rope=spatial_rope,
            spatial_sequence_length=combined_sequence_length,
            **kwargs,
        )
        # D4: keep only the noise tokens (per-shard first noise_local_len).
        return velocity[:, :noise_local_len]
