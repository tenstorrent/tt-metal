# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm
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
from models.tt_dit.pipelines.flux1.text_encoder import TextEncoder
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils.tensor import from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

_VAE_SCALE_FACTOR = 16

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (1, 4): {"sp": (1, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 1},
    (2, 4): {"sp": (2, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 1},
    (4, 4): {"sp": (4, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 1},
    (4, 8): {"sp": (4, 0), "tp": (8, 1), "encoder_tp": (4, 0), "vae_tp": (4, 0), "num_links": 4},
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (1, 2): {"sp": (1, 0), "tp": (2, 1), "encoder_tp": (2, 1), "vae_tp": (2, 1), "num_links": 2},
    (2, 2): {"sp": (2, 0), "tp": (2, 1), "encoder_tp": (2, 1), "vae_tp": (2, 1), "num_links": 2},
    (2, 4): {"sp": (2, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 2},
}


@dataclass(frozen=True, kw_only=True)
class Flux1PipelineConfig:
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
    ) -> Flux1PipelineConfig:
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
        )


class Flux1Pipeline(PipelineAPIMixin):
    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = False,
        checkpoint_name: str,
    ) -> Flux1Pipeline:
        config = Flux1PipelineConfig.default(
            mesh_shape=mesh_device.shape,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: Flux1PipelineConfig) -> None:
        self._mesh_device = device
        self._parallel_config = config.dit_parallel_config
        self._encoder_parallel_config = config.encoder_parallel_config
        self._vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._submesh_devices = create_submeshes(self._mesh_device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._submesh_devices[0].shape}")

        self._ccl_managers = [
            CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
            for submesh_device in self._submesh_devices
        ]
        self._cfg_combiner = CFGCombiner(self._submesh_devices)

        self.encoder_device = self._submesh_devices[0]
        self.original_submesh_shape = tuple(self.encoder_device.shape)
        self.vae_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = 0  # Use submesh 0 for VAE

        logger.info("creating TT-NN transformer...")
        checkpoint_name = config.checkpoint_name

        checkpoint = Flux1Checkpoint(checkpoint_name)
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

        logger.info("Pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, cfg_scale=2 if config.cfg_enabled else 1, traced=False)

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 1,  # Flux.1 is not indented to be used with CFG
        guidance_scale: float = 3.5,
        prompts: Sequence[str],
        prompts_2: Sequence[str] | None = None,
        negative_prompts: Sequence[str] | None = None,
        negative_prompts_2: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        traced: bool = False,
        # currently defaults to off due to ttnn.synchronize_device inside vae_all_gather
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
        width = self._width
        height = self._height

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        on_event(SectionStart("total"))
        assert height % (_VAE_SCALE_FACTOR * self._patch_size) == 0
        assert width % (_VAE_SCALE_FACTOR * self._patch_size) == 0

        latents_height = height // _VAE_SCALE_FACTOR
        latents_width = width // _VAE_SCALE_FACTOR
        latents_sequence_length = latents_height * latents_width

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

        logger.info("preparing timesteps...")

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

        logger.info("preparing latents...")

        text_ids = torch.zeros([prompt_sequence_length, 3])
        image_ids = _latent_image_ids(height=latents_height, width=latents_width)
        ids = torch.cat((text_ids, image_ids), dim=0)
        torch_rope_cos, torch_rope_sin = self._pos_embed.forward(ids)

        context = distribute_cfg(torch_context, devices=self._submesh_devices)
        pooled = distribute_cfg(torch_pooled, devices=self._submesh_devices)
        latents = self._random_latents(batch_size=prompt_count * num_images_per_prompt, seed=seed)
        guidance = (
            from_torch_to_devices(torch_guidance.unsqueeze(-1), devices=self._submesh_devices)
            if torch_guidance is not None
            else [None] * len(self._submesh_devices)
        )
        latents_rope_cos = from_torch_to_devices(
            torch_rope_cos[prompt_sequence_length:], devices=self._submesh_devices, mesh_axes=[sp_axis, None]
        )
        latents_rope_sin = from_torch_to_devices(
            torch_rope_sin[prompt_sequence_length:], devices=self._submesh_devices, mesh_axes=[sp_axis, None]
        )
        prompt_rope_cos = from_torch_to_devices(torch_rope_cos[:prompt_sequence_length], devices=self._submesh_devices)
        prompt_rope_sin = from_torch_to_devices(torch_rope_sin[:prompt_sequence_length], devices=self._submesh_devices)

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
                        latents=latents[idx],
                        prompt=context[idx] if i == 0 else inputs["prompt"],
                        pooled=pooled[idx] if i == 0 else inputs["pooled"],
                        timestep=timestep,
                        guidance=guidance[idx] if i == 0 and guidance else inputs["guidance"],
                        spatial_rope=(latents_rope_cos[idx], latents_rope_sin[idx])
                        if i == 0
                        else inputs["spatial_rope"],
                        prompt_rope=(prompt_rope_cos[idx], prompt_rope_sin[idx]) if i == 0 else inputs["prompt_rope"],
                        spatial_sequence_length=latents_sequence_length,
                        prompt_sequence_length=prompt_sequence_length,
                        traced=traced,
                        tracer_blocking_execution=False,
                    )
                )

                # latents can be overwritten by trace execution, use the captured input instead,
                # which is safe.
                latents[idx] = tracer.inputs["latents"]

            if self._cfg_enabled:
                velocity_preds = self._cfg_combiner.combine(velocity_preds, cfg_scale)

            latents = [
                solver.step(step=i, latent=latents[idx], velocity_pred=velocity_preds[idx])
                for idx, solver in enumerate(self._solvers)
            ]

            self.synchronize_devices()  # Helps with accurate time profiling.

            on_event(SectionEnd(f"denoising_step_{i}"))
        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")

        on_event(SectionStart("vae"))
        output = self._decode_latents(latents[self.vae_submesh_idx], traced=vae_traced)
        on_event(SectionEnd("vae"))
        on_event(SectionEnd("total"))

        return output

    def synchronize_devices(self) -> None:
        for submesh_device in self._submesh_devices:
            ttnn.synchronize_device(submesh_device)

    def _random_latents(self, *, batch_size: int, seed: int) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)

        latents_height = self._height // _VAE_SCALE_FACTOR
        latents_width = self._width // _VAE_SCALE_FACTOR
        # We let randn generate a permuted latent tensor, so that the generated noise matches the
        # reference implementation.
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

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, traced: bool) -> list[Image.Image]:
        # Sync because we don't pass a persistent buffer or a barrier semaphore.
        ttnn.synchronize_device(self.vae_device)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
            tt_latents,
            dim=1,
            mesh_axis=sp_axis,
            use_hyperparams=True,
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = _unpack_latents(torch_latents, self._height, self._width, _VAE_SCALE_FACTOR)
        # The adapter expects NHWC; _unpack_latents produces BCHW.
        torch_latents = torch_latents.permute(0, 2, 3, 1)

        decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)
        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

    def _traced_step(self, *, cfg_enabled: bool, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if cfg_enabled and not self._parallel_config.cfg_parallel.factor > 1:
            latents = ttnn.concat([latents, latents])

        return self.transformers[submesh_idx].forward(spatial=latents, **kwargs)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    # B, C, H * P, W * Q -> B, H * W, C * P * Q
    latents = latents.view(batch_size, num_channels_latents, height, 2, width, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, height * width, num_channels_latents * 4)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    # B, H * W, C * P * Q -> B, C, H * P, W * Q
    batch_size, _num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    return latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _latent_image_ids(*, height: int, width: int) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    return latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)


def _calculate_shift(image_seq_len: int, scheduler: FlowMatchEulerDiscreteScheduler) -> float:
    base_seq_len = scheduler.config.get("base_image_seq_len", 256)
    max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
