# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn

# NOTE: SD35Transformer is the new tt-dit implementation
from models.tt_dit.models.transformers.transformer_sd35 import SD35Checkpoint
from models.tt_dit.models.vae.vae_sd35 import VAEDecoderAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, distribute_cfg
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.stable_diffusion_35_large.text_encoder import TextEncoder
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.tensor import from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

TILE_SIZE = 32

_VAE_SCALE_FACTOR = 8

_DEFAULT_CHECKPOINT = "stabilityai/stable-diffusion-3.5-large"

_PRESETS: dict[tuple[int, ...], dict] = {
    (2, 4): {"cfg": (2, 1), "sp": (2, 0), "tp": (2, 1), "num_links": 1},
    (4, 8): {"cfg": (2, 1), "sp": (4, 0), "tp": (4, 1), "num_links": 4},
}


@dataclass(frozen=True, kw_only=True)
class StableDiffusion3PipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    enable_t5_text_encoder: bool

    height: int
    width: int
    cfg_enabled: bool
    max_t5_sequence_length: int

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
        enable_t5_text_encoder: bool | None = None,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = True,
        max_t5_sequence_length: int = 256,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> StableDiffusion3PipelineConfig:
        preset = _PRESETS.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"])

        # Encoder/VAE always run on a 1x4 (or 4x1) submesh, parallel along the row axis.
        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig.from_tuple((4, 1))
        if vae_parallel_config is None:
            vae_parallel_config = VAEParallelConfig.from_tuple((4, 1))

        # T5 only fits when the submesh has shape (_, 4) — i.e. no reshape needed for CLIP.
        submesh_shape = list(mesh_shape)
        submesh_shape[dit_parallel_config.cfg_parallel.mesh_axis] //= dit_parallel_config.cfg_parallel.factor
        t5_fits = submesh_shape[1] == 4
        if enable_t5_text_encoder is None:
            enable_t5_text_encoder = t5_fits
        elif enable_t5_text_encoder and not t5_fits:
            logger.warning("VAE submesh requires reshape for CLIP, T5 cannot fit on this configuration. Disabling T5.")
            enable_t5_text_encoder = False

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            enable_t5_text_encoder=enable_t5_text_encoder,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            max_t5_sequence_length=max_t5_sequence_length,
            checkpoint_name=checkpoint_name,
        )


class StableDiffusion3Pipeline(PipelineAPIMixin):
    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = True,
        max_t5_sequence_length: int = 256,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> StableDiffusion3Pipeline:
        config = StableDiffusion3PipelineConfig.default(
            mesh_shape=mesh_device.shape,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
            max_t5_sequence_length=max_t5_sequence_length,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: StableDiffusion3PipelineConfig,
    ) -> None:
        self._mesh_device = device
        self.dit_parallel_config = config.dit_parallel_config
        self.encoder_parallel_config = config.encoder_parallel_config
        self.vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self.submesh_devices = create_submeshes(self._mesh_device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self.submesh_devices[0].shape}")

        self.ccl_managers = [
            CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
            for submesh_device in self.submesh_devices
        ]
        self._cfg_combiner = CFGCombiner(self.submesh_devices)

        # Submesh reshape decisions for CLIP/T5 encoder + VAE placement.
        encoder_device = self.submesh_devices[0]
        encoder_shape = tuple(encoder_device.shape)
        if encoder_shape[1] != 4:
            # If reshaping, vae_device must be on submesh 0 (T5 already disabled by config.default).
            vae_submesh_idx = 0
            assert encoder_shape[0] * encoder_shape[1] == 4, f"Cannot reshape {encoder_shape} to a 1x4 mesh"
            self.encoder_mesh_shape = ttnn.MeshShape(1, 4)
        else:
            vae_submesh_idx = 1
            self.encoder_mesh_shape = ttnn.MeshShape(*encoder_shape)
        vae_device = self.submesh_devices[vae_submesh_idx]

        self.encoder_device = encoder_device
        self.vae_device = vae_device
        self.vae_submesh_idx = vae_submesh_idx

        logger.info("creating TT-NN transformer...")
        checkpoint_name = config.checkpoint_name

        checkpoint = SD35Checkpoint(checkpoint_name)
        self.transformers = [
            checkpoint.build(ccl_manager=mgr, parallel_config=self.dit_parallel_config) for mgr in self.ccl_managers
        ]
        self.synchronize_devices()

        self._tracers = [Tracer(self._traced_step, device=submesh, prep_run=False) for submesh in self.submesh_devices]
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self._solvers = [EulerSolver() for _ in self.submesh_devices]

        self._num_channels_latents = checkpoint.num_channels_latents
        self._joint_attention_dim = checkpoint.joint_attention_dim
        self.patch_size = checkpoint.patch_size

        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR)

        with self._reshape_encoder():
            logger.info("creating text encoder...")
            self._text_encoder = TextEncoder(
                checkpoint_name=checkpoint_name,
                device=encoder_device,
                ccl_manager=self.ccl_managers[0],
                parallel_config=self.encoder_parallel_config,
                enable_t5=config.enable_t5_text_encoder,
                joint_attention_dim=self._joint_attention_dim,
                max_t5_sequence_length=config.max_t5_sequence_length,
            )

            logger.info("creating VAE decoder...")
            self._vae = VAEDecoderAdapter(
                checkpoint_name=checkpoint_name,
                parallel_config=self.vae_parallel_config,
                ccl_manager=self.ccl_managers[vae_submesh_idx],
                use_torch=False,
            )

        ttnn.synchronize_device(self.encoder_device)

        logger.info("Pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, guidance_scale=2 if config.cfg_enabled else 1, traced=False)

    def _reshape_encoder(self) -> AbstractContextManager[None]:
        return reshape_device(self.encoder_device, self.encoder_mesh_shape)

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        prompts_2: Sequence[str] | None = None,
        prompts_3: Sequence[str] | None = None,
        negative_prompts: Sequence[str] | None = None,
        negative_prompts_2: Sequence[str] | None = None,
        negative_prompts_3: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 3.5,
        traced: bool = False,
        # currently defaults to off due to ttnn.synchronize_device inside vae_all_gather
        vae_traced: bool | None = False,
        encoder_traced: bool | None = None,
        clip_skip: int | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        on_event = on_event if on_event is not None else null_callback
        prompts_2 = prompts_2 if prompts_2 is not None else prompts
        prompts_3 = prompts_3 if prompts_3 is not None else prompts
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * len(prompts)
        negative_prompts_2 = negative_prompts_2 if negative_prompts_2 is not None else negative_prompts
        negative_prompts_3 = negative_prompts_3 if negative_prompts_3 is not None else negative_prompts

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        if guidance_scale > 1 and not self._cfg_enabled:
            msg = "guidance_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)
        on_event(SectionStart("total"))
        batch_size = len(prompts)
        width = self._width
        height = self._height

        assert height % (_VAE_SCALE_FACTOR * self.patch_size) == 0
        assert width % (_VAE_SCALE_FACTOR * self.patch_size) == 0

        logger.info("encoding prompts...")

        on_event(SectionStart("encoder"))
        with self._reshape_encoder():
            torch_context, torch_pooled = self._text_encoder.encode_cfg(
                (prompts, prompts_2, prompts_3),
                (negative_prompts, negative_prompts_2, negative_prompts_3),
                num_images_per_prompt=num_images_per_prompt,
                cfg_enabled=self._cfg_enabled,
                clip_skip=clip_skip,
                traced=encoder_traced,
                on_event=on_event,
            )
        on_event(SectionEnd("encoder"))

        logger.info("preparing timesteps...")

        self._scheduler.set_timesteps(num_inference_steps)
        sigmas = self._scheduler.sigmas.tolist()
        for solver in self._solvers:
            solver.set_schedule(sigmas)
        timesteps = self._scheduler.timesteps

        logger.info("preparing latents...")

        context = distribute_cfg(torch_context.unsqueeze(1), devices=self.submesh_devices)
        pooled = distribute_cfg(torch_pooled.unsqueeze(1).unsqueeze(1), devices=self.submesh_devices)
        latents = self._random_latents(
            batch_size=batch_size * num_images_per_prompt, dtype=torch_context.dtype, seed=seed
        )

        latents_sequence_length = (height // (_VAE_SCALE_FACTOR * self.patch_size)) * (
            width // (_VAE_SCALE_FACTOR * self.patch_size)
        )

        logger.info("denoising...")

        on_event(SectionStart("denoising"))
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            on_event(SectionStart(f"denoising_step_{i}"))

            velocity_preds = []
            for idx, tracer in enumerate(self._tracers):
                timestep = ttnn.full(
                    [1, 1, 1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self.submesh_devices[idx],
                )

                velocity_preds.append(
                    tracer(
                        cfg_enabled=self._cfg_enabled,
                        submesh_idx=idx,
                        latents=latents[idx],
                        prompt_embed=context[idx] if i == 0 else tracer.inputs["prompt_embed"],
                        pooled_projections=pooled[idx] if i == 0 else tracer.inputs["pooled_projections"],
                        timestep=timestep,
                        N=latents_sequence_length,
                        traced=traced,
                        tracer_blocking_execution=False,
                    )
                )

                # latents can be overwritten by trace execution, use the captured input instead,
                # which is safe.
                latents[idx] = tracer.inputs["latents"]

            if self._cfg_enabled:
                velocity_preds = self._cfg_combiner.combine(velocity_preds, guidance_scale)

            for device in self.submesh_devices:
                ttnn.synchronize_device(device)

            latents = [
                solver.step(step=i, latent=latents[idx], velocity_pred=velocity_preds[idx])
                for idx, solver in enumerate(self._solvers)
            ]

            on_event(SectionEnd(f"denoising_step_{i}"))
        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")

        on_event(SectionStart("vae"))
        output = self._decode_latents(latents[self.vae_submesh_idx], traced=vae_traced)
        on_event(SectionEnd("vae"))
        on_event(SectionEnd("total"))

        return output

    def _traced_step(self, *, cfg_enabled: bool, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if cfg_enabled and not self.dit_parallel_config.cfg_parallel.factor > 1:
            latents = ttnn.concat([latents, latents])

        return self.transformers[submesh_idx](spatial=latents, **kwargs)

    def _random_latents(self, *, batch_size: int, dtype: torch.dtype, seed: int) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)

        latents_shape = (
            batch_size,
            self._height // _VAE_SCALE_FACTOR,
            self._width // _VAE_SCALE_FACTOR,
            self._num_channels_latents,
        )

        latents = torch.randn(latents_shape, dtype=dtype)
        latents = self.transformers[0].patchify(latents)
        return from_torch_to_devices(
            latents,
            devices=self.submesh_devices,
            mesh_axes=[None, None, self.dit_parallel_config.sequence_parallel.mesh_axis, None],
        )

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, traced: bool) -> list[Image.Image]:
        ttnn.synchronize_device(self.vae_device)

        tt_latents = self.ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
            tt_latents, dim=2, mesh_axis=self.dit_parallel_config.sequence_parallel.mesh_axis
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = self.transformers[0].unpatchify(
            torch_latents,
            width=self._width // _VAE_SCALE_FACTOR,
            height=self._height // _VAE_SCALE_FACTOR,
        )

        # Upload + VAE forward + extract-to-host run under the encoder/VAE shape.
        with self._reshape_encoder():
            decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")

        assert isinstance(image, torch.Tensor)
        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

    def synchronize_devices(self) -> None:
        for submesh_device in self.submesh_devices:
            ttnn.synchronize_device(submesh_device)

    def t5_enabled(self) -> bool:
        return self._text_encoder.t5_enabled
