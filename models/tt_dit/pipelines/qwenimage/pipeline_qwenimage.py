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
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageCheckpoint
from models.tt_dit.models.vae.vae_wan_2d import WanVaeDecoder2DAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, submesh_shape
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.qwenimage.text_encoder import TextEncoder
from models.tt_dit.solvers import EulerSolver, calculate_shift
from models.tt_dit.utils.tensor import from_torch, from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

_VAE_SCALE_FACTOR = 8
_LATENT_CHANNELS = 16
_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image"

# The prompt lengths the text encoder and the transformer run at, counted after the template prefix
# is dropped.
_SEQUENCE_LENGTH_BUCKETS = (128, 256, 512)

_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (1, 0),
        "sp": (2, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": (2, 0),
        "num_links": 1,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": (4, 0),
        "num_links": 4,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 2): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (2, 1),
        "encoder_tp": (2, 1),
        "encoder_fsdp": None,
        "num_links": 1,
    },
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": None,
        "num_links": 1,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "encoder_fsdp": (4, 0),
        "num_links": 4,
    },
}


@dataclass(frozen=True, kw_only=True)
class QwenImagePipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VaeHWParallelConfig

    use_torch_text_encoder: bool
    use_torch_vae_decoder: bool

    height: int
    width: int
    cfg_enabled: bool
    sequence_length_buckets: tuple[int, ...]

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
        vae_parallel_config: VaeHWParallelConfig | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = True,
        sequence_length_buckets: Sequence[int] = _SEQUENCE_LENGTH_BUCKETS,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"])

        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig.from_tuples(
                tp=preset["encoder_tp"], sp=None, fsdp=preset["encoder_fsdp"]
            )

        if vae_parallel_config is None:
            vae_parallel_config = VaeHWParallelConfig.from_axes(submesh_shape(dit_parallel_config), h_axis=1, w_axis=0)

        return cls(
            topology=topology,
            num_links=num_links if num_links is not None else preset["num_links"],
            dit_parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            sequence_length_buckets=tuple(sequence_length_buckets),
            checkpoint_name=checkpoint_name,
        )


class QwenImagePipeline(PipelineAPIMixin):
    """QwenImagePipeline is a pipeline for generating images from text prompts.

    It uses a transformer to encode the text prompts and a VAE to decode the latent space.
    Dynamic loading is controlled by the initialization state. During inference, modules
    will be loaded/offloaded as needed.
    """

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        width: int = 1024,
        height: int = 1024,
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipeline:
        config = QwenImagePipelineConfig.default(
            mesh_shape=mesh_device.shape,
            width=width,
            height=height,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: QwenImagePipelineConfig) -> None:
        self._parallel_config = config.dit_parallel_config
        self._sp_axis = config.dit_parallel_config.sequence_parallel.mesh_axis
        self._cfg_parallel = config.dit_parallel_config.cfg_parallel.factor != 1
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled
        self._sequence_length_buckets = config.sequence_length_buckets

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._devices = create_submeshes(device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._devices[0].shape}")

        self._ccl_managers = [
            CCLManager(d, num_links=config.num_links, topology=config.topology) for d in self._devices
        ]
        self._combiner = CFGCombiner(self._devices)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config.checkpoint_name, subfolder="scheduler")
        self._solvers = [EulerSolver(scheduler=scheduler) for _ in self._devices]
        # One tracer per submesh and prompt bucket.
        self._tracers = {
            (idx, bucket): Tracer(self._traced_step, device=d, prep_run=False)
            for idx, d in enumerate(self._devices)
            for bucket in config.sequence_length_buckets
        }
        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR * 2)

        # The encoder is loaded before the transformers, for memory efficiency.
        logger.info("creating text encoder...")
        self._text_encoder = TextEncoder(
            checkpoint_name=config.checkpoint_name,
            device=self._devices[0],
            ccl_manager=self._ccl_managers[0],
            parallel_config=config.encoder_parallel_config,
            sequence_length_buckets=config.sequence_length_buckets,
            use_torch=config.use_torch_text_encoder,
        )

        logger.info("creating transformers...")
        self._checkpoint = QwenImageCheckpoint(config.checkpoint_name)
        self._transformers = []
        for m in self._ccl_managers:
            transformer = self._checkpoint.build(
                ccl_manager=m, parallel_config=config.dit_parallel_config, is_fsdp=False
            )
            self._checkpoint.load(
                transformer, mesh_device=m.mesh_device, parallel_config=config.dit_parallel_config, is_fsdp=False
            )
            self._transformers.append(transformer)

        logger.info("creating VAE decoder...")
        self._vae = WanVaeDecoder2DAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._ccl_managers[-1],
            use_torch=config.use_torch_vae_decoder,
        )

        logger.info("pipeline allocation run...")
        self._warm_up(traced=False)

        logger.info("pipeline capture run...")
        self._warm_up(traced=True)

    def _warm_up(self, *, traced: bool) -> None:
        # Only the transformer is captured here. The text encoder and the VAE decoder are captured
        # on their first traced call.
        for bucket in sorted(self._sequence_length_buckets, reverse=True):
            # Each "a " is one token, and the trailing space and the template suffix add six more, so
            # this comes to two tokens short of the bucket, which the 32-token spacing keeps above
            # the next-smaller one.
            prompt = "a " * (bucket - 8)
            self(
                prompts=[prompt],
                negative_prompts=[prompt],
                num_inference_steps=2,
                cfg_scale=2 if self._cfg_enabled else 1,
                traced=traced,
                encoder_traced=False,
                vae_traced=False,
            )

    def __call__(
        self,
        *,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 4.0,
        traced: bool = True,
        vae_traced: bool | None = False,
        encoder_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * prompt_count

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // _VAE_SCALE_FACTOR
        latents_width = self._width // _VAE_SCALE_FACTOR
        p = self._checkpoint.patch_size
        latents_sequence_length = (latents_height // p) * (latents_width // p)

        on_event(SectionStart("total"))

        logger.info("encoding prompts...")
        on_event(SectionStart("encoder"))
        # One entry per submesh: the CFG pass it runs, or both passes batched on the only one.
        torch_contexts = self._text_encoder.encode_cfg(
            prompts,
            negative_prompts,
            num_images_per_prompt=num_images_per_prompt,
            cfg_enabled=self._cfg_enabled,
            batch_passes=not self._cfg_parallel,
            on_event=on_event,
            traced=encoder_traced,
        )
        on_event(SectionEnd("encoder"))
        prompt_sequence_lengths = [c.shape[1] for c in torch_contexts]

        logger.info("preparing timesteps...")
        mu = calculate_shift(latents_sequence_length, self._solvers[0].scheduler)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        for solver in self._solvers:
            solver.set_schedule(sigmas=sigmas, mu=mu)
        timesteps = self._solvers[0].timesteps

        logger.info("preparing inputs...")
        context = [from_torch(c, device=d) for c, d in zip(torch_contexts, self._devices, strict=True)]
        latents = self._random_latents(batch_size=prompt_count * num_images_per_prompt, seed=seed)
        ropes = [
            self._checkpoint.rope_tables(
                latents_height=latents_height,
                latents_width=latents_width,
                prompt_sequence_length=length,
                device=device,
                sp_axis=self._sp_axis,
            )
            for length, device in zip(prompt_sequence_lengths, self._devices, strict=True)
        ]
        tracers = [self._tracers[idx, length] for idx, length in enumerate(prompt_sequence_lengths)]

        logger.info("denoising...")
        on_event(SectionStart("denoising"))

        for step, t in enumerate(tqdm.tqdm(timesteps)):
            on_event(SectionStart(f"denoising_step_{step}"))

            velocity_preds = []
            for idx, tracer in enumerate(tracers):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self._devices[idx],
                )
                spatial_rope, prompt_rope = ropes[idx]

                velocity_preds.append(
                    tracer(
                        submesh_idx=idx,
                        latents=latents[idx],
                        prompt=context[idx] if step == 0 else tracer.inputs["prompt"],
                        timestep=timestep,
                        spatial_rope=spatial_rope if step == 0 else tracer.inputs["spatial_rope"],
                        prompt_rope=prompt_rope if step == 0 else tracer.inputs["prompt_rope"],
                        spatial_sequence_length=latents_sequence_length,
                        prompt_sequence_length=prompt_sequence_lengths[idx],
                        traced=traced,
                        tracer_blocking_execution=False,
                    )
                )

                # latents can be overwritten by trace execution, use the captured input instead,
                # which is safe.
                latents[idx] = tracer.inputs["latents"]

            if self._cfg_enabled:
                velocity_preds = self._combiner.combine(velocity_preds, cfg_scale)

            latents = [
                solver.step(step=step, latent=latents[idx], velocity_pred=velocity_preds[idx])
                for idx, solver in enumerate(self._solvers)
            ]

            self.synchronize_devices()  # for time profiling
            on_event(SectionEnd(f"denoising_step_{step}"))

        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        images = self._decode_latents(latents[-1], traced=vae_traced)
        on_event(SectionEnd("vae"))

        on_event(SectionEnd("total"))
        return images

    def _traced_step(self, *, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if self._cfg_enabled and not self._cfg_parallel:
            latents = ttnn.concat([latents, latents])

        return self._transformers[submesh_idx].forward(spatial=latents, **kwargs)

    def synchronize_devices(self) -> None:
        for d in self._devices:
            ttnn.synchronize_device(d)

    def _random_latents(self, *, batch_size: int, seed: int) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)

        shape = [
            batch_size,
            _LATENT_CHANNELS,
            self._height // _VAE_SCALE_FACTOR,
            self._width // _VAE_SCALE_FACTOR,
        ]

        # We let randn generate a permuted latent tensor in float32, so that the generated noise
        # matches the reference implementation.
        latents = self._transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))

        return from_torch_to_devices(latents, devices=self._devices, mesh_axes=[None, self._sp_axis, None])

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, traced: bool) -> list[Image.Image]:
        # Sync because we don't pass a persistent buffer or a barrier semaphore.
        ttnn.synchronize_device(self._devices[-1])

        tt_latents = self._ccl_managers[-1].all_gather_persistent_buffer(
            tt_latents, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = self._transformers[0].unpatchify(
            torch_latents,
            height=self._height // _VAE_SCALE_FACTOR,
            width=self._width // _VAE_SCALE_FACTOR,
        )

        decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))
