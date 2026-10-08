# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.models.transformers.transformer_qwenimage import QwenImageCheckpoint
from models.tt_dit.models.vae.vae_wan_2d import WanVaeDecoder2DAdapter, WanVaeEncoder2DAdapter
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, submesh_shape
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.qwenimage.text_encoder import TextEncoder
from models.tt_dit.solvers import EulerSolver, calculate_shift
from models.tt_dit.utils import tensor
from models.tt_dit.utils.padding import torch_pad
from models.tt_dit.utils.tracing import Tracer

_VAE_SCALE_FACTOR = 8
_LATENT_CHANNELS = 16
_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image"

# The prompt lengths the text encoder and the transformer run at, counted after the template prefix
# is dropped.
_SEQUENCE_LENGTH_BUCKETS = (128, 256, 512)

# The supported image sizes, (width, height): the sizes recommended for Qwen-Image and
# Qwen-Image-2512, moved to multiples of 32 px so that the VAE can split them evenly, plus 1024².
_RESOLUTIONS = (
    (1024, 1024),
    (1344, 1344),
    (1664, 928),
    (928, 1664),
    (1472, 1120),
    (1120, 1472),
    (1600, 1056),
    (1056, 1600),
)

# The spatial lengths the transformer runs at. Each image is padded to the smallest that fits it,
# and attention masks the padding.
_SPATIAL_SEQUENCE_LENGTH_BUCKETS = (4096, 7168)

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
    use_torch_vae_encoder: bool

    resolutions: tuple[tuple[int, int], ...]
    spatial_sequence_length_buckets: tuple[int, ...]
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
        use_torch_vae_encoder: bool = False,
        resolutions: Sequence[tuple[int, int]] = _RESOLUTIONS,
        spatial_sequence_length_buckets: Sequence[int] = _SPATIAL_SEQUENCE_LENGTH_BUCKETS,
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
            use_torch_vae_encoder=use_torch_vae_encoder,
            resolutions=tuple(resolutions),
            spatial_sequence_length_buckets=tuple(spatial_sequence_length_buckets),
            cfg_enabled=cfg_enabled,
            sequence_length_buckets=tuple(sequence_length_buckets),
            checkpoint_name=checkpoint_name,
        )


class _InpaintInputs(NamedTuple):
    image_latents: ttnn.Tensor
    noise: ttnn.Tensor
    mask: ttnn.Tensor


class _ImageSize(NamedTuple):
    width: int
    height: int
    latents_width: int
    latents_height: int
    sequence_length: int
    padded_sequence_length: int


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
        cfg_enabled: bool = True,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipeline:
        config = QwenImagePipelineConfig.default(
            mesh_shape=mesh_device.shape,
            cfg_enabled=cfg_enabled,
            checkpoint_name=checkpoint_name,
        )
        return cls(device=mesh_device, config=config)

    def __init__(self, *, device: ttnn.MeshDevice, config: QwenImagePipelineConfig) -> None:
        self._parallel_config = config.dit_parallel_config
        self._sp_axis = config.dit_parallel_config.sequence_parallel.mesh_axis
        self._cfg_parallel = config.dit_parallel_config.cfg_parallel.factor != 1
        self._cfg_enabled = config.cfg_enabled
        self._sequence_length_buckets = config.sequence_length_buckets
        self._vae_parallel_config = config.vae_parallel_config
        self._spatial_sequence_length_buckets = sorted(set(config.spatial_sequence_length_buckets))

        sp_factor = config.dit_parallel_config.sequence_parallel.factor
        for bucket in self._spatial_sequence_length_buckets:
            # Ring attention needs whole tiles on each device.
            if bucket % (ttnn.TILE_SIZE * sp_factor) != 0:
                msg = f"spatial bucket {bucket} must be a multiple of {ttnn.TILE_SIZE * sp_factor}"
                raise ValueError(msg)

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
        # One tracer per submesh, prompt bucket and spatial bucket.
        self._tracers = {
            (idx, bucket, spatial_bucket): Tracer(self._traced_step, device=d, prep_run=False)
            for idx, d in enumerate(self._devices)
            for bucket in config.sequence_length_buckets
            for spatial_bucket in self._spatial_sequence_length_buckets
        }
        self._image_processor = VaeImageProcessor(vae_scale_factor=_VAE_SCALE_FACTOR * 2)
        self._mask_processor = VaeImageProcessor(
            vae_scale_factor=_VAE_SCALE_FACTOR * 2,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

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
        self._image_sizes = {(w, h): self._image_size(w, h) for w, h in config.resolutions}
        self._transformers = [
            self._checkpoint.build(ccl_manager=m, parallel_config=config.dit_parallel_config, is_fsdp=False)
            for m in self._ccl_managers
        ]

        logger.info("creating VAE decoder...")
        self._vae = WanVaeDecoder2DAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._ccl_managers[-1],
            use_torch=config.use_torch_vae_decoder,
        )

        logger.info("creating VAE encoder...")
        self._vae_encoder = WanVaeEncoder2DAdapter(
            checkpoint_name=config.checkpoint_name,
            parallel_config=config.vae_parallel_config,
            ccl_manager=self._ccl_managers[-1],
            use_torch=config.use_torch_vae_encoder,
        )

        # Allocated before the traces: the inpainting inputs of each spatial bucket and submesh,
        # patchified and padded like the latents, and the spatial length without padding of each
        # submesh.
        p = self._checkpoint.patch_size
        mesh_axes = [None, self._sp_axis, None]
        self._inpaint_inputs: dict[int, list[_InpaintInputs]] = {}
        for bucket in self._spatial_sequence_length_buckets:
            shape = [1, bucket, _LATENT_CHANNELS * p * p]
            self._inpaint_inputs[bucket] = [
                _InpaintInputs(
                    image_latents=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                    noise=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                    mask=tensor.zeros(shape, device=d, mesh_axes=mesh_axes),
                )
                for d in self._devices
            ]
        self._sequence_lengths = [self._sequence_length_tensor(0, device=d) for d in self._devices]

        logger.info("pipeline allocation run...")
        self._warm_up(traced=False)

        logger.info("pipeline capture run...")
        self._warm_up(traced=True)

    def _warm_up(self, *, traced: bool) -> None:
        """Runs every prompt bucket with every spatial bucket, and every resolution."""
        covered_spatial_buckets = set()

        for size in self._image_sizes.values():
            image = Image.new("RGB", (size.width, size.height))
            mask_image = Image.new("L", (size.width, size.height), 255)

            # The VAE needs each resolution, but the transformer only each spatial bucket.
            prompt_buckets = sorted(self._sequence_length_buckets, reverse=True)
            if size.padded_sequence_length in covered_spatial_buckets:
                prompt_buckets = prompt_buckets[-1:]
            covered_spatial_buckets.add(size.padded_sequence_length)

            for bucket in prompt_buckets:
                # Each "a " is one token, and the trailing space and the template suffix add six
                # more, so this comes to two tokens short of the bucket, which the 32-token spacing
                # keeps above the next-smaller one.
                prompt = "a " * (bucket - 8)
                self(
                    prompts=[prompt],
                    negative_prompts=[prompt],
                    num_inference_steps=2,
                    cfg_scale=2 if self._cfg_enabled else 1,
                    width=size.width,
                    height=size.height,
                    traced=traced,
                    encoder_traced=traced,
                    vae_traced=traced,
                    image=image,
                    mask_image=mask_image,
                    strength=1.0,
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
        width: int = 1024,
        height: int = 1024,
        traced: bool = True,
        vae_traced: bool | None = None,
        encoder_traced: bool | None = None,
        image: Image.Image | None = None,
        mask_image: Image.Image | None = None,
        strength: float = 0.6,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        """Generates an image from the prompt.

        ``width`` and ``height`` must be one of the configured resolutions.

        Inpainting: pass ``image`` and ``mask_image``. The white part of the mask is regenerated,
        the rest is kept. ``strength`` sets how much the white part may change: 1 ignores what was
        there, lower values stay closer to it.
        """
        prompt_count = len(prompts)

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        size = self._image_sizes.get((width, height))
        if size is None:
            msg = f"unsupported size {width}x{height}, supported are {list(self._image_sizes)}"
            raise ValueError(msg)

        if (image is None) != (mask_image is None):
            msg = "inpainting needs both image and mask_image"
            raise ValueError(msg)
        inpaint = image is not None

        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * prompt_count

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

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
        mu = calculate_shift(size.sequence_length, self._solvers[0].scheduler)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        for solver in self._solvers:
            solver.set_schedule(sigmas=sigmas, mu=mu)
        timesteps = self._solvers[0].timesteps

        # Inpainting skips the first steps of the schedule, by the same rounding as diffusers.
        first_step = int(max(num_inference_steps - min(num_inference_steps * strength, num_inference_steps), 0))
        if inpaint and first_step >= num_inference_steps:
            msg = f"strength {strength} leaves no denoising steps"
            raise ValueError(msg)
        first_step = first_step if inpaint else 0

        logger.info("preparing inputs...")
        context = [tensor.from_torch(c, device=d) for c, d in zip(torch_contexts, self._devices, strict=True)]
        noise = self._random_latents(size, batch_size=prompt_count * num_images_per_prompt, seed=seed, inpaint=inpaint)
        if inpaint:
            noise = self._prepare_inpainting(
                image, mask_image, size=size, noise=noise, first_step=first_step, traced=vae_traced
            )
        latents = tensor.from_torch_to_devices(noise, devices=self._devices, mesh_axes=[None, self._sp_axis, None])
        ropes = [
            self._checkpoint.rope_tables(
                latents_height=size.latents_height,
                latents_width=size.latents_width,
                prompt_sequence_length=length,
                padded_spatial_sequence_length=size.padded_sequence_length,
                device=device,
                sp_axis=self._sp_axis,
            )
            for length, device in zip(prompt_sequence_lengths, self._devices, strict=True)
        ]
        for d, sequence_length in zip(self._devices, self._sequence_lengths, strict=True):
            host = self._sequence_length_tensor(size.sequence_length, device=d, on_host=True)
            ttnn.copy_host_to_device_tensor(host, sequence_length)
        tracers = [
            self._tracers[idx, length, size.padded_sequence_length]
            for idx, length in enumerate(prompt_sequence_lengths)
        ]

        logger.info("denoising...")
        on_event(SectionStart("denoising"))

        for step, t in enumerate(tqdm.tqdm(timesteps[first_step:]), start=first_step):
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
                        prompt=context[idx] if step == first_step else tracer.inputs["prompt"],
                        timestep=timestep,
                        spatial_rope=spatial_rope if step == first_step else tracer.inputs["spatial_rope"],
                        prompt_rope=prompt_rope if step == first_step else tracer.inputs["prompt_rope"],
                        spatial_sequence_length=self._sequence_lengths[idx],
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

            if inpaint:
                # Outside the mask, the latents are replaced by the image latents, noised to the
                # level of the next step.
                sigma_next = self._solvers[0].sigmas[step + 1]
                for idx, inputs in enumerate(self._inpaint_inputs[size.padded_sequence_length]):
                    known = inputs.noise * sigma_next + inputs.image_latents * (1 - sigma_next)
                    latents[idx] = known + inputs.mask * (latents[idx] - known)

            self.synchronize_devices()  # for time profiling
            on_event(SectionEnd(f"denoising_step_{step}"))

        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")
        on_event(SectionStart("vae"))
        images = self._decode_latents(latents[-1], size=size, traced=vae_traced)
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

    def _image_size(self, width: int, height: int) -> _ImageSize:
        p = self._checkpoint.patch_size
        latents_width = width // _VAE_SCALE_FACTOR
        latents_height = height // _VAE_SCALE_FACTOR

        # The VAE splits the latents evenly across the devices.
        h_factor = self._vae_parallel_config.height_parallel.factor
        w_factor = self._vae_parallel_config.width_parallel.factor
        if (
            width % (_VAE_SCALE_FACTOR * p) != 0
            or height % (_VAE_SCALE_FACTOR * p) != 0
            or latents_height % h_factor != 0
            or latents_width % w_factor != 0
        ):
            msg = (
                f"size {width}x{height} must be a multiple of {_VAE_SCALE_FACTOR * p}, with latents divisible "
                f"by {h_factor} in height and {w_factor} in width"
            )
            raise ValueError(msg)

        sequence_length = (latents_width // p) * (latents_height // p)

        buckets = [b for b in self._spatial_sequence_length_buckets if b >= sequence_length]
        if not buckets:
            msg = f"size {width}x{height} exceeds the largest spatial bucket"
            raise ValueError(msg)

        return _ImageSize(
            width=width,
            height=height,
            latents_width=latents_width,
            latents_height=latents_height,
            sequence_length=sequence_length,
            padded_sequence_length=buckets[0],
        )

    @staticmethod
    def _sequence_length_tensor(length: int, *, device: ttnn.MeshDevice, on_host: bool = False) -> ttnn.Tensor:
        return tensor.from_torch(
            torch.tensor([length]).reshape(1, 1, 1, 1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            on_host=on_host,
        )

    def _patchify(self, latents: torch.Tensor, size: _ImageSize) -> torch.Tensor:
        """Patchifies latents of shape (B, H, W, C) and pads them to the spatial bucket."""
        spatial = self._transformers[0].patchify(latents)
        return torch_pad(spatial, size.padded_sequence_length - size.sequence_length, dim=1)

    def _random_latents(self, size: _ImageSize, *, batch_size: int, seed: int, inpaint: bool) -> torch.Tensor:
        torch.manual_seed(seed)

        shape = [batch_size, _LATENT_CHANNELS, size.latents_height, size.latents_width]

        if inpaint:
            # The reference samples the VAE latent distribution of the image before drawing the
            # noise. Its spread is negligible, so the image is encoded to the mean, but the draw is
            # repeated so that the noise matches.
            torch.randn(shape, dtype=torch.bfloat16)

        # Noise is drawn in bfloat16 and channels first like in the reference implementation.
        noise = torch.randn(shape, dtype=torch.bfloat16)
        return self._patchify(noise.permute(0, 2, 3, 1).float(), size)

    def _prepare_inpainting(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        *,
        size: _ImageSize,
        noise: torch.Tensor,
        first_step: int,
        traced: bool,
    ) -> torch.Tensor:
        """Fill the inpainting inputs and return the image latents noised to the first step."""
        image_latents = self._encode_image(image, size=size, traced=traced)
        mask = self._latent_mask(mask_image, size=size)

        for d, buffers in zip(self._devices, self._inpaint_inputs[size.padded_sequence_length], strict=True):
            for source, buffer in (
                (image_latents, buffers.image_latents),
                (noise, buffers.noise),
                (mask, buffers.mask),
            ):
                host = tensor.from_torch(
                    source,
                    device=d,
                    mesh_axes=[None, self._sp_axis, None],
                    on_host=True,
                )
                ttnn.copy_host_to_device_tensor(host, buffer)

        sigma = self._solvers[0].sigmas[first_step]
        return sigma * noise + (1 - sigma) * image_latents

    def _encode_image(self, image: Image.Image, *, size: _ImageSize, traced: bool) -> torch.Tensor:
        pixels = self._image_processor.preprocess(image, height=size.height, width=size.width)
        latents = self._vae_encoder.encode(pixels.to(torch.float32), traced=traced)
        return self._patchify(latents, size)

    def _latent_mask(self, mask_image: Image.Image, *, size: _ImageSize) -> torch.Tensor:
        mask = self._mask_processor.preprocess(mask_image, height=size.height, width=size.width)
        mask = torch.nn.functional.interpolate(mask, size=(size.latents_height, size.latents_width))
        return self._patchify(mask.repeat(1, _LATENT_CHANNELS, 1, 1).permute(0, 2, 3, 1), size)

    def _decode_latents(self, tt_latents: ttnn.Tensor, *, size: _ImageSize, traced: bool) -> list[Image.Image]:
        # Sync because we don't pass a persistent buffer or a barrier semaphore.
        ttnn.synchronize_device(self._devices[-1])

        tt_latents = self._ccl_managers[-1].all_gather_persistent_buffer(
            tt_latents, dim=1, mesh_axis=self._sp_axis, use_hyperparams=True
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])[:, : size.sequence_length]
        torch_latents = self._transformers[0].unpatchify(
            torch_latents, height=size.latents_height, width=size.latents_width
        )

        decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)

        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))
