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
from models.tt_dit.models.vae.vae_qwenimage import QwenImageVAEDecoderAdapter
from models.tt_dit.parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VaeHWParallelConfig,
    VAEParallelConfig,
)
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import CFGCombiner, create_submeshes, distribute_cfg
from models.tt_dit.pipelines.events import PipelineEventCallback, SectionEnd, SectionStart, null_callback
from models.tt_dit.pipelines.pipeline_api import PipelineAPIMixin
from models.tt_dit.pipelines.qwenimage.text_encoder import TextEncoder
from models.tt_dit.solvers import EulerSolver
from models.tt_dit.utils import cache
from models.tt_dit.utils.mesh import reshape_device
from models.tt_dit.utils.tensor import from_torch_to_devices
from models.tt_dit.utils.tracing import Tracer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    from PIL import Image

_DEFAULT_CHECKPOINT = "Qwen/Qwen-Image"

# The encoder is currently hardcoded to always be FSDP as it is the most memory efficient
# configuration with little to no performance penalty.
_PRESETS_WH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
        "is_fsdp": False,
        "dynamic_load_encoder": True,
        "dynamic_load_vae": True,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 4,
        "is_fsdp": False,
        "dynamic_load_encoder": False,
        "dynamic_load_vae": False,
    },
}

_PRESETS_BH: dict[tuple[int, ...], dict] = {
    (2, 4): {
        "cfg": (2, 0),
        "sp": (1, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 1,
        "is_fsdp": False,
        "dynamic_load_encoder": True,
        "dynamic_load_vae": False,
    },
    (4, 8): {
        "cfg": (2, 1),
        "sp": (4, 0),
        "tp": (4, 1),
        "encoder_tp": (4, 1),
        "vae_tp": (4, 1),
        "num_links": 4,
        "is_fsdp": False,
        "dynamic_load_encoder": False,
        "dynamic_load_vae": False,
    },
}


@dataclass(frozen=True, kw_only=True)
class QwenImagePipelineConfig:
    topology: ttnn.Topology
    num_links: int

    dit_parallel_config: DiTParallelConfig
    encoder_parallel_config: EncoderParallelConfig
    vae_parallel_config: VAEParallelConfig

    use_torch_text_encoder: bool
    use_torch_vae_decoder: bool

    height: int
    width: int
    cfg_enabled: bool

    is_fsdp: bool
    dynamic_load_encoder: bool
    dynamic_load_vae: bool

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
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        height: int = 1024,
        width: int = 1024,
        cfg_enabled: bool = True,
        is_fsdp: bool | None = None,
        dynamic_load_encoder: bool | None = None,
        dynamic_load_vae: bool | None = None,
        checkpoint_name: str = _DEFAULT_CHECKPOINT,
    ) -> QwenImagePipelineConfig:
        preset_dict = _PRESETS_BH if ttnn.device.is_blackhole() else _PRESETS_WH
        preset = preset_dict.get(tuple(mesh_shape), {})

        if dit_parallel_config is None:
            dit_parallel_config = DiTParallelConfig.from_tuples(cfg=preset["cfg"], sp=preset["sp"], tp=preset["tp"])

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
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            height=height,
            width=width,
            cfg_enabled=cfg_enabled,
            is_fsdp=is_fsdp if is_fsdp is not None else preset["is_fsdp"],
            dynamic_load_encoder=(
                dynamic_load_encoder if dynamic_load_encoder is not None else preset["dynamic_load_encoder"]
            ),
            dynamic_load_vae=dynamic_load_vae if dynamic_load_vae is not None else preset["dynamic_load_vae"],
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

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: QwenImagePipelineConfig,
    ) -> None:
        if config.dynamic_load_encoder or config.dynamic_load_vae:
            assert cache.cache_dir_is_set(), (
                "Dynamic loading of encoder or vae is enabled but the cache directory "
                "(env variable TT_DIT_CACHE_DIR) is not set."
            )

        self._mesh_device = device
        self._parallel_config = config.dit_parallel_config
        self._encoder_parallel_config = config.encoder_parallel_config
        self._vae_parallel_config = config.vae_parallel_config
        self._height = config.height
        self._width = config.width
        self._cfg_enabled = config.cfg_enabled
        self._is_fsdp = config.is_fsdp
        self._checkpoint_name = config.checkpoint_name

        logger.info(f"Parallel config: {config.dit_parallel_config}")
        logger.info(f"Original mesh shape: {device.shape}")
        self._submesh_devices = create_submeshes(self._mesh_device, config.dit_parallel_config)
        logger.info(f"Created submeshes with shape {self._submesh_devices[0].shape}")

        self._ccl_managers = [
            CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
            for submesh_device in self._submesh_devices
        ]
        self._cfg_combiner = CFGCombiner(self._submesh_devices)

        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = len(self._submesh_devices) - self.encoder_submesh_idx - 1  # Use other submesh for VAE. 0

        self.encoder_device = self._submesh_devices[self.encoder_submesh_idx]
        self.vae_device = self._submesh_devices[self.vae_submesh_idx]

        self._wan_vae_parallel_config = self.get_wan_vae_parallel_config()

        self.encoder_mesh_shape = self.get_mesh_shape(
            self.encoder_device, self._encoder_parallel_config.tensor_parallel
        )
        self.vae_mesh_shape = self.get_mesh_shape(self.vae_device, self._vae_parallel_config.tensor_parallel)

        logger.info("loading models...")

        self._checkpoint = QwenImageCheckpoint(self._checkpoint_name)

        self._num_channels_latents = 16
        self._patch_size = self._checkpoint.patch_size
        self._vae_scale_factor = 8
        self._pos_embed = self._checkpoint.pos_embed

        # Initialize the transformers. Weight loading is deferred (see _load_transformers).
        self.transformers = [
            self._checkpoint.build(
                ccl_manager=mgr,
                parallel_config=self._parallel_config,
                is_fsdp=self._is_fsdp,
            )
            for mgr in self._ccl_managers
        ]
        self._tracers = [Tracer(self._traced_step, device=device, prep_run=False) for device in self._submesh_devices]
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self._checkpoint_name, subfolder="scheduler")
        self._solvers = [EulerSolver() for _ in self._submesh_devices]
        self._transformers_loaded = False

        # initialize text encoder. This will load the weights
        self._use_torch_text_encoder = config.use_torch_text_encoder
        with self._reshape_encoder():
            logger.info("creating text encoder (loading before transformers for memory efficiency)...")
            self._text_encoder = TextEncoder(
                checkpoint_name=self._checkpoint_name,
                device=self._submesh_devices[self.encoder_submesh_idx],
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=self._encoder_parallel_config,
                use_torch=config.use_torch_text_encoder,
            )
        ttnn.synchronize_device(self.encoder_device)

        # Encoder is already loaded. Decide if we should also load the transformers.
        if (
            not config.dynamic_load_encoder or config.use_torch_text_encoder
        ):  # Implies we have enough space. VAE comes after denoising, so load all transformers now.
            self._load_transformers(self.encoder_submesh_idx)

        # Always load transformers for vae since it comes before VAE
        self._load_transformers(self.vae_submesh_idx)

        self._image_processor = VaeImageProcessor(vae_scale_factor=2 * self._vae_scale_factor)

        self._use_torch_vae_decoder = config.use_torch_vae_decoder

        with self._reshape_vae():
            logger.info("creating VAE decoder...")
            self._vae = QwenImageVAEDecoderAdapter(
                checkpoint_name=self._checkpoint_name,
                parallel_config=self._wan_vae_parallel_config,
                ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                use_torch=config.use_torch_vae_decoder,
            )
        ttnn.synchronize_device(self.vae_device)

        # Load VAE weights based on configuration
        if not config.use_torch_vae_decoder and not config.dynamic_load_vae:
            self._vae.reload_weights()

        logger.info("Pipeline allocation run...")
        self(prompts=[""], num_inference_steps=2, cfg_scale=2 if config.cfg_enabled else 1, traced=False)

    def _load_transformers(self, idx: int) -> None:
        """Load transformer weights to device. Called lazily for device encoder path."""
        if self.transformers[idx].is_loaded():
            return

        self._checkpoint.load(
            self.transformers[idx],
            mesh_device=self._submesh_devices[idx],
            parallel_config=self._parallel_config,
            is_fsdp=self._is_fsdp,
        )

        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_transformers(self, idx: int) -> None:
        """Deallocate transformer weights from device to free memory."""
        if not self.transformers[idx].is_loaded():
            return

        logger.info("deallocating transformer weights to free memory...")
        self.transformers[idx].deallocate_weights()
        ttnn.synchronize_device(self._submesh_devices[idx])

    @staticmethod
    def get_mesh_shape(mesh_device: ttnn.MeshDevice, parallel_factor: ParallelFactor) -> ttnn.MeshShape:
        mesh_shape = list(mesh_device.shape)
        mesh_shape[parallel_factor.mesh_axis] = parallel_factor.factor
        mesh_shape[1 - parallel_factor.mesh_axis] = mesh_device.shape.mesh_size() // parallel_factor.factor
        return ttnn.MeshShape(tuple(mesh_shape))

    # TODO: Configure the correct parallel config
    def get_wan_vae_parallel_config(self) -> VaeHWParallelConfig:
        return VaeHWParallelConfig(
            height_parallel=ParallelFactor(
                factor=self.vae_device.shape[self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
            width_parallel=ParallelFactor(
                factor=self.vae_device.shape[1 - self._vae_parallel_config.tensor_parallel.mesh_axis],
                mesh_axis=1 - self._vae_parallel_config.tensor_parallel.mesh_axis,
            ),
        )

    def prepare_encoder(self) -> None:
        """Prepare encoder for inference."""
        if not self._text_encoder.encoder_loaded():
            self._deallocate_transformers(self.encoder_submesh_idx)
            with self._reshape_encoder():
                self._text_encoder.reload_encoder_weights()

    def prepare_transformers(self) -> None:
        if not self.transformers[self.encoder_submesh_idx].is_loaded():
            self._text_encoder.deallocate_encoder_weights()
            self._load_transformers(self.encoder_submesh_idx)

        if not self.transformers[self.vae_submesh_idx].is_loaded():
            if not self._use_torch_vae_decoder and self._vae.is_loaded():
                logger.info("deallocating VAE decoder weights to free memory...")
                self._vae.deallocate_weights()
                ttnn.synchronize_device(self.vae_device)
            self._load_transformers(self.vae_submesh_idx)

    def prepare_vae(self) -> None:
        if self._vae.is_loaded():
            return
        self._deallocate_transformers(self.vae_submesh_idx)
        with self._reshape_vae():
            logger.info("loading VAE decoder weights to device...")
            self._vae.reload_weights()
        ttnn.synchronize_device(self.vae_device)

    def _reshape_encoder(self) -> AbstractContextManager[None]:
        return reshape_device(self.encoder_device, self.encoder_mesh_shape)

    def _reshape_vae(self) -> AbstractContextManager[None]:
        return reshape_device(self.vae_device, self.vae_mesh_shape)

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float = 4.0,
        prompts: Sequence[str],
        negative_prompts: Sequence[str] | None = None,
        num_inference_steps: int,
        seed: int = 0,
        traced: bool = False,
        vae_traced: bool | None = False,
        encoder_traced: bool | None = None,
        on_event: PipelineEventCallback | None = None,
    ) -> list[Image.Image]:
        on_event = on_event if on_event is not None else null_callback
        negative_prompts = negative_prompts if negative_prompts is not None else [""] * len(prompts)
        vae_traced = vae_traced if vae_traced is not None else traced
        encoder_traced = encoder_traced if encoder_traced is not None else traced
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        if cfg_scale > 1 and not self._cfg_enabled:
            msg = "cfg_scale > 1 requires CFG to be enabled"
            raise ValueError(msg)

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        latents_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        on_event(SectionStart("total"))
        logger.info("encoding prompts...")

        self.prepare_encoder()

        on_event(SectionStart("encoder"))
        with self._reshape_encoder():
            torch_context, _prompt_mask = self._text_encoder.encode_cfg(
                prompts,
                negative_prompts,
                num_images_per_prompt=num_images_per_prompt,
                cfg_enabled=self._cfg_enabled,
                on_event=on_event,
                traced=encoder_traced,
            )
        on_event(SectionEnd("encoder"))
        _, prompt_sequence_length, _ = torch_context.shape

        self.prepare_transformers()

        logger.info("preparing timesteps...")

        self._scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            mu=_calculate_shift(latents_sequence_length, self._scheduler),
        )
        sigmas = self._scheduler.sigmas.tolist()
        for solver in self._solvers:
            solver.set_schedule(sigmas)
        timesteps = self._scheduler.timesteps

        logger.info("preparing latents...")

        p = self._patch_size
        img_shapes = [[(1, latents_height // p, latents_width // p)]] * transformer_batch_size
        txt_seq_lens = [prompt_sequence_length] * transformer_batch_size
        torch_latents_rope, torch_prompt_rope = self._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")

        torch_latents_rope_cos = torch_latents_rope.real.repeat_interleave(2, dim=-1)
        torch_latents_rope_sin = torch_latents_rope.imag.repeat_interleave(2, dim=-1)
        torch_prompt_rope_cos = torch_prompt_rope.real.repeat_interleave(2, dim=-1)
        torch_prompt_rope_sin = torch_prompt_rope.imag.repeat_interleave(2, dim=-1)

        context = distribute_cfg(torch_context, devices=self._submesh_devices)
        latents = self._random_latents(batch_size=transformer_batch_size, seed=seed)
        latents_rope_cos = from_torch_to_devices(
            torch_latents_rope_cos, devices=self._submesh_devices, mesh_axes=[sp_axis, None]
        )
        latents_rope_sin = from_torch_to_devices(
            torch_latents_rope_sin, devices=self._submesh_devices, mesh_axes=[sp_axis, None]
        )
        prompt_rope_cos = from_torch_to_devices(torch_prompt_rope_cos, devices=self._submesh_devices)
        prompt_rope_sin = from_torch_to_devices(torch_prompt_rope_sin, devices=self._submesh_devices)

        logger.info("denoising...")

        on_event(SectionStart("denoising"))
        for i, t in enumerate(tqdm.tqdm(timesteps)):
            on_event(SectionStart(f"denoising_step_{i}"))

            for idx, (device, tracer) in enumerate(zip(self._submesh_devices, self._tracers, strict=True)):
                timestep = ttnn.full(
                    [1, 1],
                    fill_value=t,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=device,
                )

                velocity_pred = tracer(
                    cfg_enabled=self._cfg_enabled,
                    submesh_idx=idx,
                    latents=latents[idx],
                    prompt=context[idx] if i == 0 else tracer.inputs["prompt"],
                    timestep=timestep,
                    spatial_rope=(latents_rope_cos[idx], latents_rope_sin[idx])
                    if i == 0
                    else tracer.inputs["spatial_rope"],
                    prompt_rope=(prompt_rope_cos[idx], prompt_rope_sin[idx])
                    if i == 0
                    else tracer.inputs["prompt_rope"],
                    spatial_sequence_length=latents_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    traced=traced,
                )

                # latents can be overwritten by trace execution, use the captured input instead,
                # which is safe.
                latents[idx] = tracer.inputs["latents"]

                if self._cfg_enabled:
                    velocity_pred = self._cfg_combiner.combine(velocity_pred, cfg_scale)

                latents[idx] = self._solvers[idx].step(step=i, latent=latents[idx], velocity_pred=velocity_pred)

            self.synchronize_devices()

            on_event(SectionEnd(f"denoising_step_{i}"))
        on_event(SectionEnd("denoising"))

        logger.info("decoding image...")

        on_event(SectionStart("vae"))
        output = self._decode_latents(
            latents[self.vae_submesh_idx],
            latents_height=latents_height,
            latents_width=latents_width,
            traced=vae_traced,
        )
        on_event(SectionEnd("vae"))
        on_event(SectionEnd("total"))

        return output

    def _traced_step(self, *, cfg_enabled: bool, submesh_idx: int, latents: ttnn.Tensor, **kwargs: Any) -> ttnn.Tensor:
        if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1:
            latents = ttnn.concat([latents, latents])

        return self.transformers[submesh_idx].forward(spatial=latents, **kwargs)

    def synchronize_devices(self) -> None:
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)

    def _random_latents(self, *, batch_size: int, seed: int) -> list[ttnn.Tensor]:
        torch.manual_seed(seed)
        shape = [
            batch_size,
            self._num_channels_latents,
            self._height // self._vae_scale_factor,
            self._width // self._vae_scale_factor,
        ]
        # We let randn generate a permuted latent tensor in float32, so that the generated noise
        # matches the reference implementation.
        latents = self.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        return from_torch_to_devices(latents, devices=self._submesh_devices, mesh_axes=[None, sp_axis, None])

    def _decode_latents(
        self,
        tt_latents: ttnn.Tensor,
        *,
        latents_height: int,
        latents_width: int,
        traced: bool,
    ) -> list[Image.Image]:
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
        torch_latents = self.transformers[0].unpatchify(
            torch_latents,
            height=latents_height,
            width=latents_width,
        )

        if not self._use_torch_vae_decoder:
            self.prepare_vae()
        with self._reshape_vae():
            decoded_output = self._vae.decode(torch_latents, traced=traced)

        image = self._image_processor.postprocess(decoded_output, output_type="pt")
        assert isinstance(image, torch.Tensor)
        return self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))


def _calculate_shift(image_seq_len: int, scheduler: FlowMatchEulerDiscreteScheduler) -> float:
    base_seq_len = scheduler.config.get("base_image_seq_len", 256)
    max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
