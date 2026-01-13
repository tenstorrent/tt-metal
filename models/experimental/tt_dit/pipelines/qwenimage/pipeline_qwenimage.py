# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffusers
import numpy as np
import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from models.perf.benchmarking_utils import BenchmarkProfiler

from ...encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair
from ...models.transformers.transformer_qwenimage import QwenImageTransformer
from ...models.vae.vae_qwenimage import QwenImageVaeDecoder
from ...parallel.config import (
    DiTParallelConfig,
    EncoderParallelConfig,
    ParallelFactor,
    VAEParallelConfig,
    VaeHWParallelConfig,
)
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig

if TYPE_CHECKING:
    pass

    from PIL import Image

PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
PROMPT_DROP_IDX = 34


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor


class QwenImagePipeline:
    """
    QwenImagePipeline is a pipeline for generating images from text prompts.
    It uses a transformer to encode the text prompts and a VAE to decode the latent space.
    Dynamic loading is controlled by the initialization state. During inference, modules will be loaded/offloaded as needed.
    """

    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = "Qwen/Qwen-Image",
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        parallel_config: DiTParallelConfig,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
        is_fsdp: bool = False,  # This only appies to the transformer model.
        dynamic_load_encoder: bool = True,  # Set to true if it wouldn't fit with the transformer given the configuration
        dynamic_load_vae: bool = False,  # Set to true if it wouldn't fit with the transformer given the configuration
    ) -> None:
        if dynamic_load_encoder or dynamic_load_vae:
            assert (
                cache.cache_dir_is_set()
            ), "Dynamic loading of encoder or vae is enabled but the cache directory (env variable TT_DIT_CACHE_DIR) is not set."

        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._height = height
        self._width = width
        self._is_fsdp = is_fsdp

        # Create submeshes based on CFG parallel configuration
        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
        submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")

        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))[
            0 : parallel_config.cfg_parallel.factor
        ]
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = len(self._submesh_devices) - self.encoder_submesh_idx - 1  # Use other submesh for VAE. 0

        self.encoder_device = self._submesh_devices[self.encoder_submesh_idx] if not use_torch_text_encoder else None
        self.vae_device = self._submesh_devices[self.vae_submesh_idx] if not use_torch_vae_decoder else None

        # setup parallel configs
        self._encoder_parallel_config = encoder_parallel_config
        self._vae_parallel_config = vae_parallel_config
        self._wan_vae_parallel_config = self.get_wan_vae_parallel_config()

        self.encoder_mesh_shape = self.get_mesh_shape(
            self.encoder_device, self._encoder_parallel_config.tensor_parallel
        )
        self.vae_mesh_shape = self.get_mesh_shape(self.vae_device, self._vae_parallel_config.tensor_parallel)

        logger.info("loading models...")

        torch_transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()

        self._torch_vae = AutoencoderKLQwenImage.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKLQwenImage)
        # Store VAE state dict for loading/reloading
        self._vae_state_dict = self._torch_vae.state_dict()

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")

        self._num_channels_latents = 16
        self._patch_size = torch_transformer.config.patch_size
        self._vae_scale_factor = 8

        if torch_transformer.config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                torch_transformer.config.num_attention_heads,
                torch_transformer.config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self._transformer_state_dict = torch_transformer.state_dict()
        self._padding_config = padding_config
        self._pos_embed = torch_transformer.pos_embed

        # Initialize the transformers. Loading logic comes after.
        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            self.transformers.append(
                QwenImageTransformer(
                    patch_size=torch_transformer.config.patch_size,
                    in_channels=torch_transformer.config.in_channels,
                    num_layers=torch_transformer.config.num_layers,
                    attention_head_dim=torch_transformer.config.attention_head_dim,
                    num_attention_heads=torch_transformer.config.num_attention_heads,
                    joint_attention_dim=torch_transformer.config.joint_attention_dim,
                    out_channels=torch_transformer.config.out_channels,
                    device=submesh_device,
                    ccl_manager=self._ccl_managers[i],
                    parallel_config=self._parallel_config,
                    padding_config=self._padding_config,
                    is_fsdp=self._is_fsdp,
                )
            )
        self._transformers_loaded = False

        # initialize text encoder. This will load the weights
        self._use_torch_text_encoder = use_torch_text_encoder
        with self.mesh_reshape(self.encoder_device, self.encoder_mesh_shape, synchronize=True):
            logger.info("creating TT-NN text encoder (loading before transformers for memory efficiency)...")
            self._text_encoder = Qwen25VlTokenizerEncoderPair(
                checkpoint_name,
                tokenizer_subfolder="tokenizer",
                encoder_subfolder="text_encoder",
                device=self._submesh_devices[self.encoder_submesh_idx],
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=self._encoder_parallel_config,
                use_torch=use_torch_text_encoder,
                is_fsdp=True,  # Best configuration for wh t3k and galaxy
            )

        # Encoder is already loaded. Decide if we should also load the transformers.
        if (
            not dynamic_load_encoder or use_torch_text_encoder
        ):  # Implies we have enough space. VAE comes after denoising, so load all transformers now.
            self._load_transformers(self.encoder_submesh_idx)

        # Always load transformers for vae since it comes before VAE
        self._load_transformers(self.vae_submesh_idx)

        self._latents_scaling = 1.0 / torch.tensor(self._torch_vae.config.latents_std)
        self._latents_shift = torch.tensor(self._torch_vae.config.latents_mean)

        self._image_processor = VaeImageProcessor(vae_scale_factor=2 * self._vae_scale_factor)

        self._use_torch_vae_decoder = use_torch_vae_decoder

        if use_torch_vae_decoder:
            self._vae_decoder = None
        else:
            with self.mesh_reshape(self.vae_device, self.vae_mesh_shape, synchronize=True):
                logger.info("creating TT-NN VAE decoder...")
                self._vae_decoder = QwenImageVaeDecoder(
                    base_dim=self._torch_vae.config.base_dim,
                    z_dim=self._torch_vae.config.z_dim,
                    dim_mult=self._torch_vae.config.dim_mult,
                    num_res_blocks=self._torch_vae.config.num_res_blocks,
                    temperal_downsample=self._torch_vae.config.temperal_downsample,
                    device=self.vae_device,
                    parallel_config=self._wan_vae_parallel_config,
                    ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                )

            # Load VAE weights based on configuration
            if not dynamic_load_vae:
                self._vae_decoder.load_torch_state_dict(self._vae_state_dict)

        self._traces = None

    def _load_transformers(self, idx) -> None:
        """Load transformer weights to device. Called lazily for device encoder path."""
        if self.transformers[idx].is_loaded():
            return

        if not cache.initialize_from_cache(
            tt_model=self.transformers[idx],
            torch_state_dict=self._transformer_state_dict,
            model_name="qwen-image",
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._submesh_devices[idx].shape),
            dtype="bf16",
            is_fsdp=self._is_fsdp,
        ):
            logger.info("Loading transformer weights from PyTorch state dict")
            self.transformers[idx].load_torch_state_dict(self._transformer_state_dict)

        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_transformers(self, idx) -> None:
        """Deallocate transformer weights from device to free memory."""
        if not self.transformers[idx].is_loaded():
            return

        logger.info("deallocating transformer weights to free memory...")
        self.transformers[idx].deallocate_weights()
        ttnn.synchronize_device(self._submesh_devices[idx])

    def _deallocate_vae(self) -> None:
        """Deallocate VAE decoder weights from device to free memory."""
        if self._use_torch_vae_decoder or not self._vae_decoder.is_loaded():
            return

        logger.info("deallocating VAE decoder weights to free memory...")
        self._vae_decoder.deallocate_weights()
        ttnn.synchronize_device(self.vae_device)

    def _reload_vae(self) -> None:
        """Load or reload VAE decoder weights to device."""
        if self._use_torch_vae_decoder or self._vae_decoder.is_loaded():
            return

        with self.mesh_reshape(self.vae_device, self.vae_mesh_shape, synchronize=True):
            logger.info("loading VAE decoder weights to device...")
            self._vae_decoder.load_torch_state_dict(self._vae_state_dict)

    @contextmanager
    def mesh_reshape(
        self, device: ttnn.MeshDevice | None, mesh_shape: ttnn.MeshShape | None, synchronize: bool = False
    ):
        if device is None:
            yield
        else:
            original_mesh_shape = ttnn.MeshShape(tuple(device.shape))
            assert (
                original_mesh_shape.mesh_size() == mesh_shape.mesh_size()
            ), f"Device cannot be reshaped device shape: {device.shape} mesh shape: {mesh_shape}"
            if original_mesh_shape != mesh_shape:
                device.reshape(mesh_shape)
            yield
            if original_mesh_shape != device.shape:
                device.reshape(original_mesh_shape)
            if synchronize:
                ttnn.synchronize_device(device)

    @staticmethod
    def get_mesh_shape(mesh_device, parallel_factor):
        if mesh_device is None:
            return None
        mesh_shape = list(mesh_device.shape)
        mesh_shape[parallel_factor.mesh_axis] = parallel_factor.factor
        mesh_shape[1 - parallel_factor.mesh_axis] = mesh_device.shape.mesh_size() // parallel_factor.factor
        return ttnn.MeshShape(tuple(mesh_shape))

    # TODO: Configure the correct parallel config
    def get_wan_vae_parallel_config(self):
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

    def run_single_prompt(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 4.0,
        seed: int = 0,
        traced: bool = True,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        """Run inference for a single prompt. Convenience method for inference server."""
        return self(
            prompts=[prompt],
            negative_prompts=[negative_prompt],
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            traced=traced,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
        )

    @staticmethod
    def create_pipeline(
        *,
        mesh_device: ttnn.MeshDevice,
        dit_cfg: tuple[int, int] | None = None,
        dit_sp: tuple[int, int] | None = None,
        dit_tp: tuple[int, int] | None = None,
        encoder_tp: tuple[int, int] | None = None,
        vae_tp: tuple[int, int] | None = None,
        use_torch_text_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        num_links: int,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
        is_fsdp: bool = None,
        dynamic_load_encoder: bool | None = None,
        dynamic_load_vae: bool | None = None,
    ) -> QwenImagePipeline:
        default_config = {
            # The default cofigurations are the best found from sweeping the following: is_fsdp, dynamic_load_encoder, and dynamic_load_vae.
            # The encoder is currently hardcoded to always be FSDP as it is the most memory efficient configuration with little to no performance penalty.
            (2, 4): {
                "cfg_config": (2, 1),
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
                "cfg_config": (2, 1),
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
        cfg_factor, cfg_axis = dit_cfg or default_config[tuple(mesh_device.shape)]["cfg_config"]
        sp_factor, sp_axis = dit_sp or default_config[tuple(mesh_device.shape)]["sp"]
        tp_factor, tp_axis = dit_tp or default_config[tuple(mesh_device.shape)]["tp"]
        encoder_tp_factor, encoder_tp_axis = encoder_tp or default_config[tuple(mesh_device.shape)]["encoder_tp"]
        vae_tp_factor, vae_tp_axis = vae_tp or default_config[tuple(mesh_device.shape)]["vae_tp"]
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]
        is_fsdp = is_fsdp or default_config[tuple(mesh_device.shape)]["is_fsdp"]
        dynamic_load_encoder = dynamic_load_encoder or default_config[tuple(mesh_device.shape)]["dynamic_load_encoder"]
        dynamic_load_vae = dynamic_load_vae or default_config[tuple(mesh_device.shape)]["dynamic_load_vae"]

        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
        )
        vae_parallel_config = VAEParallelConfig(
            tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
        )

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {dit_parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel config: {vae_parallel_config}")

        return QwenImagePipeline(
            mesh_device=mesh_device,
            use_torch_text_encoder=use_torch_text_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
            is_fsdp=is_fsdp,
            dynamic_load_encoder=dynamic_load_encoder,
            dynamic_load_vae=dynamic_load_vae,
        )

    def prepare_encoder(self) -> None:
        """Prepare encoder for inference."""
        if not self._text_encoder.encoder_loaded():
            self._deallocate_transformers(self.encoder_submesh_idx)
            with self.mesh_reshape(self.encoder_device, self.encoder_mesh_shape):
                self._text_encoder.reload_encoder_weights()

    def prepare_transformers(self) -> None:
        if not self.transformers[self.encoder_submesh_idx].is_loaded():
            self._text_encoder.deallocate_encoder_weights()
            self._load_transformers(self.encoder_submesh_idx)

        if not self.transformers[self.vae_submesh_idx].is_loaded():
            self._deallocate_vae()
            self._load_transformers(self.vae_submesh_idx)

    def prepare_vae(self) -> None:
        if not self._vae_decoder.is_loaded():
            self._deallocate_transformers(self.vae_submesh_idx)
            with self.mesh_reshape(self.vae_device, self.vae_mesh_shape):
                self._reload_vae()

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float,
        prompts: list[str],
        negative_prompts: list[str | None],
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        cfg_factor = self._parallel_config.cfg_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        spatial_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        with profiler("total", profiler_iteration) if profiler else nullcontext():
            cfg_enabled = cfg_scale > 1
            logger.info("encoding prompts...")

            self.prepare_encoder()

            with profiler("encoder", profiler_iteration) if profiler else nullcontext():
                with self.mesh_reshape(self.encoder_device, self.encoder_mesh_shape):
                    prompt_embeds, prompt_mask = self._encode_prompts(
                        prompts=prompts,
                        negative_prompts=negative_prompts,
                        num_images_per_prompt=num_images_per_prompt,
                        cfg_enabled=cfg_enabled,
                        profiler=profiler,
                        profiler_iteration=profiler_iteration,
                    )
            _, prompt_sequence_length, _ = prompt_embeds.shape

            self.prepare_transformers()

            logger.info("preparing timesteps...")
            timesteps, sigmas = _schedule(
                self._scheduler,
                step_count=num_inference_steps,
                spatial_sequence_length=spatial_sequence_length,
            )

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            shape = [
                transformer_batch_size,
                self._num_channels_latents,
                self._height // self._vae_scale_factor,
                self._width // self._vae_scale_factor,
            ]
            # We let randn generate a permuted latent tensor in float32, so that the generated noise
            # matches the reference implementation.
            latents = self.transformers[0].patchify(torch.randn(shape).permute(0, 2, 3, 1))

            p = self._patch_size
            img_shapes = [[(1, latents_height // p, latents_width // p)]] * transformer_batch_size
            txt_seq_lens = [prompt_sequence_length] * transformer_batch_size
            spatial_rope, prompt_rope = self._pos_embed.forward(img_shapes, txt_seq_lens, "cpu")

            spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
            spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
            prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
            prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

            tt_prompt_embeds_device_list = []
            tt_prompt_embeds_list = []
            tt_latents_step_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []
            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds_device = tensor.from_torch(
                    prompt_embeds[i : i + 1] if cfg_factor == 2 else prompt_embeds,
                    device=submesh_device,
                    on_host=traced,
                )
                tt_prompt_embeds = tensor.from_torch(
                    prompt_embeds[i : i + 1] if cfg_factor == 2 else prompt_embeds,
                    device=submesh_device,
                    on_host=True,
                )

                tt_initial_latents = tensor.from_torch(
                    latents, device=submesh_device, on_host=traced, mesh_axes=[None, sp_axis, None]
                )

                tt_spatial_rope_cos = tensor.from_torch(
                    spatial_rope_cos, device=submesh_device, on_host=traced, mesh_axes=[sp_axis, None]
                )
                tt_spatial_rope_sin = tensor.from_torch(
                    spatial_rope_sin, device=submesh_device, on_host=traced, mesh_axes=[sp_axis, None]
                )
                tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=submesh_device, on_host=traced)
                tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=submesh_device, on_host=traced)

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds_device = tt_prompt_embeds_device.to(submesh_device)
                        tt_spatial_rope_cos = tt_spatial_rope_cos.to(submesh_device)
                        tt_spatial_rope_sin = tt_spatial_rope_sin.to(submesh_device)
                        tt_prompt_rope_cos = tt_prompt_rope_cos.to(submesh_device)
                        tt_prompt_rope_sin = tt_prompt_rope_sin.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds_device, self._traces[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_cos, self._traces[i].spatial_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_sin, self._traces[i].spatial_rope_sin)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_cos, self._traces[i].prompt_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_sin, self._traces[i].prompt_rope_sin)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds_device = self._traces[i].prompt_input
                        tt_spatial_rope_cos = self._traces[i].spatial_rope_cos
                        tt_spatial_rope_sin = self._traces[i].spatial_rope_sin
                        tt_prompt_rope_cos = self._traces[i].prompt_rope_cos
                        tt_prompt_rope_sin = self._traces[i].prompt_rope_sin

                tt_prompt_embeds_device_list.append(tt_prompt_embeds_device)
                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)
                tt_spatial_rope_cos_list.append(tt_spatial_rope_cos)
                tt_spatial_rope_sin_list.append(tt_spatial_rope_sin)
                tt_prompt_rope_cos_list.append(tt_prompt_rope_cos)
                tt_prompt_rope_sin_list.append(tt_prompt_rope_sin)

            logger.info("denoising...")

            with profiler("denoising", profiler_iteration) if profiler else nullcontext():
                for i, t in enumerate(tqdm.tqdm(timesteps)):
                    with profiler(f"denoising_step_{i}", profiler_iteration) if profiler else nullcontext():
                        sigma_difference = sigmas[i + 1] - sigmas[i]

                        tt_timestep_list = []
                        tt_sigma_difference_list = []
                        for submesh_nr, submesh_device in enumerate(self._submesh_devices):
                            tt_timestep = ttnn.full(
                                [1, 1],
                                fill_value=t,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.float32,
                                device=submesh_device if not traced else None,
                            )
                            tt_timestep_list.append(tt_timestep)

                            tt_sigma_difference = ttnn.full(
                                [1, 1],
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=submesh_device if not traced else None,
                            )
                            tt_sigma_difference_list.append(tt_sigma_difference)

                            # TODO: move out of the loop
                            ttnn.copy_host_to_device_tensor(
                                tt_prompt_embeds_list[submesh_nr],
                                tt_prompt_embeds_device_list[submesh_nr],
                            )

                        tt_latents_step_list = self._step(
                            timestep=tt_timestep_list,
                            latents=tt_latents_step_list,
                            cfg_enabled=cfg_enabled,
                            prompt_embeds=tt_prompt_embeds_device_list,
                            cfg_scale=cfg_scale,
                            sigma_difference=tt_sigma_difference_list,
                            spatial_rope_cos=tt_spatial_rope_cos_list,
                            spatial_rope_sin=tt_spatial_rope_sin_list,
                            prompt_rope_cos=tt_prompt_rope_cos_list,
                            prompt_rope_sin=tt_prompt_rope_sin_list,
                            spatial_sequence_length=spatial_sequence_length,
                            prompt_sequence_length=prompt_sequence_length,
                            traced=traced,
                            profiler=profiler,
                            profiler_iteration=profiler_iteration,
                        )

            logger.info("decoding image...")

            with profiler("vae", profiler_iteration) if profiler else nullcontext():
                # Sync because we don't pass a persistent buffer or a barrier semaphore.
                ttnn.synchronize_device(self.vae_device)

                tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                    tt_latents_step_list[self.vae_submesh_idx],
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

                torch_latents = torch_latents / self._latents_scaling + self._latents_shift

                if self._vae_decoder is None:
                    torch_latents = torch_latents.permute(0, 3, 1, 2).unsqueeze(2)
                    with torch.no_grad():
                        decoded_output = self._torch_vae.decode(torch_latents).sample[:, :, 0]
                else:
                    self.prepare_vae()

                    with self.mesh_reshape(self.vae_device, self.vae_mesh_shape):
                        tt_latents, logical_h = self._vae_decoder.prepare_input(torch_latents)
                        tt_decoded_output, logical_h = self._vae_decoder.forward(tt_latents, logical_h)
                        decoded_output = self._vae_decoder.postprocess_output(tt_decoded_output, logical_h)

                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        submesh_index: int,
        spatial_rope_cos: ttnn.Tensor,
        spatial_rope_sin: ttnn.Tensor,
        prompt_rope_cos: ttnn.Tensor,
        prompt_rope_sin: ttnn.Tensor,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1:
            latent = ttnn.concat([latent, latent])

        return self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            timestep=timestep,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

    def _step(
        self,
        *,
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],  # device tensor
        timestep: list[ttnn.Tensor],  # host tensor
        prompt_embeds: list[ttnn.Tensor],  # device tensor
        sigma_difference: list[ttnn.Tensor],  # device tensor
        spatial_rope_cos: list[ttnn.Tensor],
        spatial_rope_sin: list[ttnn.Tensor],
        prompt_rope_cos: list[ttnn.Tensor],
        prompt_rope_sin: list[ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        traced: bool,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[ttnn.Tensor]:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        if traced and self._traces is None:
            self._traces = []
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                timestep_device = timestep[submesh_id].to(submesh_device)
                sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)

                # Warmup run before trace capture
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )

                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

                for device in self._submesh_devices:
                    ttnn.synchronize_device(device)

                self._traces.append(
                    PipelineTrace(
                        spatial_input=latents[submesh_id],
                        prompt_input=prompt_embeds[submesh_id],
                        timestep_input=timestep_device,
                        spatial_rope_cos=spatial_rope_cos[submesh_id],
                        spatial_rope_sin=spatial_rope_sin[submesh_id],
                        prompt_rope_cos=prompt_rope_cos[submesh_id],
                        prompt_rope_sin=prompt_rope_sin[submesh_id],
                        latents_output=pred,
                        sigma_difference_input=sigma_difference_device,
                        tid=trace_id,
                    )
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
                ttnn.copy_host_to_device_tensor(
                    sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input
                )
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)

            # TODO: If we don't do this, we get noise when tracing is enabled. But why, since sigma
            # difference is only used outside of tracing region?
            sigma_difference_device = [trace.sigma_difference_input for trace in self._traces]
        else:
            for submesh_id in range(len(self._submesh_devices)):
                noise_pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                noise_pred_list.append(noise_pred)

            sigma_difference_device = sigma_difference

        # CFG combine
        # NOTE: With cfg_parallel.factor > 1, the .cpu(blocking=True) call is the sync point
        # where the actual denoising compute happens. This is NOT wasted time - it's the
        # actual forward pass execution. The 1.3s/step is the real denoising time.
        if cfg_enabled:
            if self._parallel_config.cfg_parallel.factor == 1:
                split_pos = noise_pred_list[0].shape[0] // 2
                uncond = noise_pred_list[0][0:split_pos]
                cond = noise_pred_list[0][split_pos:]
                noise_pred_list[0] = uncond + cfg_scale * (cond - uncond)
            else:
                # With CFG parallel > 1 and SP > 1, noise predictions are sharded across SP axis.
                # We need to all-gather to get full sequence, do CFG, then re-shard.
                # The .cpu(blocking=True) is the sync point where actual compute happens.
                uncond = tensor.to_torch(
                    noise_pred_list[0].cpu(blocking=True),
                    mesh_axes=[None, sp_axis, None],
                    composer_device=self._submesh_devices[0],
                ).to(torch.float32)
                cond = tensor.to_torch(
                    noise_pred_list[1].cpu(blocking=True),
                    mesh_axes=[None, sp_axis, None],
                    composer_device=self._submesh_devices[1],
                ).to(torch.float32)

                torch_noise_pred = uncond + cfg_scale * (cond - uncond)

                # Re-shard the CFG result back to the submeshes with the same sharding as latents
                noise_pred_list[0] = tensor.from_torch(
                    torch_noise_pred, device=self._submesh_devices[0], mesh_axes=[None, sp_axis, None]
                )

                noise_pred_list[1] = tensor.from_torch(
                    torch_noise_pred, device=self._submesh_devices[1], mesh_axes=[None, sp_axis, None]
                )

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)
            ttnn.multiply_(noise_pred_list[submesh_id], sigma_difference_device[submesh_id])
            ttnn.add_(latents[submesh_id], noise_pred_list[submesh_id])

        return latents

    def _encode_prompts(
        self,
        *,
        prompts: list[str],
        negative_prompts: list[str | None],
        num_images_per_prompt: int,
        cfg_enabled: bool,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(prompts) == len(negative_prompts), "prompts and negative_prompts must have the same length"

        # TODO: necessary?
        negative_prompts = [x if x is not None else "" for x in negative_prompts]

        if cfg_enabled:
            prompts = negative_prompts + prompts

        prompts = [PROMPT_TEMPLATE.format(e) for e in prompts]

        embeds, mask = self._text_encoder.encode(
            prompts,
            num_images_per_prompt=num_images_per_prompt,
            sequence_length=512 + PROMPT_DROP_IDX,
        )

        embeds[torch.logical_not(mask)] = 0.0

        return embeds[:, PROMPT_DROP_IDX:], mask[:, PROMPT_DROP_IDX:]

    def synchronize_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)


def _schedule(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    step_count: int,
    spatial_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1 / step_count, step_count),
        mu=_calculate_shift(
            spatial_sequence_length,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        ),
    )

    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    assert isinstance(timesteps, torch.Tensor)
    assert isinstance(sigmas, torch.Tensor)

    return timesteps, sigmas


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b
